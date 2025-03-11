import copy
from typing import Dict, List, Tuple
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import os
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from permutation_task import PermutationTask, PermutationState, compute_parity
from utils.model_utils import setup_tokenizer, setup_model

sns.set_style("darkgrid")

plt.rcParams['font.family'] = 'Times New Roman'


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--num_items", type=int, default=3, choices=[3, 5], help="Number of items (3 or 5)")
    parser.add_argument("--random_baseline", action="store_true", default=False)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--num_stories", type=int, default=10000)
    parser.add_argument("--skip_layers", type=str, default=None, help="comma separated list of layers to skip, e.g. 0,1,2")
    parser.add_argument("--use_bfloat16", action="store_true", default=False, help="If true, uses bfloat16")
    return parser.parse_args()


def setup_task(args):
    """Set up the permutation task based on the arguments."""
    task = PermutationTask(num_items=args.num_items)
    
    # Extract tokens and actions
    state_to_nl = {state.permutation: state.to_string() for state in task.states}
    action_to_nl = {action.permutation: action.to_string() for action in task.actions}
    
    return task, state_to_nl, action_to_nl


def load_stories(args):
    """
    Load stories from data directory or create empty ones.
    
    Args:
        args: Command line arguments
        
    Returns:
        stories: List of story dictionaries
    """
    if args.data_dir is None:
        task = PermutationTask(num_items=args.num_items)
        stories = [{
            "story": "",
            'state_seq': [list(range(1, args.num_items+1))],
        } for _ in range(args.num_stories)]
    else:
        stories = []
        for filename in tqdm(os.listdir(args.data_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(args.data_dir, filename), 'r') as f:
                    data = json.load(f)
                    stories.append(data)
                    if len(stories) > args.num_stories:
                        break
    return stories


def setup_model_and_tokenizer(args, state_to_nl, action_to_nl):
    """
    Set up model and tokenizer based on arguments.
    
    Args:
        args: Command line arguments
        state_to_nl: Dictionary of state tokens
        action_to_nl: Dictionary of action tokens
        
    Returns:
        model: The language model
        tokenizer: The tokenizer
        model_name: Name of the model
    """    
    tokenizer = setup_tokenizer(
        model_name=args.checkpoint_dir,
        state_tokens=state_to_nl,
        action_tokens=action_to_nl,
    )
    
    model = setup_model(
        tokenizer=tokenizer,
        checkpoint_path=args.checkpoint_dir if hasattr(args, 'checkpoint_dir') and args.checkpoint_dir else None,
        use_bfloat16=args.use_bfloat16 if hasattr(args, 'use_bfloat16') else False
    )
    
    # Skip layers if specified
    if args.skip_layers:
        skip_layers = [int(layer) for layer in args.skip_layers.split(",")]
        for layer in skip_layers:
            # Skip both attention and MLP by making them identity functions
            def skip_forward(self, *args, **kwargs):
                # For attention layer, return the first argument (hidden_states) and empty tuples for other expected outputs
                if isinstance(self, GPT2Attention):
                    # Return hidden states, present (None), and attentions (None if not requested)
                    outputs = (args[0], None)
                    if kwargs.get('output_attentions', False):
                        outputs = outputs + (None,)
                    return outputs
                # For MLP layer, just return the input as-is
                return args[0]
            
            # Bind the skip_forward function to both attention and MLP layers
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                model.transformer.h[layer].attn.forward = skip_forward.__get__(model.transformer.h[layer].attn)
                model.transformer.h[layer].mlp.forward = skip_forward.__get__(model.transformer.h[layer].mlp)
                print(f"Skipping layer {layer} (attention and MLP)")
    
    return model, tokenizer


def prepare_batch_data(batch_stories, batch_labels, tokenizer, device, max_len):
    """
    Prepare batch data for model inference.
    
    Args:
        batch_stories: List of story strings
        batch_labels: List of label strings
        tokenizer: The tokenizer
        device: Device to run inference on
        max_len: Maximum sequence length
        
    Returns:
        Dictionary containing model inputs and processed data
    """
    inputs = tokenizer(batch_stories, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].bool().to(device)
    
    labels = tokenizer(
        batch_labels, return_tensors='pt', padding=True, truncation=True, max_length=max_len+1,
    )['input_ids']
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def process_model_outputs(outputs, batch_data, action_to_nl, tokenizer, device):
    """
    Process model outputs to get predictions and probabilities.
    
    Args:
        outputs: Model outputs
        batch_data: Batch data dictionary
        action_to_nl: Dictionary of action tokens
        tokenizer: The tokenizer
        device: Device to run inference on
        
    Returns:
        Dictionary containing processed outputs
    """
    loss = outputs.loss
    logits = outputs.logits
    # Apply action token masking
    action_values = list(action_to_nl.values())
    action_token_ids = [token for action in action_values for token in tokenizer.encode(action) + tokenizer.encode(f" {action}")]
    
    mask = torch.zeros_like(logits)
    mask[:, :, action_token_ids] = 1
    logits = logits + (1 - mask) * -1e9  # set non-action tokens to -inf
    
    pred_tokens = logits.argmax(dim=-1).cpu()
    if (batch_data['input_ids'][:,0] == tokenizer.bos_token_id).all():
        pred_tokens = pred_tokens[:,1:]
    
    output_probs = torch.softmax(outputs.logits, dim=-1).cpu()
    label_tok_probs = output_probs.gather(2, batch_data['labels'].unsqueeze(-1)).squeeze(-1)
    
    return {
        'pred_tokens': pred_tokens,
        'label_tok_probs': label_tok_probs,
        'output_probs': output_probs,
        'loss': loss
    }


def process_predictions(batch_pred_tokens, batch_labels, batch_stories, task, tokenizer, validity_data):
    """
    Process model predictions and compute validity metrics.
    
    Args:
        batch_pred_tokens: Predicted tokens
        batch_labels: Ground truth labels
        batch_stories: Input stories
        task: PermutationTask instance
        tokenizer: Tokenizer
        validity_data: Dictionary to store validity metrics
        
    Returns:
        validity_data: Updated validity metrics
    """
    batch_size = len(batch_pred_tokens)
    
    for i in range(batch_size):
        # Get the predicted and ground truth tokens
        pred_token_ids = batch_pred_tokens[i]
        label_tokens = batch_labels[i].strip().split()
        assert len(pred_token_ids) == len(label_tokens)
        
        for seq_len in range(len(pred_token_ids)):
            # Convert token IDs to strings
            pred_perm = task.nl_to_action[tokenizer.decode(pred_token_ids[seq_len]).strip()]
            label_perm = task.nl_to_action[label_tokens[seq_len]]
            
            # Compute parity
            pred_parity = compute_parity(pred_perm)
            label_parity = compute_parity(label_perm)
            
            # Update validity data
            if seq_len+1 not in validity_data["validity"]:
                validity_data["validity"][seq_len+1] = []
                validity_data["parity_validity"][seq_len+1] = []
            
            validity_data["validity"][seq_len+1].append(float(pred_perm == label_perm))
            validity_data["parity_validity"][seq_len+1].append(float(pred_parity == label_parity))
    
    return validity_data


def prepare_story_and_states(test_item, state_to_nl, max_len, num_items):
    """
    Prepare a story and its corresponding states for evaluation.
    
    Args:
        test_item: Test item from the dataset
        args: Command-line arguments
        task: PermutationTask instance
        
    Returns:
        story: Story text
        states: List of states
    """
    story = test_item['story']
    states = [state_to_nl[tuple(state)] for state in test_item['state_seq'][1:max_len+1]]
    
    # Extend story if needed
    if len(story.split()) < max_len:
        curr_task = PermutationTask(num_items=num_items)
        curr_task.current_state = PermutationState(permutation=test_item['state_seq'][-1])
        
        while len(story.split()) < max_len:
            action = curr_task.choose_random_action()
            story += f" {curr_task.action_to_natural_language(action)}"
            curr_task.update_state(action)
            states.append(state_to_nl[tuple(curr_task.current_state.permutation)])
    else:
        story = " ".join(story.split()[:max_len])
    
    states = " ".join(states)
    return story.strip(), states


def plot_validity_results(validity_across_seqlens, parity_validity_across_seqlens, args):
    """
    Plot validity results across sequence lengths.
    
    Args:
        validity_across_seqlens: Dictionary of validity scores by sequence length
        parity_validity_across_seqlens: Dictionary of parity validity scores by sequence length
        args: Command line arguments
    """
    # See where validity first drops below threshold
    threshold = 0.98
    cutoff = None
    for k, v in validity_across_seqlens.items():
        validity_across_seqlens[k] = np.mean(v)
        if np.mean(v) < threshold:
            cutoff = k
            print(f"Validity first drops below {threshold} at sequence length {k}")
            break
    parity_cutoff = None
    for k, v in parity_validity_across_seqlens.items():
        parity_validity_across_seqlens[k] = np.mean(v)
        if np.mean(v) < threshold:
            parity_cutoff = k
            print(f"Parity Validity first drops below {threshold} at sequence length {k}")
            break
    
    # Create figure with smaller dimensions and set seaborn style
    plt.figure(figsize=(4.5, 5))  # Reduced from default size
    TEXT_FONTSIZE = 16
    
    # plot with seaborn color palette
    palette = sns.color_palette()
    seqlen = sorted(list(validity_across_seqlens.keys()))
    plt.plot(seqlen, [np.mean(validity_across_seqlens[k]) for k in seqlen], 
             color=palette[0], label="State", linewidth=2)
    plt.plot(seqlen, [np.mean(parity_validity_across_seqlens[k]) for k in seqlen], 
             color=palette[1], label="Parity", linewidth=2)
    
    # Add vertical lines with text annotations
    if cutoff and parity_cutoff:
        if cutoff == parity_cutoff:
            # If cutoffs are identical, create a single line with alternating colors
            segments = 30  # Number of segments to create
            segment_height = 1.0 / segments
            for i in range(segments):
                ymin = i * segment_height
                ymax = (i + 1) * segment_height
                color = palette[0] if i % 2 == 0 else palette[1]
                plt.axvline(x=cutoff, color=color, linestyle='--', ymin=ymin, ymax=ymax, alpha=0.7)
            
            plt.text(cutoff - 12, 0.25, f"State cutoff: {cutoff}", rotation=90, 
                     verticalalignment='center', color=palette[0], fontsize=TEXT_FONTSIZE)
            plt.text(cutoff - 12, 0.75, f"Parity cutoff: {parity_cutoff}", rotation=90, 
                     verticalalignment='center', color=palette[1], fontsize=TEXT_FONTSIZE)
        else:
            plt.axvline(x=cutoff, color=palette[0], linestyle='--', alpha=0.7)
            plt.axvline(x=parity_cutoff, color=palette[1], linestyle='--', alpha=0.7)
            if abs(cutoff - parity_cutoff) < 10:
                plt.text(cutoff - 12, 0.25, f"State cutoff: {cutoff}", rotation=90, 
                        verticalalignment='center', color=palette[0], fontsize=TEXT_FONTSIZE)
                plt.text(parity_cutoff - 12, 0.75, f"Parity cutoff: {parity_cutoff}", rotation=90, 
                        verticalalignment='center', color=palette[1], fontsize=TEXT_FONTSIZE)
            else:
                plt.text(cutoff - 12, 0.5, f"State cutoff: {cutoff}", rotation=90, 
                        verticalalignment='center', color=palette[0], fontsize=TEXT_FONTSIZE)
                plt.text(parity_cutoff - 12, 0.5, f"Parity cutoff: {parity_cutoff}", rotation=90, 
                        verticalalignment='center', color=palette[1], fontsize=TEXT_FONTSIZE)
    
    plt.xlabel("Sequence Length", fontsize=TEXT_FONTSIZE)
    plt.ylabel("Accuracy", fontsize=TEXT_FONTSIZE)
    plt.title("Accuracy Across Sequence Lengths", fontsize=TEXT_FONTSIZE, pad=10)
    plt.legend(frameon=True, fancybox=True, framealpha=0.8, fontsize=TEXT_FONTSIZE)
    plt.grid(True, alpha=0.3)
    
    # Increase tick label font sizes
    plt.xticks(fontsize=TEXT_FONTSIZE)
    plt.yticks(fontsize=TEXT_FONTSIZE)
    
    # Set axis limits to ensure consistent spacing
    plt.ylim(0, 1.05)
    plt.xlim(0, 200)
    
    figures_fn = f"figures/gen_accuracy/{args.checkpoint_dir.rstrip('/')}{'_skip_layers'+args.skip_layers if args.skip_layers else ''}.png"
    figures_dir = os.path.split(figures_fn)[0]
    os.makedirs(figures_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(figures_fn, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {figures_fn}")


def evaluate(args, model, tokenizer, stories, task, action_to_nl, state_to_nl):
    """
    Evaluate the model on the given stories.
    
    Args:
        args: Command-line arguments
        model: Model to evaluate
        tokenizer: Tokenizer
        stories: List of stories to evaluate
        task: PermutationTask instance
        action_to_nl: Dictionary mapping actions to tokens
        
    Returns:
        validity_data: Dictionary containing validity metrics
    """
    device = "cuda"
    batch_size = 4
    batch_stories = []
    batch_labels = []
    correct_state_prob = []
    
    # Initialize validity tracking data
    validity_data = {
        'validity': {},
        'parity_validity': {}
    }
    
    pbar = tqdm(enumerate(stories), desc="Evaluating", total=len(stories))
    
    for t, test_item in pbar:
        # Prepare story and states
        story, states = prepare_story_and_states(test_item, state_to_nl, args.max_len, args.num_items)
        batch_stories.append(story)
        batch_labels.append(states)

        if len(batch_stories) >= batch_size:
            with torch.no_grad():
                # Prepare batch data
                batch_data = prepare_batch_data(batch_stories, batch_labels, tokenizer, device, args.max_len)
                
                # Run model inference
                outputs = model(**batch_data)
                
                # Process model outputs
                processed_outputs = process_model_outputs(outputs, batch_data, action_to_nl, tokenizer, device)
                
                # Process predictions
                validity_data = process_predictions(
                    processed_outputs['pred_tokens'], 
                    batch_labels,
                    batch_stories,
                    task, 
                    tokenizer, 
                    validity_data
                )
                
                correct_state_prob.extend(processed_outputs['label_tok_probs'].flatten().tolist())
            
            batch_stories = []
            batch_labels = []
            seq_validities = [item for sublist in validity_data['validity'].values() for item in sublist]
            seq_parity_validities = [item for sublist in validity_data['parity_validity'].values() for item in sublist]
            pbar.set_description(
                f"Evaluating: p(correct)={np.mean(correct_state_prob):.2f}, "
                f"%valid={np.mean(seq_validities):.2f}, "
                f"%parity_valid={np.mean(seq_parity_validities):.2f}"
            )
    
    print(f"Average Correct State Probability: {np.mean(correct_state_prob)}")
    
    # Calculate averages
    validity_across_seqlens = {k: np.mean(v).item() for k, v in validity_data['validity'].items()}
    parity_validity_across_seqlens = {k: np.mean(v).item() for k, v in validity_data['parity_validity'].items()}
    
    print(f"Average Validity Across Seqlens: {validity_across_seqlens}")
    print(f"Average Parity Validity Across Seqlens: {parity_validity_across_seqlens}")
    
    return validity_data


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up task
    task, state_to_nl, action_to_nl = setup_task(args)
    
    # Load stories
    stories = load_stories(args)
    
    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args, state_to_nl, action_to_nl)
    
    # Evaluate model
    validity_data = evaluate(args, model, tokenizer, stories, task, action_to_nl, state_to_nl)
    
    # Plot results
    plot_validity_results(validity_data['validity'], validity_data['parity_validity'], args)


if __name__ == "__main__":
    main()
