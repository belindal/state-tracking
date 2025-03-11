import numpy as np
import random
from tqdm import tqdm
from typing import Set, Dict, List, Tuple, Any
from permutation_task import compute_parity
from interpret.interpreters.base_interpreter import BaseInterpreter
import torch
import os


class ActivationPatchingInterpreter(BaseInterpreter):
    """Class for activation patching analysis."""
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.intervene_output_type = kwargs.get("intervene_output_type", "state")
        self.patching_mode = kwargs.get("patching_mode", "substitution")
        self.token_increments = kwargs.get("token_increments", 1)
    
    def generate_prompt_pairs(self, prompts: Set[str], diff_parity=False) -> Dict[str, List[Tuple[str, str]]]:
        """Generate pairs of prompts for analysis"""
        if diff_parity:
            pairs = {
                "same_parity": [],
                "diff_parity": [],
            }
        else:
            pairs = {
                "all": [],
            }
        for prompt in prompts:
            item_to_replace = 0
            split_prompt = prompt.split()

            action_parity = compute_parity(self.nl_to_action[split_prompt[item_to_replace]])
            for parity_type in pairs:
                if parity_type == "same_parity" or parity_type == "all":
                    # Generate same parity pair
                    correct_parity_actions = [
                        action for action in self.action_to_nl
                        if compute_parity(action) == action_parity and action != self.nl_to_action[split_prompt[item_to_replace]]
                    ]
                    split_prompt[item_to_replace] = self.action_to_nl[random.choice(correct_parity_actions)]
                    new_prompt = " ".join(split_prompt)
                    pairs[parity_type].append((prompt, new_prompt))
                if parity_type == "diff_parity" or parity_type == "all":
                    # Generate different parity pair
                    other_parity_actions = [
                        action for action in self.action_to_nl
                        if compute_parity(action) != action_parity
                    ]
                    split_prompt[item_to_replace] = self.action_to_nl[random.choice(other_parity_actions)]
                    new_prompt = " ".join(split_prompt)
                    pairs[parity_type].append((prompt, new_prompt))
            
        return pairs

    def get_hs_logits(
        self,
        prompt,
        layer_components=None,
    ):
        """Get hidden states and logits for a given prompt"""
        hidden_states = {}
        n_layers = self.model.config.num_hidden_layers if self.model_type != "gpt2" else self.model.config.n_layer
        
        with torch.no_grad():
            with self.model.trace() as tracer:
                with tracer.invoke(prompt) as invoker:
                    # Get embeddings
                    hidden_states = {-1: {"embed": self.embeddings.output[0].cpu().save()}}
                        
                    # Get hidden states for all layers
                    for layer_idx in range(n_layers):
                        if self.model_type == "gpt2":
                            hidden_states[layer_idx] = {
                                "ln1": self.model_layers[layer_idx].ln_1.output[0].cpu().save(),
                                "attn": self.model_layers[layer_idx].attn.output[0].cpu().save(),
                                "ln2": self.model_layers[layer_idx].ln_2.output[0].cpu().save(),
                                "mlp": self.model_layers[layer_idx].mlp.output[0].cpu().save(),
                            }
                        elif self.model_type == "llama":
                            hidden_states[layer_idx] = {
                                "ln1": self.model_layers[layer_idx].input_layernorm.output[0].cpu().save(),
                                "attn": self.model_layers[layer_idx].self_attn.output[0].cpu().save(),
                                "ln2": self.model_layers[layer_idx].post_attention_layernorm.output[0].cpu().save(),
                                "mlp": self.model_layers[layer_idx].mlp.output[0].cpu().save(),
                            }
                        elif self.model_type == "pythia":
                            hidden_states[layer_idx] = {
                                "ln1": self.model_layers[layer_idx].input_layernorm.output[0].cpu().save(),
                                "attn": self.model_layers[layer_idx].attention.output[0].cpu().save(),
                                "ln2": self.model_layers[layer_idx].post_attention_layernorm.output[0].cpu().save(),
                                "mlp": self.model_layers[layer_idx].mlp.output[0].cpu().save(),
                            }
                    
                    # Get specific components if requested
                    if layer_components is not None:
                        for layer_idx, component in layer_components:
                            if component not in hidden_states[layer_idx]:
                                component_parts = component.split(".")
                                model_component = self.model_layers[layer_idx]
                                for component_part in component_parts:
                                    model_component = getattr(model_component, component_part)
                                model_component = model_component[0]
                                if component in ["output", "attn.output", "self_attn.output","attention.output"]:
                                    model_component = model_component[0]
                                hidden_states[layer_idx][component] = model_component.cpu().save()
                    
                    # Get logits from the lm_head
                    clean_logits = self.sf(self.lm_head.output)
                    clean_logits = {
                        action: clean_logits[-1,-1,self.action_to_token_ids[action]].cpu().to(dtype=float).item().save()
                        for action in self.action_to_token_ids
                    }
        
        clean_logits = {action: clean_logits[action] * 1.0 for action in clean_logits}
        return hidden_states, clean_logits

    def get_patching_results_pair(
        self,
        minimal_pair,
        layer_components,
        token_units=1,
        replace_prev_tokens=False,
        replace_next_tokens=False,
        patching_mode="substitution",
    ):
        """Get patching results for a pair of prompts"""
        # Get hidden states and logits for both prompts
        _, new_logits = self.get_hs_logits(minimal_pair[1], layer_components)
        _, base_logits = self.get_hs_logits(minimal_pair[0], layer_components)
        
        all_patching_logits = []
        prompt_tokens = self.tokenizer.tokenize(minimal_pair[0])
        
        with torch.no_grad():
            # Iterate through all the layers
            for lc_idx, (layer_idx, component) in enumerate(layer_components):
                _patching_logits = []
                
                # Iterate through tokens in chunks of token_units
                for token_idx in range(0, len(prompt_tokens), token_units):
                    # Determine token range to patch
                    if replace_prev_tokens:
                        token_range = np.arange(0, min(len(prompt_tokens), token_idx+token_units))
                    elif replace_next_tokens:
                        token_range = np.arange(token_idx, len(prompt_tokens))
                    else:
                        token_range = np.arange(token_idx, min(len(prompt_tokens), token_idx+token_units))
                    
                    # Perform patching using nnsight
                    with self.model.trace() as tracer:
                        # Get replacement values if not using deletion patching
                        if patching_mode == "substitution":
                            with tracer.invoke(minimal_pair[1]) as invoker:
                                # Get the component to replace from
                                if layer_idx == -1:
                                    replace_model_component = self.embeddings.output[0]
                                else:
                                    component_parts = component.split(".")
                                    replace_model_component = self.model_layers[layer_idx]
                                    for component_part in component_parts:
                                        replace_model_component = getattr(replace_model_component, component_part)
                                    replace_model_component = replace_model_component[0]
                                    if component in ["attn.output", "output", "attention.output", "self_attn.output"]:
                                        replace_model_component = replace_model_component[0]
                        
                        # Apply the patch to the original prompt
                        with tracer.invoke(minimal_pair[0]) as invoker:
                            # Handle embeddings layer
                            if layer_idx == -1:
                                if patching_mode == "deletion":
                                    self.embeddings.output[0][token_range] = 0
                                else:
                                    self.embeddings.output[0][token_range] = replace_model_component[token_range]
                            # Handle other layers
                            else:
                                component_parts = component.split(".")
                                model_component = self.model_layers[layer_idx]
                                for component_part in component_parts:
                                    model_component = getattr(model_component, component_part)
                                model_component = model_component[0]
                                if component in ["attn.output", "output", "attention.output", "self_attn.output"]:
                                    model_component = model_component[0]
                                
                                if patching_mode == "deletion":
                                    model_component[token_range] = 0
                                else:
                                    model_component[token_range] = replace_model_component[token_range]
                            
                            # Get logits from the patched model
                            patched_logits = self.sf(self.lm_head.output)
                            patched_logits = {
                                action: patched_logits[-1, -1, self.action_to_token_ids[action]].cpu().to(dtype=float).item().save()
                                for action in self.action_to_token_ids
                            }
                    
                    # Store the patched logits
                    _patching_logits.append({
                        action: patched_logits[action] * 1.0 for action in patched_logits
                    })
                
                all_patching_logits.append(_patching_logits)
        
        return all_patching_logits, base_logits, new_logits

    def get_metric_from_logits(
        self,
        patching_results,
        base_logits,
        new_logits,
        patching_mode="substitution",
    ):
        """Calculate metrics from patching results"""
        base_answer = max(base_logits, key=base_logits.get)
        new_answer = max(new_logits, key=new_logits.get)
        if patching_mode == "substitution":
            answer = new_answer
        else:
            answer = base_answer
        all_actions = list(self.action_to_nl.values())
        correct_actions = [answer]
        if patching_mode == "substitution":
            incorrect_actions = [base_answer]
        else:
            incorrect_actions = [
                action for action in all_actions
                if action != answer
            ]

        # n_layers x seq_len x n_correct_actions
        correct_logits = np.array([[
            [component[action] for action in correct_actions]
            for component in layer
        ] for layer in patching_results])
        # n_layers x seq_len x n_incorrect_actions
        incorrect_logits = np.array([[
            [component[action] for action in incorrect_actions]
            for component in layer
        ] for layer in patching_results])

        # sum up the ones with the correct parity minus ones with wrong parity
        logit_diff = correct_logits.max(-1) - incorrect_logits.max(-1)

        base_correct_probs = np.array([
            base_logits[action] for action in correct_actions
        ]).max(-1)
        base_incorrect_probs = np.array([
            base_logits[action] for action in incorrect_actions
        ]).max(-1)
        base_logit_diff = base_correct_probs - base_incorrect_probs

        # Calculate the metric based on patching mode
        if patching_mode == "deletion":
            # For deletion patching, assume new_logit_diff is 0
            delta_metric = (logit_diff - base_logit_diff) / (0 - base_logit_diff)
            # delta_metric = logit_diff / -base_logit_diff
        else:  # substitution patching
            new_correct_probs = np.array([
                new_logits[action] for action in correct_actions
            ]).max(-1)
            new_incorrect_probs = np.array([
                new_logits[action] for action in incorrect_actions
            ]).max(-1)
            new_logit_diff = new_correct_probs - new_incorrect_probs
            delta_metric = (logit_diff - base_logit_diff) / (new_logit_diff - base_logit_diff)
        
        # Cap to [0,1]
        delta_metric = np.clip(delta_metric, 0, 1)
        return delta_metric

    def run(self) -> None:
        """Run activation patching analysis"""
        prompts = self.generate_prompts()

        # Generate prompt pairs for patching
        pairs = self.generate_prompt_pairs(
            prompts,
            diff_parity="parity" in self.intervene_output_type,
        )
        
        # Set up layer components for patching
        layer_components = [(-1, "embed")]
        layer_names_plot = ["embed"]
        
        for layer_idx, layer in enumerate(self.layer_names):
            for component in layer:
                layer_components.append((layer_idx, component))
                component_name = "res" if component == "output" else component
                layer_names_plot.append(f"({layer_idx}, {component_name})")
        
        # Run patching for different intervention types
        for intervene_type in ["prefix", "suffix", "window"]:
            for parity_type in pairs:                    
                # Set up output directory
                output_fn = os.path.join(
                    self.checkpoint_dir,
                    f"{self.intervene_output_type}_{self.patching_mode}",
                )
                    
                os.makedirs(f"figures/intervene/{output_fn}", exist_ok=True)
                print(f"Saving to figures/intervene/{output_fn}")
                
                # Process pairs and collect results
                all_logit_diffs = []
                
                for pair in tqdm(pairs[parity_type], desc=f"Processing {parity_type} pairs for {intervene_type}"):
                    patching_results, base_logits, new_logits = self.get_patching_results_pair(
                        pair,
                        layer_components,
                        token_units=self.token_increments,
                        replace_prev_tokens=intervene_type == "prefix",
                        replace_next_tokens=intervene_type == "suffix",
                        patching_mode=self.patching_mode,
                    )
                    
                    all_logit_diffs.append(self.get_metric_from_logits(
                        patching_results, base_logits, new_logits, self.patching_mode,
                    ))
                
                # Plot individual results
                for i, (_, logit_diff) in enumerate(zip(pairs[parity_type], all_logit_diffs)):
                    self.visualization_manager.plot_logits(
                        logit_diff,
                        np.arange(0, self.n_tokens, self.token_increments),
                        layer_names_plot,
                        plot_name=os.path.join(
                            output_fn,
                            f"{parity_type}/{i}/{intervene_type}",
                        ),
                    )
                
                plot_name = os.path.join(
                    output_fn,
                    f"{intervene_type}_{parity_type}",
                )
                # Plot average results
                self.visualization_manager.plot_logits(
                    np.stack(all_logit_diffs).mean(0),
                    np.arange(0, self.n_tokens, self.token_increments),
                    layer_names_plot,
                    plot_name=plot_name,
                )
                print(f"Plot saved to figures/intervene/{plot_name}.png")