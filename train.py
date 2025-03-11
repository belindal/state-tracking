import torch
import os
import json
import argparse
import numpy as np
import wandb
import random

from transformers import (
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
)

from utils.data_loaders import (
    ChunkedDataset,
)
from utils.data_collators import (
    DataCollatorForLanguageModelingWithNextTokenSupervision,
    DataCollatorForLanguageModelingWithDirectSupervision
)

from permutation_task import PermutationTask, compute_parity
from utils.model_utils import setup_tokenizer, setup_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", choices=[
        "gpt2", "gpt2-large", "gpt2-medium", "gpt2-xl", "distilgpt2",
        "EleutherAI/pythia-70M", "EleutherAI/pythia-160M", "EleutherAI/pythia-410M",
        "EleutherAI/pythia-1B", "EleutherAI/pythia-1.4B", "EleutherAI/pythia-2.8B",
        "EleutherAI/pythia-6.9B", "EleutherAI/pythia-12B",
    ])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--num_items", type=int, default=3, choices=[3, 5], help="Number of items for permutation task")
    parser.add_argument("--supervision_type", type=str, default="next_token", choices=["direct_state", "direct_topic", "next_token"])
    parser.add_argument("--layerwise_supervision_type", type=str, default=None, help="File containing layerwise supervision keys")
    parser.add_argument("--no_pretrain", action="store_true", default=False, help="If true, does not use pre-trained model")
    parser.add_argument("--from_checkpoint", type=str, default=None, help="If provided, loads model from checkpoint")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_bfloat16", action="store_true", default=False, help="If true, uses bfloat16")
    parser.add_argument("--save_all_checkpoints", type=int, default=None, help="If set, saves checkpoints and interval specified in argument")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_determinism", action="store_true", default=False)
    parser.add_argument("--full_determinism", action="store_true", default=False)
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--is_parity_cur", action="store_true", default=False)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def setup_data_collator(args, tokenizer, state_tokens, parity=None, layerwise_supervision_config=None):
    """
    Set up the data collator based on the supervision type.
    
    Args:
        args: Command-line arguments
        tokenizer: Tokenizer
        state_tokens: Dictionary mapping states to tokens
        parity: Dictionary mapping states to parity values
        layerwise_supervision_config: Configuration for layerwise supervision
        
    Returns:
        data_collator: Data collator for training
    """
    if args.supervision_type == "next_token":
        # For next token prediction, use standard language modeling collator
        data_collator = DataCollatorForLanguageModelingWithNextTokenSupervision(
            tokenizer=tokenizer,
            max_len=args.max_len,
        )
    elif args.supervision_type == "direct_state":
        # For direct state prediction, use sequence-to-sequence collator
        data_collator = DataCollatorForLanguageModelingWithDirectSupervision(
            tokenizer=tokenizer,
            max_len=args.max_len,
            STATE_TOKENS=state_tokens,
            label_key="state_seq",
            PARITY=parity if args.is_parity_cur else None,
            layerwise_intermediate_keys=layerwise_supervision_config
        )
    else:
        raise ValueError(f"Unknown supervision type: {args.supervision_type}")
        
    return data_collator


def prepare_dataset(args, tokenizer, state_tokens, data_collator, debug=False):
    """Prepare the dataset for training."""
    print("Loading data")
    full_dataset = ChunkedDataset(args.data_dir, args.max_len, chunk_size=1, debug=debug)
    # Split into train/test
    total_size = len(full_dataset)
    train_size = int(0.95 * total_size)
    
    if args.full_determinism or args.data_determinism:
        # Use a fixed split by taking the first train_size examples for training
        indices = list(range(total_size))
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        eval_dataset = torch.utils.data.Subset(full_dataset, eval_indices)
    else:
        torch.manual_seed(args.seed)
        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, total_size - train_size]
        )
    
    print("Train dataset size:", len(train_dataset))
    print("Eval dataset size:", len(eval_dataset))
    
    return train_dataset, eval_dataset


def setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, data_collator):
    """Set up the trainer with the appropriate arguments."""
    print("Setting up trainer")
    wandb.init(project="state-tracking", name=args.output_dir)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=2000 if not args.save_all_checkpoints else args.save_all_checkpoints,
        save_strategy="steps",
        save_total_limit=None if args.save_all_checkpoints else 1,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps" if args.early_stopping else "epoch",
        eval_steps=2000 if args.early_stopping else None,
        batch_eval_metrics=True,
        eval_on_start=True if args.early_stopping else False,
        remove_unused_columns=False,
        report_to="none" if args.disable_wandb else "wandb",
        dataloader_num_workers=1,
        bf16=args.use_bfloat16,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=args.full_determinism,
        metric_for_best_model="eval_loss" if args.early_stopping else None,
        greater_is_better=False if args.early_stopping else True,
        load_best_model_at_end=True if args.early_stopping else False,
    )
    
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    if args.early_stopping:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
    
    return trainer


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up determinism
    if args.full_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up task
    task = PermutationTask(num_items=args.num_items)
    
    # Extract tokens and compute parity
    state_tokens = {state.permutation: state.to_string() for state in task.states}
    action_tokens = {action.permutation: action.to_string() for action in task.actions}
    parity = None
    
    # Update parity if needed
    if args.is_parity_cur:
        # Use the compute_parity function to calculate parity for all states
        parity = {state: compute_parity(state) for state in state_tokens}
    
    # Load layerwise supervision config if provided
    layerwise_supervision_config = None
    if args.layerwise_supervision_type is not None and os.path.exists(args.layerwise_supervision_type):
        with open(args.layerwise_supervision_type, "r") as f:
            layerwise_supervision_config = json.load(f)
    
    # Set up tokenizer
    tokenizer = setup_tokenizer(args.model, state_tokens, action_tokens)
    
    model = setup_model(
        tokenizer=tokenizer,
        model_name=args.model,
        checkpoint_path=args.from_checkpoint if hasattr(args, 'from_checkpoint') and args.from_checkpoint else None,
        use_bfloat16=args.use_bfloat16,
        no_pretrain=args.no_pretrain,
        output_dir=args.output_dir,
        use_custom_models=True,
        layerwise_supervision_config=layerwise_supervision_config
    )
    
    # Set up data collator
    data_collator = setup_data_collator(
        args, tokenizer, state_tokens, parity, layerwise_supervision_config)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset(args, tokenizer, state_tokens, data_collator, debug=args.debug)
    
    # Set up trainer
    trainer = setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, data_collator)
    
    # Train model
    trainer.train(resume_from_checkpoint=args.from_checkpoint)
    
    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
