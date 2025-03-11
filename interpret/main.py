import random
from tqdm import tqdm
import argparse
import os
import json
from typing import Set

from interpret.interpreters import BaseInterpreter, ActivationPatchingInterpreter, ProbeInterpreter, LengthwiseProbeInterpreter
from interpret.visualization_manager import VisualizationManager


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the model interpreter.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Neural network interpretation tools")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="gpt2", 
                        choices=["llama", "gpt2", "pythia"],
                        help="Type of model to interpret")
    parser.add_argument("--checkpoint_dir", type=str, 
                        default="checkpoint_fromscratch_llama3_S3_state/checkpoint-7032",
                        help="Directory containing model checkpoint")
    parser.add_argument("--num_items", type=int, default=3,
                        help="Number of items in the permutation task")
    
    # Data generation settings
    parser.add_argument("--data_dir", type=str, default="",
                        help="Directory containing data files (if empty, will generate random prompts)")
    parser.add_argument("--n_prompts", type=int, default=100,
                        help="Number of prompts to generate/use")
    parser.add_argument("--n_tokens", type=int, default=100,
                        help="Number of tokens per prompt")

    # Interpretation type and settings
    parser.add_argument("--interpret_type", type=str, default="intervene", 
                        choices=["intervene", "probe", "lengthwise_probe"],
                        help="Type of interpretation to perform")
    
    # Intervention settings
    parser.add_argument("--intervene_output_type", type=str, default="state", 
                        choices=["state", "state_by_parity"],
                        help="Type of output for intervention analysis")
    parser.add_argument("--patching_mode", type=str, default="substitution",
                        choices=["deletion", "substitution"],
                        help="Mode for activation patching: deletion (zeros) or substitution (from pair)")
    
    # Probe settings
    parser.add_argument("--probe_output_types", type=str, default="state,state_parity",
                        help="Comma-separated list of probes to run")
    
    # Lengthwise probe settings
    parser.add_argument("--token_increments", type=int, default=5,
                        help="Token increment size for lengthwise probing")
    parser.add_argument("--start_idx", type=int, default=5, 
                        help="Start index for lengthwise probe")
    parser.add_argument("--end_idx", type=int, default=None, 
                        help="End index for lengthwise probe")
    
    return parser.parse_args()


def main():
    """
    Main function to run the model interpreter.
    """
    # Parse command line arguments
    args = parse_arguments()

    if args.interpret_type == "intervene":
        interpreter = ActivationPatchingInterpreter(
            args.num_items, 
            args.checkpoint_dir, 
            model_type=args.model_type, 
            device="cuda:0",
            data_dir=args.data_dir,
            n_prompts=args.n_prompts,
            intervene_output_type=args.intervene_output_type,
            patching_mode=args.patching_mode,
            token_increments=args.token_increments,
            n_tokens=args.n_tokens,
        )
    elif args.interpret_type == "probe":
        interpreter = ProbeInterpreter(
            args.num_items, 
            args.checkpoint_dir, 
            model_type=args.model_type, 
            device="cuda:0",
            data_dir=args.data_dir,
            n_prompts=args.n_prompts,
            probe_output_types=args.probe_output_types.split(","),
            n_tokens=args.n_tokens,
        )
    elif args.interpret_type == "lengthwise_probe":
        interpreter = LengthwiseProbeInterpreter(
            args.num_items, 
            args.checkpoint_dir, 
            model_type=args.model_type, 
            device="cuda:0",
            data_dir=args.data_dir,
            n_prompts=args.n_prompts,
            probe_output_types=args.probe_output_types.split(","),
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            token_increments=args.token_increments,
            n_tokens=args.n_tokens,
        )
    else:
        raise ValueError(f"Invalid interpretation type: {args.interpret_type}")
    
    interpreter.run()


if __name__ == "__main__":
    main()