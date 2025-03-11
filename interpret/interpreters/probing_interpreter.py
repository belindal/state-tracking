import numpy as np
import pandas as pd
import os
import pickle
import random
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from typing import Set, Dict, List, Tuple, Any, Optional
from interpret.interpreters.base_interpreter import BaseInterpreter
from interpret.metadata_processors import MetadataProcessor


class ProbeInterpreter(BaseInterpreter):
    """Class for probing analysis."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_output_types = kwargs.get("probe_output_types", ["state", "state_parity"])
    
    def get_metadata_entries(self, prompts, end_idx, metadata_entries=None, token_positions=None, tqdm_desc=None):
        """Get layer representations for a set of prompts"""
        if metadata_entries is None:
            metadata_entries = [
                "tokens", "state",
                "state_parity", "action_parity",
            ]
        # Create DataFrame for metadata
        df = pd.DataFrame(columns=metadata_entries)

        # Process each prompt
        for prompt in tqdm(prompts, desc=tqdm_desc):
            prompt_toks = self.tokenizer.tokenize(prompt)
            metadata_processor = MetadataProcessor(self, prompt_toks=prompt_toks, end_idx=end_idx)
            
            df = metadata_processor.add_all_entries_to_df(df, metadata_entries, token_positions=token_positions)

            if token_positions is None:
                token_positions = list(range(len(prompt_toks)))
        return df

    def get_layer_representations(self, prompts, token_positions=None, tqdm_desc=None):
        """Get layer representations for a set of prompts"""
        all_layer_reps = {}
        # Process each prompt
        for prompt in tqdm(prompts, desc=tqdm_desc):
            layer_reps = {}
            with torch.no_grad():
                # Extract representations from model
                with self.model.trace() as tracer:
                    with tracer.invoke(prompt) as invoker:
                        # Get embeddings
                        reps_layer = self.embeddings(invoker.inputs[0][0]["input_ids"]).squeeze(0)[token_positions].save()
                        layer_reps["embed"] = reps_layer

                        # Get layer representations
                        for layer_idx in range(self.n_layers):
                            for layer_component_path in self.layer_names[layer_idx]:
                                model_component = self.model_layers[layer_idx]
                                    
                                subcomponents = layer_component_path.split(".")
                                for subcomponent_name in subcomponents:
                                    model_component = getattr(model_component, subcomponent_name)
                                    
                                if layer_component_path in ["output", "attn.output", "self_attn.output","attention.output"]:
                                    layer_reps[f"{layer_idx}.{layer_component_path}"] = model_component[0][:, token_positions, :].save()
                                else:
                                    layer_reps[f"{layer_idx}.{layer_component_path}"] = model_component[:, token_positions, :].save()

                # Convert to numpy and store
                for item in layer_reps:
                    layer_reps[item] = layer_reps[item].cpu().numpy() * 1.0
                    if item not in all_layer_reps:
                        all_layer_reps[item] = []
                    all_layer_reps[item].append(layer_reps[item].squeeze())

        # Stack representations
        for item in tqdm(all_layer_reps, desc="Stacking"):
            all_layer_reps[item] = np.stack(all_layer_reps[item], axis=0)

        return all_layer_reps

    def get_linear_probe_scores(self, reps, labels, num_labels: int, split=0.8):
        """Get linear probe scores for a set of representations and labels"""
        if labels.dtype == np.dtype('O'):
            unique_labels = np.unique(labels)
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = np.array([label_to_id[label] for label in labels])
            labels = numeric_labels
        state_probe = LogisticRegression(max_iter=5000, n_jobs=1)

        split = min(1000, int(split * len(reps)))
        train_reps, test_reps = reps[:split,:], reps[split:,:]
        train_states, test_states = labels[:split], labels[split:]
        state_probe.fit(train_reps, train_states)
        state_probe_probs = state_probe.predict_proba(test_reps)
        if state_probe_probs.shape[1] < num_labels:
            # Get the labels that the model was trained on
            trained_labels = state_probe.classes_
            # Create a full probability matrix with zeros
            full_probs = np.zeros((state_probe_probs.shape[0], num_labels))
            # Fill in the probabilities for the trained labels
            for i, label in enumerate(trained_labels):
                full_probs[:, label] = state_probe_probs[:, i]
            state_probe_probs = full_probs

        state_probe_labels = test_states
        return (
            state_probe.score(test_reps, test_states),
            state_probe_probs,
            state_probe_labels,
            state_probe,
        )

    def train_probes(self, layer_reps, metadata_df, probe_output_types=None):
        """Train linear probes on layer representations"""
        if probe_output_types is None:
            probe_output_types = [
                "state_parity", "state"
            ]
                          
        probes = {}
        layerwise_type_scores = {}
        layerwise_type_probs = {}
        layerwise_type_labels = {}
        for probe_type in probe_output_types:
            probes[probe_type] = {}
            layerwise_type_scores[probe_type] = []
            layerwise_type_probs[probe_type] = []
            layerwise_type_labels[probe_type] = []

            y = metadata_df[probe_type]
            num_labels = len(y.unique())
            # filter None values
            na_mask = y.isna()
            y = y[~na_mask]

            for layer_name in tqdm(layer_reps, desc=f"Training {probe_type} probes"):
                X = layer_reps[layer_name]
                X = X.reshape(-1, X.shape[-1])[~na_mask]
                
                score, probs, labels, probe = self.get_linear_probe_scores(
                    X, y, num_labels,
                )
                probes[probe_type][layer_name] = probe
                layerwise_type_scores[probe_type].append(score)
                layerwise_type_probs[probe_type].append(probs)
                layerwise_type_labels[probe_type].append(labels)

        return (
            probes,
            layerwise_type_scores,
            layerwise_type_probs,
            layerwise_type_labels,
        )

    def run_probe(self, prompts, end_idx, probe_output_types=None):
        """Run probing analysis on model representations"""
        # Get layer representations and metadata
        layer_reps = self.get_layer_representations(
            prompts, 
            token_positions=list(range(end_idx)),
            tqdm_desc="Extracting representations for probing"
        )
        
        # Define probe output types
        if probe_output_types is None:
            probe_output_types = ["state", "state_parity"]
        
        metadata_df = self.get_metadata_entries(
            prompts,
            end_idx=end_idx,
            tqdm_desc=f"Getting metadata for length {end_idx}"
        )
        
        # Train probes on the representations
        (
            probes,
            layerwise_type_scores,
            layerwise_type_probs,
            layerwise_type_labels,
        ) = self.train_probes(
            layer_reps, 
            metadata_df, 
            probe_output_types=probe_output_types,
        )

        return layerwise_type_scores


    def run(self) -> None:
        """Run probing analysis."""
        # Run probe
        prompts = self.generate_prompts()
        probe_results = self.run_probe(
            prompts,
            end_idx=self.n_tokens,
            probe_output_types=self.probe_output_types,
        )

        # Plot results
        self.visualization_manager.plot_probes(probe_results, plot_name=f"{self.checkpoint_dir}")


class LengthwiseProbeInterpreter(ProbeInterpreter):
    """Run probing analysis across different token lengths"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_idx = kwargs.get("start_idx", 0)
        self.end_idx = kwargs.get("end_idx", self.n_tokens)
        self.token_increments = kwargs.get("token_increments", 1)

    def per_length_lm_representations(self, end_idx, n_prompts):
        """Generate prompts and get representations for a specific token length"""
        # Generate random prompts
        prompts = set()
        for _ in tqdm(range(n_prompts), desc=f"Generating prompts for length {end_idx}"):
            prompt = random.choices(list(self.action_to_nl.values()), k=end_idx)
            prompt_str = " ".join(prompt)
            prompts.add(prompt_str)
            
        return prompts

    def per_length_lm_probes(self, prompts, token_idx, probe_output_types):
        """Run probing analysis for a specific token length range"""
        # Get layer representations and metadata
        layer_reps = self.get_layer_representations(
            prompts, 
            token_positions=list(range(token_idx)),
            tqdm_desc=f"Getting representations for length {token_idx}"
        )

        metadata_df = self.get_metadata_entries(
            prompts,
            end_idx=token_idx,
            tqdm_desc=f"Getting metadata for length {token_idx}"
        )
        
        # Train probes on the representations
        (
            probes,
            layerwise_type_scores,
            layerwise_type_probs,
            layerwise_type_labels,
        ) = self.train_probes(
            layer_reps, 
            metadata_df, 
            probe_output_types=probe_output_types
        )
        
        return probes, layerwise_type_scores, layerwise_type_probs, layerwise_type_labels


    def sweep_length_probes(self, n_prompts, n_tokens, token_increments, checkpoint_name,
                           start_idx=0, end_idx=None, probe_output_types=None):
        """Run probing analysis across different token lengths"""
        # Set end index if not provided
        if end_idx is None:
            end_idx = n_tokens
            
        # Create output directories
        os.makedirs(f"probe_results/{checkpoint_name}", exist_ok=True)
        
        # Initialize results dictionary
        if probe_output_types is None:
            probe_output_types = ["state", "state_parity"]
        probe_results = {
            probe_type: {} for probe_type in probe_output_types
        }
        
        # Run probing for each token length
        for length in range(start_idx, end_idx, token_increments):
            print(f"Running probes for length {length}")
            
            # Get prompts for this length
            length_prompts = self.per_length_lm_representations(length, n_prompts)
            
            not_computed = []
            for probe_type in probe_output_types:
                save_file = f"probe_results/{checkpoint_name}/{length}_{probe_type}.npy"
                if os.path.exists(save_file):
                    probe_results[probe_type][length] = np.load(save_file)
                else:
                    not_computed.append(probe_type)

            if len(not_computed) > 0:
                # Run probes for this length
                (
                    _,
                    layerwise_type_scores,
                    _,
                    _,
                ) = self.per_length_lm_probes(
                    length_prompts, 
                    token_idx=length,
                    probe_output_types=not_computed,
                )
            
            # Store results
            for probe_type in not_computed:
                probe_results[probe_type][length] = layerwise_type_scores[probe_type]

                np.save(
                    f"probe_results/{checkpoint_name}/{length}_{probe_type}.npy", 
                    np.array(layerwise_type_scores[probe_type])
                )
        
        print(f"Saved lengthwise probe results to probe_results/{checkpoint_name}/{length}_state.npy and probe_results/{checkpoint_name}/{length}_parity.npy")

        return probe_results

    def run(self):
        """Run lengthwise probing"""
        # Set up layer names for plotting
        layer_names_plot = ["embed"]
        for layer_idx, layer in enumerate(self.layer_names):
            for component in layer:
                component_name = "res" if component == "output" else component
                layer_names_plot.append(f"({layer_idx}, {component_name})")

        # Run lengthwise probing
        probe_results = self.sweep_length_probes(
            self.n_prompts, self.n_tokens, self.token_increments, self.checkpoint_dir,
            start_idx=self.start_idx, end_idx=self.end_idx,
            probe_output_types=self.probe_output_types,
        )
        
        # Plot results
        self.visualization_manager.plot_length_probe_heatmap(
            probe_results["state"],
            layer_names=layer_names_plot,
            plot_name=f"{self.checkpoint_dir}_{self.n_tokens}",
        )
