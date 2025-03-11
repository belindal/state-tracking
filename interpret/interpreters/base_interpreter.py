import torch
import numpy as np
from transformers import (
    GPT2Tokenizer, 
    AutoModel,
    GPT2LMHeadModel,
    LlamaForCausalLM,
    AutoTokenizer
)
from nnsight import LanguageModel
from torch.nn import LogSoftmax
from permutation_task import compute_parity, PermutationTask
import torch
from interpret.visualization_manager import VisualizationManager
import random
import os
from tqdm import tqdm
import json
from typing import Set


class BaseInterpreter:
    """Base class for model interpretation with core functionality."""
    
    def __init__(
        self,
        num_items: int,
        checkpoint_dir: str,
        model_type: str = "pythia",
        device: str = "cuda:0",
        n_tokens: int = 100,
        data_dir: str = None,
        n_prompts: int = 100,
        **kwargs
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_type = model_type
        self.device = device
        self.num_items = num_items
        self.n_tokens = n_tokens
        self.data_dir = data_dir
        self.n_prompts = n_prompts

        # Set up permutation task
        perm_task = PermutationTask(num_items=self.num_items)
        self.action_to_nl = perm_task.action_to_nl
        self.nl_to_action = perm_task.nl_to_action
        self.state_to_nl = perm_task.state_to_nl
        self.nl_to_state = perm_task.nl_to_state

        # Set initial state
        if self.num_items == 3:
            self.INIT_STATE = np.array([1,2,3])
        elif self.num_items == 5:  # Assuming this is what "swap5" means
            self.INIT_STATE = np.array([1,2,3,4,5])

        # Set up model and tokenizer
        self.setup_model()
        self.sf = LogSoftmax(dim=-1)
        self.visualization_manager = VisualizationManager()

    def setup_model(self):
        """Initialize model and tokenizer"""
        if self.model_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.checkpoint_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_dir)

        # Add special tokens
        self.tokenizer.add_tokens(list(self.action_to_nl.values()) + [f" {action}" for action in self.action_to_nl.values()])

        # Initialize models
        self.model = LanguageModel(self.checkpoint_dir, device_map=self.device, dispatch=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.model_type == "gpt2":
            self.model_hf = GPT2LMHeadModel.from_pretrained(self.checkpoint_dir).cuda(0)
            self.n_layers = self.model.config.n_layer
            self.layer_names = [["output"] for _ in range(self.n_layers)]
            self.inner_model = self.model.transformer
            self.embeddings = self.model.transformer.wte
            self.model_layers = self.model.transformer.h
            self.lm_head = self.model.lm_head
        elif self.model_type == "llama":
            self.model_hf = LlamaForCausalLM.from_pretrained(self.checkpoint_dir).cuda(0)
            self.n_layers = self.model.config.num_hidden_layers
            self.layer_names = [["output"] for _ in range(self.n_layers)]
            self.inner_model = self.model.model
            self.embeddings = self.model.model.embed_tokens
            self.model_layers = self.model.model.layers
            self.lm_head = self.model.lm_head
        elif self.model_type == "pythia":
            self.model_hf = AutoModel.from_pretrained(self.checkpoint_dir).cuda(0)
            self.n_layers = self.model.config.num_hidden_layers
            self.layer_names = [["output"] for _ in range(self.n_layers)]
            self.inner_model = self.model.gpt_neox
            self.embeddings = self.model.gpt_neox.embed_in
            self.model_layers = self.model.gpt_neox.layers
            self.lm_head = self.model.embed_out
            
        self.model_hf.resize_token_embeddings(len(self.tokenizer))
        self.action_to_token_ids = {
            action: self.tokenizer.convert_tokens_to_ids(" "+action)
            for action in self.nl_to_action
        }

    def generate_prompts(self) -> Set[str]:
        """
        Generate prompts either from data files or randomly.
        
        Args:
            interpreter: The model interpreter instance
            args: Command line arguments
            
        Returns:
            Set[str]: Set of generated prompts
        """
        if self.data_dir:
            # Load prompts from files
            prompts = set()
            for data_file in tqdm(os.listdir(self.data_dir), desc="Loading prompts from files"):
                with open(os.path.join(self.data_dir, data_file), "r") as f:
                    content = json.load(f)
                    prompts.add(" ".join(content["story"].split()[:self.n_tokens]))
                if len(prompts) >= self.n_prompts:
                    break
        else:
            # Generate random prompts
            prompts = set()
            for _ in tqdm(range(self.n_prompts), desc="Generating random prompts"):
                prompt = random.choices(list(self.action_to_nl.values()), k=self.n_tokens)
                prompt_str = " ".join(prompt)
                prompts.add(prompt_str)
        
        return prompts

    def cumulative_product(self, prompt):
        """Calculate cumulative states from a sequence of actions"""
        states = []
        curr_state = list(self.INIT_STATE)
        prompt_toks = self.tokenizer.tokenize(prompt)
        for token in prompt_toks:
            curr_action = self.nl_to_action[token.strip()]
            new_state = np.copy(curr_state)
            for old_pos, new_pos in enumerate(curr_action):
                new_state[old_pos] = curr_state[new_pos-1]
            curr_state = new_state
            states.append(tuple(curr_state))
        return states
