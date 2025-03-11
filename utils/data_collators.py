import torch
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DataCollatorForLanguageModelingWithNextTokenSupervision:
    """
    Data collator for language modeling with next token supervision.
    This collator prepares the data for next token prediction tasks.
    """
    tokenizer: Any
    max_len: int = 512
    mlm: bool = False
    mlm_probability: float = 0.15
    
    def __call__(self, examples):
        batch = {}
        
        # Tokenize if not already tokenized
        if "input_ids" not in examples[0]:
            batch = self.tokenizer.batch_encode_plus(
                [example["story"] for example in examples],
                padding='longest',
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            batch["labels"] = batch["input_ids"].clone()
        else:
            # If already tokenized, just stack the tensors
            batch["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            batch["attention_mask"] = torch.stack([example["attention_mask"] for example in examples])
            batch["labels"] = torch.stack([example["labels"] for example in examples])
            
        return batch


@dataclass
class DataCollatorForLanguageModelingWithDirectSupervision:
    """
    Data collator for language modeling with direct supervision.
    This collator prepares the data for tasks where we have direct supervision signals.
    """
    tokenizer: Any
    max_len: int = 512
    STATE_TOKENS: Dict = None
    label_key: str = None
    PARITY: Dict = None
    layerwise_intermediate_keys: Dict = None
    
    def __call__(self, examples):
        batch = self.tokenizer(
            [example["story"] for example in examples],
            padding='longest',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Create labels for direct supervision
        if self.label_key is not None:
            # Convert state sequences to token IDs
            # state_token_ids = []
            label_seq = []
            converter = self.PARITY if self.PARITY is not None else self.STATE_TOKENS
            label_seq = [
                " ".join([converter[tuple(item)] for item in example[self.label_key]])
                for example in examples
            ]
            batch["labels"] = self.tokenizer(
                label_seq,
                padding='longest',
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )["input_ids"]
            # make same shape as input_ids
            batch["labels"] = batch["labels"][:, :batch["input_ids"].size(1)]
        
        # Add layerwise supervision if needed
        if self.layerwise_intermediate_keys is not None:
            for layer_idx, layer_config in self.layerwise_intermediate_keys.items():
                key = layer_config["key"]
                batch[f"layer_{layer_idx}_labels"] = torch.zeros(
                    (len(examples), batch["input_ids"].size(1)),
                    dtype=torch.long,
                    device=batch["input_ids"].device
                )
                intermediate_seq = self.tokenizer(
                    [example[key] for example in examples],
                    padding='longest',
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt"
                )["input_ids"]
                batch[f"layer_{layer_idx}_labels"] = intermediate_seq[:, :batch["input_ids"].size(1)]
        
        return batch
