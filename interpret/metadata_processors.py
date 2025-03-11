from typing import List, Dict, Any
from permutation_task import compute_parity


class MetadataProcessor:
    """
    Process and extract metadata from model inputs and outputs.
    
    This class handles the extraction of various types of metadata from token sequences,
    such as state information, token IDs, and suffixes.
    """
    
    def __init__(self, model_interpreter, prompt_toks, end_idx=0):
        """
        Initialize the metadata processor.
        
        Args:
            model_interpreter: The model interpreter instance
            prompt_toks: List of tokens in the prompt
            end_idx: End index for processing
        """
        self.mi = model_interpreter
        self.prompt_toks = prompt_toks
        self.end_idx = end_idx
        
        # Map metadata keys to their processor methods
        self.processors = {
            "tokens": self.process_tokens,
            "length": self.process_tokens,  # Uses same base data as tokens
            "state": self.process_state_info,
            "state_parity": self.process_state_info,
            "action_parity": self.process_state_info,
        }
        
        # Add token ID processors
        for i in range(4):
            self.processors[f"Token {i} ID"] = self.process_token_ids
        for i in range(-4, 0):
            self.processors[f"Token {i} ID"] = self.process_token_ids

        # Add suffix processors
        for i in range(1, 100):  # Large enough range for any prompt length
            self.processors[f"State -{i} suffix"] = self.process_suffixes
    
    def add_all_entries_to_df(self, df, metadata_entries, token_positions=None):
        """
        Add metadata entries for all specified token positions to a dataframe.
        
        Args:
            df: Dataframe to add entries to
            metadata_entries: List of metadata entry names to add
            token_positions: List of token positions to process (default: all positions)
            
        Returns:
            DataFrame: The updated dataframe with metadata entries
        """
        if token_positions is None:
            token_positions = list(range(len(self.prompt_toks)))
            
        for t in token_positions:
            # Get tokens up to current position
            curr_toks = self.prompt_toks[:t+1]
            
            # Get metadata for current tokens
            metadata_values = self.get_metadata(
                metadata_entries,
                curr_toks=curr_toks,
            )
            
            # Add row to dataframe
            df.loc[len(df)] = [
                metadata_values.get(col)
                for col in metadata_entries
            ]
            
        return df

    def process_tokens(self, curr_toks: List[str], **kwargs) -> Dict[str, Any]:
        """
        Process basic token information.
        
        Args:
            curr_toks: List of tokens to process
            
        Returns:
            Dict: Dictionary with token information
        """
        return {
            "tokens": "".join(curr_toks),
            "length": len(curr_toks),
        }

    def process_state_info(self, curr_toks: List[str], **kwargs) -> Dict[str, Any]:
        """
        Process current state information.
        
        Args:
            curr_toks: List of tokens to process
            
        Returns:
            Dict: Dictionary with state information
        """
        # Calculate current state
        curr_state = self.mi.cumulative_product("".join(curr_toks[:self.end_idx]))[-1]
        
        # Get action and state information
        curr_action = self.mi.nl_to_action[curr_toks[-1].strip()]
        curr_state_nl = self.mi.action_to_nl[curr_state]
        curr_state_parity = compute_parity(curr_state)
        curr_action_parity = compute_parity(curr_action)
        
        # Return state information
        return_dict = {
            "state": curr_state_nl,
            "state_parity": curr_state_parity,
            "action_parity": curr_action_parity,
        }
        
        return return_dict

    def process_token_ids(self, curr_toks: List[str], **kwargs) -> Dict[str, Any]:
        """
        Process token IDs for first/last 4 positions.
        
        Args:
            curr_toks: List of tokens to process
            
        Returns:
            Dict: Dictionary with token ID information
        """
        result = {}
        
        # Process first 4 tokens
        for i in range(4):
            result[f"Token {i} ID"] = curr_toks[i] if i < len(curr_toks) else None
            
        # Process last 4 tokens
        for i in range(-4, 0):
            result[f"Token {i} ID"] = curr_toks[i] if abs(i) <= len(curr_toks) else None
            
        return result

    def process_suffixes(self, curr_toks: List[str], **kwargs) -> Dict[str, Any]:
        """
        Process state suffixes.
        
        Args:
            curr_toks: List of tokens to process
            
        Returns:
            Dict: Dictionary with suffix information
        """
        prompt_len = len(self.prompt_toks)
        
        # Calculate suffixes of different lengths
        suffix_cumprod = [
            self.mi.action_to_nl[self.mi.cumulative_product("".join(curr_toks[-i:]))[-1]]
            for i in range(1, len(curr_toks)+1)
        ] + [None for _ in range(len(curr_toks)+1, prompt_len+1)]
        
        # Return suffix information
        return {
            f"State -{i} suffix": suffix_cumprod[i-1] if i-1 < len(suffix_cumprod) else None
            for i in range(1, prompt_len+1)
        }

    def get_metadata(self, metadata_entries: List[str], **kwargs) -> Dict[str, Any]:
        """
        Get all requested metadata entries.
        
        Args:
            metadata_entries: List of metadata entry names to get
            
        Returns:
            Dict: Dictionary with all requested metadata
        """
        metadata_values = {}
        processed_groups = set()
        
        # Process each metadata entry
        for entry in metadata_entries:
            if entry in self.processors and self.processors[entry] not in processed_groups:
                metadata_values.update(self.processors[entry](**kwargs))
                processed_groups.add(self.processors[entry])
                
        return metadata_values
