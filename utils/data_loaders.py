import os
import json
import gc
import torch
import numpy as np


class ChunkedDataset:
    """
    Dataset that loads data in chunks to save memory.
    Useful for large datasets that don't fit in memory.
    """
    def __init__(self, data_dir, max_len, chunk_size=10000, debug=False):
        self._format_type = None
        self._format_columns = None
        self._fingerprint = None  # Add fingerprint attribute
        self.data_dir = data_dir
        self.max_len = max_len
        self.chunk_size = chunk_size
        self.debug = debug

        # Get list of all files
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        if self.debug:
            self.files = self.files[:100]
        self.current_chunk = {}
        self.current_chunk_start = 0
        
        # Pre-compute file index mapping
        self.file_index_map = []  # List of (start_idx, end_idx, filename) tuples
        current_idx = 0
        
        # Calculate total size and build index mapping
        for filename in self.files:
            self.file_index_map.append((current_idx, current_idx + 1, filename))
            current_idx += 1
        
        self.total_size = current_idx

    def set_format(self, type=None, columns=None):
        """Sets the format of the dataset."""
        self._format_type = type
        self._format_columns = columns
        
        # If format is torch, convert current chunk to tensors
        if type == 'torch':
            for col in columns:
                if col in self.current_chunk:
                    self.current_chunk[col] = torch.tensor(self.current_chunk[col])
        
        return self

    def load_chunk(self, start_idx):
        """Load a chunk of data starting from start_idx."""
        if hasattr(self, "current_chunk"):
            del self.current_chunk
            if start_idx % 5000 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        self.current_chunk = {}
        items_loaded = 0
        
        # Binary search to find the starting file
        left, right = 0, len(self.file_index_map)
        while left < right:
            mid = (left + right) // 2
            if self.file_index_map[mid][0] <= start_idx:
                if mid + 1 == len(self.file_index_map) or self.file_index_map[mid + 1][0] > start_idx:
                    left = mid
                    break
                left = mid + 1
            else:
                right = mid
                
        # Load files until chunk_size is reached
        current_file_idx = left
        while items_loaded < self.chunk_size and current_file_idx < len(self.file_index_map):
            filename = self.file_index_map[current_file_idx][2]
            with open(os.path.join(self.data_dir, filename), 'r') as f:
                data = json.load(f)
                for k in data:
                    if k not in self.current_chunk:
                        self.current_chunk[k] = []

                    if k == "story":
                        self.current_chunk[k].append(" ".join(data[k].split()[:self.max_len+1]))
                    else:
                        self.current_chunk[k].append(data[k][:self.max_len+1])
            items_loaded += 1
            if items_loaded >= self.chunk_size:
                break
            current_file_idx += 1
            
        # Store the actual start index of this chunk
        self.current_chunk_start = self.file_index_map[left][0]

    def __len__(self):
        """Return the total size of the dataset."""
        return self.total_size

    def __getitem__(self, idx):
        """Get an item from the dataset."""
        # Handle batch indices
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]

        # Load new chunk if necessary
        if idx < self.current_chunk_start or idx >= self.current_chunk_start + len(self.current_chunk.get("story", [])):
            self.load_chunk(idx)

        chunk_idx = idx - self.current_chunk_start
        return {k: self.current_chunk[k][chunk_idx] for k in self.current_chunk}

    def map(self, function, batched=True, remove_columns=None, **kwargs):
        """Apply a function to the dataset."""
        # Create a new MappedChunkedDataset that wraps this dataset
        return MappedChunkedDataset(self, function, remove_columns)
    
    @property
    def column_names(self):
        """Return list of available columns."""
        return list(self.current_chunk.keys())


def tokenize_function(examples, tokenizer, max_len=1024):
    """
    Tokenize the text and create labels for next token prediction.
    
    Args:
        examples: Dictionary of examples
        tokenizer: Tokenizer to use
        
    Returns:
        tokenized: Dictionary of tokenized examples
    """
    try:
        tokenized = tokenizer([ex['story'] for ex in examples], padding='longest', truncation=True, max_length=max_len)
        # Create labels by shifting input_ids one position to the right (automatically done in huggingface LM)
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        # Convert lists to numpy arrays to avoid slow tensor creation warning
        for key in tokenized:
            if isinstance(tokenized[key], list):
                tokenized[key] = np.array(tokenized[key])
                
        return tokenized
    except Exception as e:
        breakpoint()
        print(f"Error tokenizing example: {examples}")
        raise e


class MappedChunkedDataset:
    """Dataset that wraps a ChunkedDataset and applies a function to each chunk on-the-fly."""
    
    def __init__(self, base_dataset, map_function, remove_columns=None):
        self.base_dataset = base_dataset
        self.map_function = map_function
        self.remove_columns = remove_columns
        self.processed_chunks = {}  # Cache for processed chunks
        self._format_type = base_dataset._format_type
        self._format_columns = base_dataset._format_columns
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Determine which chunk this index belongs to
        chunk_idx = idx // self.base_dataset.chunk_size * self.base_dataset.chunk_size
        
        # Process the chunk if not already processed
        if chunk_idx not in self.processed_chunks:
            # Get the raw chunk from the base dataset
            raw_chunk = [self.base_dataset[i] for i in range(
                chunk_idx, 
                min(chunk_idx + self.base_dataset.chunk_size, len(self.base_dataset))
            )]
            
            # Apply the mapping function
            processed_chunk = self.map_function(raw_chunk)
            
            # Remove columns if specified
            if self.remove_columns:
                for col in self.remove_columns:
                    if col in processed_chunk:
                        del processed_chunk[col]
            
            # Store the processed chunk
            self.processed_chunks[chunk_idx] = processed_chunk
        
        # Get the item from the processed chunk
        relative_idx = idx - chunk_idx
        result = {}
        for key in self.processed_chunks[chunk_idx]:
            if relative_idx < len(self.processed_chunks[chunk_idx][key]):
                result[key] = self.processed_chunks[chunk_idx][key][relative_idx]
        
        # Apply format if needed
        if self._format_type and self._format_columns:
            formatted_result = {}
            for col in self._format_columns:
                if col in result:
                    if self._format_type == 'torch':
                        formatted_result[col] = torch.tensor(result[col])
                    else:
                        formatted_result[col] = result[col]
            return formatted_result
        
        return result
    
    def set_format(self, type=None, columns=None):
        """Set the format of the dataset."""
        self._format_type = type
        self._format_columns = columns
    
    @property
    def column_names(self):
        """Return list of available columns."""
        # Get a sample processed chunk to determine columns
        if not self.processed_chunks:
            sample_idx = 0
            _ = self[sample_idx]  # This will populate processed_chunks
        
        if self.processed_chunks:
            first_chunk_idx = next(iter(self.processed_chunks))
            return list(self.processed_chunks[first_chunk_idx].keys())
        
        # Fallback to base dataset columns
        return self.base_dataset.column_names
