import sys
from utils.data_loaders import ChunkedDataset, tokenize_function
from transformers import AutoTokenizer


def test_map_functionality():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    
    # Create dataset
    print(f"Loading dataset from {data_dir}")
    dataset = ChunkedDataset(data_dir, max_len=512, chunk_size=1, debug=True)
    
    # Print original dataset info
    print(f"Original dataset length: {len(dataset)}")
    print(f"Original dataset columns: {dataset.column_names}")
    
    # Apply map function
    print("Applying map function...")
    mapped_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_len=512),
        batched=True
    )
    
    # Print mapped dataset info
    print(f"Mapped dataset length: {len(mapped_dataset)}")
    print(f"Mapped dataset columns: {mapped_dataset.column_names}")
    
    # Check if tokenization was applied by accessing an item
    print("Checking first item...")
    first_item = mapped_dataset[0]
    print(f"Keys in first item: {list(first_item.keys())}")
    
    # Set format and check again
    print("Setting format...")
    mapped_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Check formatted item
    formatted_item = mapped_dataset[0]
    print(f"Keys in formatted item: {list(formatted_item.keys())}")
    
    # Check a few items to ensure consistency
    print("Checking multiple items...")
    for i in range(min(3, len(mapped_dataset))):
        item = mapped_dataset[i]
        print(f"Item {i} has keys: {list(item.keys())}")
        if 'input_ids' in item:
            print(f"Item {i} input_ids shape: {item['input_ids'].shape}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_map_functionality() 