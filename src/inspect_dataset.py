# src/inspect_dataset.py

from datasets import load_dataset

def inspect_dataset():
    print("Loading dataset...")
    dataset = load_dataset("empathetic_dialogues")

    for split in ['train', 'validation', 'test']:
        print(f"\n=== First example in '{split}' ===")
        example = dataset[split][0]
        print(f"Type of example: {type(example)}")
        print("Keys:", example.keys())
        print("\nFull example (first 500 chars):")
        print(str(example)[:500])

inspect_dataset()