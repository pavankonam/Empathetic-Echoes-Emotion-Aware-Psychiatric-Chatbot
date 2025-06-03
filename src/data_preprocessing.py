# src/data_processing.py

import sys
import os

# Add project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset, DatasetDict
from tqdm import tqdm
import json

def process_dataset():
    print("🧠 Loading FULL empathetic_dialogues dataset...")

    # Load full dataset splits
    raw_train = load_dataset("empathetic_dialogues", split="train[:100%]")
    raw_val = load_dataset("empathetic_dialogues", split="validation[:100%]")
    raw_test = load_dataset("empathetic_dialogues", split="test[:100%]")

    dataset = DatasetDict({
        "train": raw_train,
        "validation": raw_val,
        "test": raw_test
    })

    print("\n📊 Raw dataset sizes (utterances):")
    for split in ['train', 'validation', 'test']:
        print(f"{split}: {len(dataset[split])} utterances")

    final_examples = {'train': [], 'validation': [], 'test': []}

    for split in ['train', 'validation', 'test']:
        print(f"\n🔄 Processing split: {split}")
        split_data = dataset[split]

        # Build conversation map per split
        conversations = {}

        for example in tqdm(split_data, desc=f"Grouping utterances for {split}"):
            conv_id = example['conv_id']
            if conv_id not in conversations:
                conversations[conv_id] = {
                    'emotion': example['context'],
                    'utterances': []
                }
            
            conversations[conv_id]['utterances'].append({
                'speaker_idx': example['speaker_idx'],
                'text': example['utterance'],
                'utterance_idx': example['utterance_idx']
            })

        print(f"Found {len(conversations)} unique conversations in {split}")

        # Sort utterances by utterance_idx to ensure correct order
        for conv_id in conversations:
            conversations[conv_id]['utterances'].sort(key=lambda x: x['utterance_idx'])

        def build_pairs(conv_data):
            examples = []
            utterances = conv_data['utterances']
            emotion = conv_data['emotion']

            # Since conversations alternate, we need to identify which speaker starts
            # The pattern should be: Speaker A -> Speaker B -> Speaker A -> Speaker B
            if len(utterances) < 2:
                return examples

            # Determine speaker roles based on position
            # In empathetic dialogues, typically the person sharing emotion speaks first
            first_speaker = utterances[0]['speaker_idx']
            
            for i in range(len(utterances) - 1):
                current_utterance = utterances[i]
                next_utterance = utterances[i + 1]
                
                # Create pairs where first speaker talks and second speaker responds
                # The first speaker is typically the "user" sharing their experience
                # The second speaker is the "listener" providing empathetic responses
                if current_utterance['speaker_idx'] == first_speaker and next_utterance['speaker_idx'] != first_speaker:
                    # Build context up to current user message
                    context = ""
                    for j in range(i + 1):  # Include current user message
                        utt = utterances[j]
                        # Label speakers consistently within conversation
                        if utt['speaker_idx'] == first_speaker:
                            speaker_label = "User"
                        else:
                            speaker_label = "Bot"
                        context += f"{speaker_label}: {utt['text']}\n"

                    bot_response = next_utterance['text']
                    examples.append({
                        'emotion': emotion,
                        'context': context.strip(),
                        'response': bot_response.strip()
                    })
            
            return examples

        print(f"🛠️ Building training examples for {split}...")
        examples_list = []
        skipped_short = 0
        skipped_no_pairs = 0
        total_conversations = len(conversations)

        for conv_id, conv_data in tqdm(conversations.items(), desc=f"Converting {split} convs"):
            if len(conv_data['utterances']) < 2:
                skipped_short += 1
                continue

            examples = build_pairs(conv_data)
            if not examples:
                skipped_no_pairs += 1
                continue

            examples_list.extend(examples)

        print(f"✅ Processing complete for {split}:")
        print(f"  - Total conversations: {total_conversations}")
        print(f"  - Skipped {skipped_short} conversations (too short)")
        print(f"  - Skipped {skipped_no_pairs} conversations (no valid pairs)")
        print(f"  - Successfully processed: {total_conversations - skipped_short - skipped_no_pairs}")
        print(f"  - Generated {len(examples_list)} training pairs")

        final_examples[split].extend(examples_list)

    # Save results
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n💾 Saving processed data...")
    for split, data in final_examples.items():
        output_path = os.path.join(output_dir, f"{split}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} examples to {output_path}")

    # Print summary statistics
    print(f"\n📈 Final Statistics:")
    total_pairs = 0
    for split in ['train', 'validation', 'test']:
        count = len(final_examples[split])
        total_pairs += count
        print(f"{split}: {count:,} training pairs")
    print(f"Total: {total_pairs:,} training pairs")

    # Print sample for debugging
    print("\n📄 Sample examples:")
    for split in ['test', 'train']:
        if final_examples[split]:
            sample = final_examples[split][0]
            print(f"\n--- {split.upper()} Sample ---")
            print("Emotion:", sample['emotion'])
            print("Context:")
            print(sample['context'])
            print("\nExpected Response:")
            print(sample['response'])
            print("-" * 50)

    # Calculate expected vs actual pairs
    print(f"\n🔍 Validation check:")
    for split in ['test']:
        split_data = dataset[split]
        conversations_check = {}
        
        for example in split_data:
            conv_id = example['conv_id']
            if conv_id not in conversations_check:
                conversations_check[conv_id] = 0
            conversations_check[conv_id] += 1
        
        total_utterances = len(split_data)
        total_conversations = len(conversations_check)
        avg_length = total_utterances / total_conversations
        expected_pairs = sum(max(0, length - 1) for length in conversations_check.values())
        
        print(f"{split}: {total_conversations} conversations, avg length {avg_length:.1f}")
        print(f"Expected max pairs: {expected_pairs}, Actual pairs: {len(final_examples[split])}")

if __name__ == "__main__":
    process_dataset()