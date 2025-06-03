import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, DatasetDict
import mlflow
import mlflow.transformers
import numpy as np
import gc
import random
from torch.utils.data import DataLoader

# Optimized settings for full dataset training
MODEL_NAME = "gpt2"
MAX_LENGTH = 512  # Keep reasonable length for quality
EMOTION_TOKENS = [f"[{e}]" for e in [
    "sentimental", "worried", "excited", "nervous", "lonely",
    "sad", "terrified", "hopeful", "ashamed", "grateful",
    "angry", "surprised", "disgusted", "proud", "caring",
    "trusting", "annoyed", "anticipating", "nostalgic", "confident",
    "furious", "jealous", "prepared", "embarrassed", "content",
    "devastated", "impressed", "apprehensive", "guilty", "joyful"
]]

# Special formatting tokens
SPECIAL_TOKENS = [
    "<|emotion|>", "<|context|>", "<|response|>", "<|endoftext|>"
]

DATA_DIR = "../data/processed"
MODEL_OUTPUT_DIR = "../models/gpt2_finetuned_improved"
MLFLOW_TRACKING_URI = "file:///gpfs/home/bol2142/GenAI_Text_Chatbot/mlflow_runs"
EXPERIMENT_NAME = "Empathetic Echoes - GPT2 Fine-tuning Improved"
SEED = 42

def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def aggressive_memory_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def format_input_improved(example):
    """Improved formatting with clear structure and special tokens"""
    emotion = example['emotion'].strip()
    context = example['context'].strip()
    response = example['response'].strip()
    
    # Clean and format the text with better structure
    formatted_text = (
        f"<|emotion|>[{emotion}]<|context|>{context}<|response|>{response}<|endoftext|>"
    )
    
    return {"text": formatted_text}

def tokenize_function_streaming(examples):
    """Memory-efficient streaming tokenization"""
    # Tokenize without padding to save memory
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # No padding during preprocessing
        return_special_tokens_mask=False,
        return_tensors=None  # Return lists, not tensors
    )
    
    # Labels are the same as input_ids for language modeling
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    
    return tokenized

class StreamingDataCollator(DataCollatorForLanguageModeling):
    """Memory-efficient streaming data collator"""
    
    def __call__(self, examples):
        # Convert to tensors only when needed
        input_ids = []
        labels = []
        
        for ex in examples:
            input_ids.append(torch.tensor(ex["input_ids"], dtype=torch.long))
            labels.append(torch.tensor(ex["labels"], dtype=torch.long))
        
        # Pad to the maximum length in this specific batch only
        max_len = max(len(ids) for ids in input_ids)
        max_len = min(max_len, MAX_LENGTH)  # Cap at MAX_LENGTH
        
        # Pad sequences
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
            # Truncate if too long
            ids = ids[:max_len]
            lbls = lbls[:max_len]
            
            # Calculate padding needed
            pad_len = max_len - len(ids)
            
            if pad_len > 0:
                # Pad input_ids
                padded_ids = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
                # Pad labels with -100 (ignore_index)
                padded_lbls = torch.cat([lbls, torch.full((pad_len,), -100, dtype=torch.long)])
            else:
                padded_ids = ids
                padded_lbls = lbls
            
            # Create attention mask
            attention_mask = (padded_ids != self.tokenizer.pad_token_id).long()
            
            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_lbls)
            attention_masks.append(attention_mask)
        
        # Stack into batch tensors
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(padded_labels)
        }
        
        return batch

class MemoryOptimizedTrainer(Trainer):
    """Simplified memory-optimized trainer"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_frequency = 100  # Clean memory every N steps
        self.step_count = 0
    
    def log(self, logs, *args, **kwargs):
        """Override logging to add memory cleanup"""
        super().log(logs, *args, **kwargs)  # Forward all args/kwargs to parent
        self.step_count += 1
        if self.step_count % self.cleanup_frequency == 0:
            aggressive_memory_cleanup()
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Skip evaluation during training to save memory"""
        if not hasattr(self, '_allow_eval'):
            # Return dummy metrics during training
            return {
                f"{metric_key_prefix}_loss": 0.0,
                f"{metric_key_prefix}_runtime": 0.0,
                f"{metric_key_prefix}_samples_per_second": 0.0,
                f"{metric_key_prefix}_steps_per_second": 0.0,
            }
        else:
            # Only evaluate when explicitly allowed
            return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    
    def save_model(self, output_dir=None, _internal_call=False):
        """Only save the final model"""
        if hasattr(self, '_allow_save'):
            super().save_model(output_dir, _internal_call)

def create_streaming_dataloader(dataset, tokenizer, batch_size=1, shuffle=True):
    """Create memory-efficient streaming dataloader"""
    data_collator = StreamingDataCollator(tokenizer=tokenizer, mlm=False)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator,
        num_workers=0,  # No multiprocessing to save memory
        pin_memory=False,
        drop_last=True
    )

def simple_end_evaluation(model, tokenizer, test_dataset, device, max_samples=200):
    """Simple evaluation at the end of training"""
    model.eval()
    data_collator = StreamingDataCollator(tokenizer=tokenizer, mlm=False)
    
    total_loss = 0.0
    total_samples = 0
    
    # Evaluate on a subset to save memory
    eval_samples = min(max_samples, len(test_dataset))
    indices = random.sample(range(len(test_dataset)), eval_samples)
    
    with torch.no_grad():
        for i in range(0, eval_samples, 1):  # Batch size 1 for evaluation
            if i >= len(indices):
                break
                
            # Get single sample
            sample = test_dataset[indices[i]]
            batch = data_collator([sample])
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            total_loss += loss.item()
            total_samples += 1
            
            # Memory cleanup every few samples
            if i % 10 == 0:
                del batch, outputs, loss
                aggressive_memory_cleanup()
    
    avg_loss = total_loss / max(total_samples, 1)
    perplexity = np.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    model.train()
    return {"eval_loss": avg_loss, "eval_perplexity": perplexity}

def main():
    set_seed(SEED)
    
    # Initial memory setup
    aggressive_memory_cleanup()
    if torch.cuda.is_available():
        # Use most of available memory but leave some buffer
        torch.cuda.set_per_process_memory_fraction(0.95)
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total")

    # Setup MLFlow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("📂 Loading FULL dataset...")
    raw = {}
    for split in ["train", "validation", "test"]:
        path = os.path.join(DATA_DIR, f"{split}.json")
        with open(path, "r", encoding="utf-8") as f:
            raw[split] = json.load(f)
    
    # Use ALL data - no limiting!
    print(f"📊 Dataset sizes:")
    print(f"   Train: {len(raw['train'])} samples")
    print(f"   Validation: {len(raw['validation'])} samples") 
    print(f"   Test: {len(raw['test'])} samples")
    
    dataset = DatasetDict({
        "train": Dataset.from_list(raw["train"]),
        "validation": Dataset.from_list(raw["validation"]),
        "test": Dataset.from_list(raw["test"])
    })

    # Clear raw data from memory
    del raw
    aggressive_memory_cleanup()

    # Load model and tokenizer
    global tokenizer
    print(f"🔄 Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    # Fix pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    all_special_tokens = {"additional_special_tokens": EMOTION_TOKENS + SPECIAL_TOKENS}
    num_added_tokens = tokenizer.add_special_tokens(all_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"✅ Added {num_added_tokens} special tokens")

    # Process dataset in streaming fashion
    print("🔄 Processing dataset (streaming mode)...")
    
    # Format the dataset
    formatted_dataset = dataset.map(
        format_input_improved, 
        num_proc=1,  # Single process to control memory
        desc="Formatting"
    )
    
    # Tokenize the dataset
    tokenized_dataset = formatted_dataset.map(
        tokenize_function_streaming,
        batched=True,
        batch_size=500,  # Process in smaller batches
        num_proc=1,
        remove_columns=formatted_dataset["train"].column_names,
        desc="Tokenizing"
    )
    
    # Clear intermediate datasets
    del dataset, formatted_dataset
    aggressive_memory_cleanup()

    # Ultra memory-efficient training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=20,  # Reasonable number of epochs
        per_device_train_batch_size=1,  # Minimal batch size
        gradient_accumulation_steps=64,  # Large accumulation for effective batch size
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        
        # Disable all intermediate operations
        do_eval=False,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        save_total_limit=0,
        load_best_model_at_end=False,
        
        # Memory optimizations
        fp16=True,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        prediction_loss_only=True,
        
        # Optimizer settings
        optim="adamw_torch",
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        
        # Disable unnecessary features
        report_to="none",
        ddp_find_unused_parameters=False,
        group_by_length=False,
        length_column_name=None,
        
        seed=SEED,
    )

    print("🚀 Starting training with FULL dataset...")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "num_train_epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "max_length": MAX_LENGTH,
            "train_size": len(tokenized_dataset["train"]),
            "val_size": len(tokenized_dataset["validation"]),
            "test_size": len(tokenized_dataset["test"]),
        })

        # Create streaming data collator
        data_collator = StreamingDataCollator(tokenizer=tokenizer, mlm=False)

        # Initialize memory-optimized trainer
        trainer = MemoryOptimizedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=None,  # No eval during training
            data_collator=data_collator,
            compute_metrics=None,
        )

        # Pre-training memory cleanup
        aggressive_memory_cleanup()
        if torch.cuda.is_available():
            print(f"GPU Memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        # Train the model with full dataset
        print("🔥 Training on FULL dataset...")
        trainer.train()

        # Final cleanup before saving
        aggressive_memory_cleanup()

        # Save the final model
        print("💾 Saving final model...")
        trainer._allow_save = True
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

        # Final evaluation
        print("🧪 Final evaluation...")
        device = next(model.parameters()).device
        test_results = simple_end_evaluation(
            model, tokenizer, tokenized_dataset["test"], device
        )
        
        mlflow.log_metrics({
            "final_test_loss": test_results["eval_loss"],
            "final_test_perplexity": test_results["eval_perplexity"]
        })

        print("✅ Training completed successfully with FULL dataset!")
        print(f"📊 Final Results:")
        print(f"   Test Loss: {test_results['eval_loss']:.4f}")
        print(f"   Test Perplexity: {test_results['eval_perplexity']:.2f}")
        print(f"   Training samples processed: {len(tokenized_dataset['train'])}")

        # Final memory cleanup
        aggressive_memory_cleanup()

if __name__ == "__main__":
    main()