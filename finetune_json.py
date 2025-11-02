#!/usr/bin/env python3

"""
Fine-tuning script for Unsloth with JSON dataset format.
Supports JSON files with format: {"question": "...", "output": ["..."]}

Usage:
    python finetune_json.py --json_file data.json --model_name unsloth/gemma-3-270m-it
"""

import argparse
import json
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig


def load_json_dataset(json_file_path):
    """Load JSON file and convert to HuggingFace dataset."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both single JSON object and list of objects
    if isinstance(data, dict):
        data = [data]
    
    # Ensure all entries have required fields
    formatted_data = []
    for item in data:
        if 'question' not in item:
            raise ValueError(f"Missing 'question' field in item: {item}")
        if 'output' not in item:
            raise ValueError(f"Missing 'output' field in item: {item}")
        
        # Convert output list to string
        output_str = item['output']
        if isinstance(output_str, list):
            output_str = '\n'.join(str(x) for x in output_str)
        
        formatted_data.append({
            'question': item['question'],
            'output': output_str
        })
    
    dataset = Dataset.from_list(formatted_data)
    return dataset


def format_dataset_func(examples, tokenizer):
    """Format dataset for training."""
    questions = examples["question"]
    outputs = examples["output"]
    
    EOS_TOKEN = tokenizer.eos_token
    
    texts = []
    for question, output in zip(questions, outputs):
        # Simple prompt format - you can customize this
        text = f"Question: {question}\n\nAnswer: {output}{EOS_TOKEN}"
        texts.append(text)
    
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with JSON dataset")
    
    # Data arguments
    parser.add_argument('--json_file', type=str, required=True,
                       help='Path to JSON file with question/output format')
    parser.add_argument('--model_name', type=str, 
                       default='unsloth/gemma-3-270m-it',
                       help='Model name to fine-tune')
    
    # Model arguments
    parser.add_argument('--max_seq_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--load_in_4bit', action='store_true',
                       help='Use 4-bit quantization')
    
    # LoRA arguments
    parser.add_argument('--r', type=int, default=16,
                       help='LoRA rank (8, 16, 32, 64, 128)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                       help='LoRA dropout rate')
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth',
                       help='Gradient checkpointing method')
    
    # Training arguments
    parser.add_argument('--per_device_train_batch_size', type=int, default=2,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--warmup_steps', type=int, default=5,
                       help='Warmup steps')
    parser.add_argument('--max_steps', type=int, default=100,
                       help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for checkpoints')
    parser.add_argument('--logging_steps', type=int, default=1,
                       help='Logging frequency')
    parser.add_argument('--save_steps', type=int, default=50,
                       help='Save checkpoint every N steps')
    parser.add_argument('--seed', type=int, default=3407,
                       help='Random seed')
    
    # Save arguments
    parser.add_argument('--save_model', action='store_true',
                       help='Save the fine-tuned model')
    parser.add_argument('--save_path', type=str, default='finetuned_model',
                       help='Path to save the model')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Starting Fine-tuning with Unsloth")
    print("=" * 60)
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.json_file}...")
    dataset = load_json_dataset(args.json_file)
    print(f"✅ Loaded {len(dataset)} examples")
    
    # Load model
    print(f"\n🤖 Loading model: {args.model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    print("✅ Model loaded")
    
    # Setup LoRA
    print(f"\n🧠 Setting up LoRA (r={args.r}, alpha={args.lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    print("✅ LoRA configured")
    
    # Format dataset
    print("\n🔄 Formatting dataset...")
    dataset = dataset.map(
        lambda examples: format_dataset_func(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    print("✅ Dataset formatted")
    
    # Setup trainer
    print("\n🎓 Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.output_dir,
            save_steps=args.save_steps,
            report_to="none",
        ),
    )
    print("✅ Trainer configured")
    
    # Train
    print(f"\n🚂 Starting training for {args.max_steps} steps...")
    trainer_stats = trainer.train()
    print("✅ Training completed!")
    
    # Save model
    if args.save_model:
        print(f"\n💾 Saving model to {args.save_path}...")
        model.save_pretrained_merged(args.save_path, tokenizer)
        print(f"✅ Model saved to {args.save_path}")
    else:
        print("\n⚠️  Model not saved (use --save_model to save)")
    
    print("\n" + "=" * 60)
    print("🎉 Fine-tuning completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

