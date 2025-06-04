"""
SynthWhisperer Model Training
Fine-tunes a small language model on synthesizer knowledge.
"""

import json
import os
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch
from torch.utils.data import Dataset
import numpy as np

@dataclass
class SynthWhispererConfig:
    """Configuration for SynthWhisperer training"""
    model_name: str = "microsoft/DialoGPT-small"  # Good base for conversational AI
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./synthwhisperer_model"
    
class SynthDataset(Dataset):
    """Dataset for SynthWhisperer training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load and process training data
        with open(data_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                self.examples.append(example)
        
        # Create conversation format
        self.processed_examples = self._process_examples()
    
    def _process_examples(self) -> List[str]:
        """Convert examples to conversation format"""
        processed = []
        
        for example in self.examples:
            user_msg = example['user']
            assistant_msg = example['assistant']
            
            # Format as conversation with special tokens
            conversation = f"<|user|>{user_msg}<|assistant|>{assistant_msg}<|endoftext|>"
            processed.append(conversation)
        
        return processed
    
    def __len__(self):
        return len(self.processed_examples)
    
    def __getitem__(self, idx):
        text = self.processed_examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class SynthWhispererTrainer:
    """Main trainer for SynthWhisperer model"""
    
    def __init__(self, config: SynthWhispererConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Add special tokens for conversation format
        special_tokens = ["<|user|>", "<|assistant|>", "<|endoftext|>"]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model and resize embeddings for new tokens
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        
    def prepare_datasets(self, train_data_path: str, eval_split: float = 0.1):
        """Prepare training and evaluation datasets"""
        # Load all data
        full_dataset = SynthDataset(train_data_path, self.tokenizer, self.config.max_length)
        
        # Split into train/eval
        dataset_size = len(full_dataset)
        eval_size = int(dataset_size * eval_split)
        train_size = dataset_size - eval_size
        
        self.train_dataset, self.eval_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, eval_size]
        )
        
        print(f"Training examples: {len(self.train_dataset)}")
        print(f"Evaluation examples: {len(self.eval_dataset)}")
    
    def train(self, train_data_path: str):
        """Train the SynthWhisperer model"""
        # Prepare datasets
        self.prepare_datasets(train_data_path)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=0.01,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=100,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb
            learning_rate=self.config.learning_rate,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        print(f"Training complete! Model saved to {self.config.output_dir}")
    
    def generate_response(self, user_input: str, max_length: int = 150) -> str:
        """Generate a response using the trained model"""
        # Format input
        prompt = f"<|user|>{user_input}<|assistant|>"
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode("<|endoftext|>")[0],
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract just the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
            if "<|endoftext|>" in response:
                response = response.split("<|endoftext|>")[0]
        
        return response.strip()

class SynthWhispererInference:
    """Inference wrapper for the trained SynthWhisperer model"""
    
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def get_synth_advice(self, user_request: str) -> str:
        """Get synthesizer advice from user request"""
        # Format input
        prompt = f"<|user|>{user_request}<|assistant|>"
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + 200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode("<|endoftext|>")[0],
            )
        
        # Decode and clean response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1]
            if "<|endoftext|>" in response:
                response = response.split("<|endoftext|>")[0]
        
        return response.strip()

def main():
    """Train SynthWhisperer model"""
    config = SynthWhispererConfig()
    trainer = SynthWhispererTrainer(config)
    
    # Train the model
    data_path = "/home/swai/secret-sauce/synthwhisperer/synthwhisperer_training_data.jsonl"
    trainer.train(data_path)
    
    # Test the model
    print("\nTesting the trained model:")
    test_queries = [
        "I need a warm bass sound for a house track",
        "Create a bright lead for synthwave",
        "What does filter cutoff do?",
        "Make me something aggressive and harsh",
    ]
    
    for query in test_queries:
        response = trainer.generate_response(query)
        print(f"\nUser: {query}")
        print(f"SynthWhisperer: {response}")

if __name__ == "__main__":
    main()