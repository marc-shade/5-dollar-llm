import sys
import os
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.smollm_config import SmolLM2Config
from configs.dataset_config import DataConfig
from data.loader import load_smollm_corpus, setup_tokenizer, tokenize_and_chunk, finalize_dataset
from training.trainer import train_moe_model
from torch.utils.data import DataLoader

def main():
    # 1. Setup Configuration
    print("🚀 Initializing SmolLM2-135M Training Setup...")
    config = SmolLM2Config()
    
    # 2. Setup Data
    # For replication, we point to the smollm-corpus.
    # Note: To replicate fully, one might need to mix subsets (fineweb-edu, stack, etc.)
    # Here we default to the cosmopedia-v2 subset as per dataset_config default, 
    # but technically we should probably iterate over mixed sources.
    # For this implementation, we use the standard loader.
    data_config = DataConfig(
        dataset_path="HuggingFaceTB/smollm-corpus",
        dataset_name="cosmopedia-v2", # Change to "fineweb-edu-dedup" or others as needed
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        seq_length=config.max_seq_len,
        streaming=True # Efficient loading
    )
    
    print(f"📁 Loading dataset: {data_config.dataset_path}/{data_config.dataset_name}")
    
    # Load and prepare
    tokenizer = setup_tokenizer(data_config)
    config.vocab_size = tokenizer.vocab_size # Ensure config matches tokenizer
    
    full_dataset = load_smollm_corpus(data_config)
    
    # Tokenize and chunk
    processed_dataset = tokenize_and_chunk(full_dataset, tokenizer, data_config)
    final_dataset = finalize_dataset(processed_dataset, data_config)
    
    # Split (simplified for streaming - typically we'd have separate validation split loaded)
    # For streaming, we can't easily split indices. We'll use the same stream for simplicity 
    # in this starter script, or better, assume the user provides a split name in config.
    # Ideally: load 'train' and 'test' splits separately.
    train_dataset = final_dataset
    
    # Load separate valid split
    val_data_config = DataConfig(
        dataset_path="HuggingFaceTB/smollm-corpus",
        dataset_name="cosmopedia-v2",
        split="train", # smollm-corpus might not have 'validation', using train slice or generic
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        seq_length=config.max_seq_len,
        streaming=True,
        num_samples=1000 # Small validation set
    )
    # Using a small subset of train as validation proxy for this script if valid split missing
    # But for real training, we'd use proper validation split.
    val_dataset = finalize_dataset(
        tokenize_and_chunk(
            load_smollm_corpus(val_data_config), tokenizer, val_data_config
        ), 
        val_data_config
    )

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # 3. Start Training
    print(f"🔥 Starting training for {config.max_steps} steps...")
    print(f"   Model: SmolLM2-135M (~135M params)")
    print(f"   Context Length: {config.max_seq_len}")
    print(f"   GQA Heads: {getattr(config, 'num_key_value_heads', config.n_heads)}")
    
    # Ensure dense mode (experts=1)
    if config.num_experts > 1:
        print(f"⚠️  Warning: num_experts={config.num_experts}. SmolLM2 is a dense model. "
              "Setting num_experts=1 for strict replication unless MoE is intended.")
        config.num_experts = 1
        config.expert_top_k = 1

    train_moe_model(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir="outputs/smollm2_replication",
        experiment_name="smollm2-135m-replication"
    )

if __name__ == "__main__":
    main()
