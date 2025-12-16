import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.smollm_config import SmolLM2Config
from models.moe_llm import MoEMinimalLLM # Using this as base class if compatible, or checking generic

def verify_smollm_setup():
    print("Locked and loaded: Verifying SmolLM2-135M Setup...")
    
    # 1. Verify Config
    config = SmolLM2Config()
    print(f"Config loaded: {config}")
    
    expected_params = 135_000_000
    # Approx check based on dimensions
    # d_model=576, layers=30, heads=9
    # This is a rough check, we'll count actual params below
    
    # 2. Initialize Model
    # Note: Using MoEMinimalLLM but with num_experts=1 (dense) as per config default
    try:
        model = MoEMinimalLLM(config)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # 3. Count Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # SmolLM2 is ~135M. 
    # Let's see how close we are.
    diff = abs(total_params - 135_000_000)
    print(f"Difference from target (135M): {diff:,}")
    
    # 4. Forward Pass
    print("Running forward pass...")
    dummy_input = torch.randint(0, config.vocab_size, (1, 128))
    try:
        output, _ = model(dummy_input)
        print(f"Forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return

    print("\n✅ Verification Complete: configuration looks valid.")

if __name__ == "__main__":
    verify_smollm_setup()
