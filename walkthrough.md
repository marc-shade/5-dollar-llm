# SmolLM2-135M Replication Walkthrough
This document outlines the changes made to replicate the SmolLM2-135M model and the next steps required to train it.

## Changes Made

### 1. Research & Configuration
- **Researched Specs**: Confirmed SmolLM2-135M has 135M parameters, 30 layers, 9 heads, 576 hidden dimension, and uses Grouped Query Attention (GQA).
- **Created Config**: [smollm_config.py](configs/smollm_config.py)
  - Defines `SmolLM2Config` with exact specifications.
  - Sets `num_key_value_heads=3` for GQA (3 groups of 3 heads).

### 2. Implementation Updates
- **Grouped Query Attention (GQA)**: 
  - Refactored `models/layers.py` to `MultiHeadAttention` to support GQA.
  - Added logic to project distinct K/V heads and repeat them to match Query heads during attention.
- **Model Integration**:
  - Updated `models/moe_llm.py` to pass `num_key_value_heads` from the config to the transformer blocks.

### 3. Training Script
- **Created Entry Point**: [train_smollm2.py](scripts/train_smollm2.py)
  - Loads the new config and `HuggingFaceTB/smollm-corpus`.
  - Initializes the tokenizer (SmolLM-135M).
  - Starts the training loop using the existing `train_moe_model` infrastructure (configured for dense training).

### 4. Version Control
- **Branch**: Created functionality on new branch `feat/smollm2-replication`.

## Next Steps

Since execution was skipped on the current environment, the following steps are required to proceed:

1.  **Environment Setup**: Ensure you are on a machine with CUDA/MPS support and sufficient RAM/VRAM.

2.  **Verify Setup (Quick Check)**:
    Run the verification script to ensure the model initializes with the correct parameter count (~135M).
    ```bash
    python scripts/verify_smollm_config.py
    ```

3.  **Start Training**:
    Run the training script. Warning: This will start downloading the dataset (streaming mode) and training.
    ```bash
    python scripts/train_smollm2.py
    ```

4.  **Monitor**:
    Watch for loss convergence. The script saves metrics to `outputs/smollm2_replication`.

## Potential Future Work
- **dataset_config.py**: Currently points to `cosmopedia-v2`. for full replication, you may want to implement a data mixer to sample from FineWeb-Edu, Stack, and others as per SmolLM2's curriculum.
