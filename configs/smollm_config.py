from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class SmolLM2Config:
    # Model architecture (SmolLM2-135M specs)
    d_model: int = 576
    n_heads: int = 9
    n_layers: int = 30
    d_ff: int = 1536  # Standard expansion
    use_mla: bool = False # Multi-Head Latent Attention not used in SmolLM2
    
    # QA/GQA parameters
    num_key_value_heads: int = 3  # Standard GQA for SmolLM2 (9 heads / 3 groups)
    
    # Not used but required for compatibility if using same model class (or we define a new one)
    qk_rope_dim: int | None = None
    qk_nope_dim: int | None = None
    kv_lora_rank: int | None = None
    v_dim: int | None = None
    
    # Batch size & Training
    batch_size: int = 64 # Adjust based on GPU memory (64 H100s used large batch)
    max_steps: int = 100000 # Placeholder for 2T tokens depending on batch size
    
    # Training parameters
    gradient_accumulation_steps: int = 1 
    muon_lr: float = 0.005 # Placeholder, need to verify or tune
    muon_momentum: float = 0.95 
    adamw_lr: float = 0.001 # SmolLM2 used AdamW
    warmup_ratio: float = 0.01 # Standard
    
    # Data parameters
    max_seq_len: int = 2048 # or 4096
    vocab_size: int = 49152 # SmolLM2 tokenizer
    
    # Evaluation
    eval_every: int = 100
    eval_steps: int = 100
    
    # Regularization
    weight_decay: float = 0.01
    dropout: float = 0.0
    grad_clip: float = 1.0
    
    # Technical
    use_amp: bool = True
    log_milestones: Tuple[int, ...] = (1000, 5000, 10000)
    
    # MoE parameters - set to simulate dense if reused
    num_experts: int = 1
    expert_top_k: int = 1
    load_balancing_weight: float = 0.0

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
