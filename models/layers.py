import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import MixtureOfExperts


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(
            dim=dim, max_seq_len=max_seq_len, base=10000
        )

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        max_seq_len: int, 
        dropout: float = 0.1,
        num_key_value_heads: Optional[int] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else n_heads
        self.d_k = d_model // n_heads
        
        # Verify GQA compatibility
        if n_heads % self.num_key_value_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")
            
        self.num_key_value_groups = n_heads // self.num_key_value_heads

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_key_value_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_key_value_heads * self.d_k, bias=False)
        
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # Basic projection
        # Q: [B, T, H_q * D]
        # K, V: [B, T, H_kv * D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [B, T, H, D]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2) # [B, H, T, D]
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.d_k).transpose(1, 2)

        # Norms (applied on head dim) and RoPE
        # RoPE expects [B, T, H, D] but our implementation does transpose inside Rotary if needed?
        # Checking Rotary implementation: it calls torchtune Rope.
        # Rotary.forward takes x_BTHD. But in previous code it was doing permutations.
        # Previous code: 
        #   Q = self.rotary(self.q_norm(Q.transpose(1, 2))).transpose(1, 2)
        #   Here Q is [B, H, T, D].
        #   Q.transpose(1, 2) is [B, T, H, D].
        #   Rotary takes [B, T, H, D].
        #   So let's stick to that pattern.
        
        q = self.q_norm(q.transpose(1, 2)) # [B, T, H, D]
        k = self.k_norm(k.transpose(1, 2)) # [B, T, H_kv, D]
        
        q = self.rotary(q).transpose(1, 2) # Back to [B, H, T, D]
        k = self.rotary(k).transpose(1, 2) # Back to [B, H_kv, T, D]
        # V no RoPE
        
        # Handle GQA repetition
        if self.num_key_value_groups > 1:
            k = torch.repeat_interleave(k, dim=1, repeats=self.num_key_value_groups)
            v = torch.repeat_interleave(v, dim=1, repeats=self.num_key_value_groups)

        # Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape back to [B, T, H * D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        qk_rope_dim: int,
        qk_nope_dim: int,
        kv_lora_rank: int,
        v_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qk_dim = qk_rope_dim + qk_nope_dim
        self.qk_rope_dim, self.qk_nope_dim = qk_rope_dim, qk_nope_dim
        self.kv_lora_dim = kv_lora_rank
        self.v_dim = v_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.query = nn.Linear(d_model, n_heads * self.qk_dim, bias=False)
        self.compressed_kv = nn.Linear(d_model, kv_lora_rank + qk_rope_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
        self.decompressed_kv = nn.Linear(
            kv_lora_rank, n_heads * (qk_nope_dim + v_dim), bias=False
        )
        self.w_o = nn.Linear(v_dim * n_heads, d_model, bias=False)
        self.rotary = Rotary(qk_rope_dim, max_seq_len)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.size(0), x.size(1)

        # Query part of the mla
        q = self.query.forward(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.qk_dim)
        q_nope, q_rope = torch.split(q, (self.qk_nope_dim, self.qk_rope_dim), dim=-1)
        q_rope = self.rotary.forward(q_rope)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # KV part of the mla
        kv = self.compressed_kv.forward(x)
        kv, k_rope = torch.split(kv, (self.kv_lora_dim, self.qk_rope_dim), dim=-1)
        ## k rope part
        k_rope = k_rope.view(batch_size, seq_len, 1, self.qk_rope_dim)
        k_rope = self.rotary.forward(k_rope)
        ## v and k part
        kv = self.kv_norm.forward(kv)
        kv = self.decompressed_kv.forward(kv)
        kv = kv.view(batch_size, seq_len, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope, v = torch.split(kv, (self.qk_nope_dim, self.v_dim), dim=-1)
        k = torch.cat([k_nope, k_rope.expand(-1, -1, self.n_heads, -1)], dim=-1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        return self.w_o.forward(attn_output)


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        use_mla: bool,
        qk_rope_dim: int | None,
        qk_nope_dim: int | None,
        kv_lora_rank: int | None,
        v_dim: int | None,
        max_seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        num_key_value_heads: Optional[int] = None,
    ):
        super().__init__()

        # Attention layer
        if use_mla:
            self.attention = MultiHeadLatentAttention(
                d_model,
                n_heads,
                qk_rope_dim,
                qk_nope_dim,
                kv_lora_rank,
                v_dim,
                max_seq_len,
                dropout,
            )
        else:
            self.attention = MultiHeadAttention(
                d_model, 
                n_heads, 
                max_seq_len, 
                dropout,
                num_key_value_heads=num_key_value_heads
            )

        # MoE layer
        self.feed_forward = MixtureOfExperts(d_model, d_ff, num_experts, top_k, dropout)

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss

