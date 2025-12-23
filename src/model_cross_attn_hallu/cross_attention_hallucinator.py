import torch
from torch import nn
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, AutoModel
from transformers.activations import ACT2FN
from typing import Optional, Callable
from contextlib import nullcontext
from src.dataloaders.hallu_dataloader import HuggingDataframe2

@dataclass
class GemmaConfig:
    # hugging face model
    hfmodel_name: str = "bkai-foundation-models/vietnamese-bi-encoder"

    # model dimensions
    hidden_size: int = 768
    intermediate_size: int = 768*2
    num_hidden_layers: int = 4

    # attention
    num_attention_heads: int = 2
    num_key_value_heads: int = 2
    head_dim: Optional[int] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # mlp
    hidden_act: str = "silu"

    # normalization
    rms_norm_eps: float = 1e-6

    # RoPE
    rope_theta: float = 10000.0
    max_position_embeddings: int = 1024

    # attention behavior
    use_bidirectional_attention: bool = False
    _attn_implementation: str = "eager"

    # embeddings
    rope_theta: float = 10000.0

    # initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        # infer head_dim if not provided
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    "hidden_size must be divisible by num_attention_heads"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads

        # check GQA validity
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

class GemmaRMSNorm(nn.Module):
    def __init__(self, config:GemmaConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_eps = config.rms_norm_eps

    def forward(self, x:torch.Tensor):
        with maybe_autocast(x.device.type):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_eps)
        
        return self.weight * x

class GemmaMLP(nn.Module):
    def __init__(self, config:GemmaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def maybe_autocast(device_type:str, dtype:Optional[torch.dtype]=None, enabled:bool=True, cache_enabled:bool=True):
    if torch.is_autocast_enabled(device_type)  and  enabled:
        return torch.autocast(device_type, dtype=dtype, cache_enabled=cache_enabled)
    else:
        return nullcontext()

class Gemma2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: GemmaConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        rope_init_fn: Callable = self.compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: GemmaConfig,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_theta
        dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x:torch.Tensor, position_ids:torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):

        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @staticmethod
    def _rotate_half(tensor:torch.Tensor):
        x1 = tensor[..., :tensor.shape[-1]//2]
        x2 = tensor[..., tensor.shape[-1]//2 :]
        return torch.cat((x1,x2), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q:torch.Tensor, k:torch.Tensor, cos:torch.Tensor, sin:torch.Tensor, position_ids=None, unsqueeze_dim:int=1) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q*cos)+ (GemmaAttention._rotate_half(q) * sin)
        k_embed = (k*cos) + (GemmaAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def repeat_kv(tensor:torch.Tensor, n_rep:int) ->  torch.Tensor:
        """
            The hidden states go from 
            (batch, num_key_value_heads, seqlen, head_dim) to 
            (batch, num_attention_heads, seqlen, head_dim)
        """

        if n_rep == 1:
            return tensor
        
        batch, num_key_value_head, seqlen, head_dim = tensor.shape
        tensor = tensor[:, :, None, :, :].expand(batch, num_key_value_head, n_rep, seqlen, head_dim)
        return tensor.reshape(batch, num_key_value_head * n_rep, seqlen, head_dim)
        

    def eager_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
    ):
        key_states = GemmaAttention.repeat_kv(key, self.num_key_value_groups)
        value_states = GemmaAttention.repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = query_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_states = self.k_proj(query_states).view(hidden_shape).transpose(1, 2)
        k_states = self.q_proj(key_states).view(hidden_shape).transpose(1, 2)
        v_states = self.v_proj(key_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q_states, k_states = self.apply_rotary_pos_emb(q_states, k_states, cos, sin)

        attn_output, attn_weights = self.eager_attention_forward(
            q_states,
            k_states,
            v_states,
            dropout=0.0 if not self.training else self.attention_dropout,
            attention_mask=None,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

 

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config)
        self.post_attention_layernorm = GemmaRMSNorm(config)

    def forward(
        self, query_states: torch.Tensor, key_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs, ) -> torch.Tensor:
        
        residual = key_states
        query_states = self.input_layernorm(query_states)
        key_states= self.input_layernorm(key_states)

        # Self Attention
        hidden_states, _ = self.attn(
            query_states = query_states,
            key_states = key_states, 
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
   

class GammaModel(nn.Module):
    def __init__(self, config:GemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.key_layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.query_layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config)
        self.rotary_emb = Gemma2RotaryEmbedding(config)

    def forward(self, query_embed:torch.Tensor, key_embed:torch.Tensor):
        query_posistional_ids  = torch.arange(0, query_embed.size(1), device=query_embed.device).unsqueeze(0)
        query_positional_embeddings = self.rotary_emb(query_embed,  query_posistional_ids)

        key_posistional_ids  = torch.arange(0, key_embed.size(1), device=query_embed.device).unsqueeze(0)
        key_positional_embeddings = self.rotary_emb(key_embed, key_posistional_ids)

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=query_embed.dtype)
        query_embed = query_embed * normalizer
        key_embed = key_embed * normalizer

        for query_layer, key_layer in zip(self.key_layers, self.query_layers):
            query_new = query_layer(query_embed, key_embed, query_posistional_ids, query_positional_embeddings)
            key_new = key_layer(key_embed, query_embed, key_posistional_ids, key_positional_embeddings)
            query_embed = query_new
            key_embed  = key_new
        
        query_embed = self.norm(query_embed)
        key_embed = self.norm(key_embed)
        hidden_states = torch.mean(query_embed, dim=1) + torch.mean(key_embed, dim=1)
        return hidden_states / 2
    
class CrossAttentionHallucination(nn.Module):
    def __init__(self, config:GemmaConfig, n_cls:int=3) -> None:
        super().__init__()
        self.model = GammaModel(config)
        self.cls = nn.Sequential(GemmaMLP(config), ACT2FN[config.hidden_act], 
                                 nn.Linear(config.hidden_size, n_cls))
        
    def forward(self, tokens:torch.Tensor):
        assert tokens.dim() == 3 or tokens.dim()== 4, "The model taking a pair of text tokens tensor"
        dim_split = 0
        if tokens.dim() == 4: # first for batch
            assert tokens.size(1) == 2, "The model taking a pair of text tokens tensor"
            dim_split = 1
        if tokens.dim() == 3:
            assert tokens.size(0) == 2, "The model taking a pair of text tokens tensor"
            dim_split = 0

        half = tokens.size(dim_split)//2
        query, key = tokens.split([half, half], dim=dim_split)
        hidden = self.model(query, key)
        return self.cls(hidden)
       # tokenizer
    padding :bool= True
    truncation:bool = True
    max_length: int = 512
 
def inference_pipeline(x1,x2, *, tokenizer:PreTrainedTokenizerBase, hfmodel:AutoModel, crosattnhllu:CrossAttentionHallucination, hfds2:HuggingDataframe2, get_prob:bool=True):
    tensors = hfds2.tensorize_fn(x1,x2, tokenizer, hfmodel)   
    if get_prob:
        return crosattnhllu(tensors)
    else:
        return torch.argmax(crosattnhllu(tensors), dim=-1)