# port of models described in RW
# We use the bloom model as a starting point for these model.
# Please refer to the bloom models for usage instructions.

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions, QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast, TokenClassifierOutput)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_RW import RWConfig

logger = logging.get_logger(__name__)


# NOTE(Hesslow): Unfortunately we did not fuse matmul and bias during training, this means that there's one additional quantization to bfloat16 between the operations.
# In order not to degrade the quality of our HF-port, we keep these characteristics in the final model.
class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ret = input @ self.weight.T
        if self.bias is None:
            return ret
        else:
            return ret + self.bias


from einops import rearrange


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def gather_cos_sin(cos, sin, position_ids, batch_size=1):
    # TODO: lookinto gather_cos_sin for user_batch
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    # cos, sin have already been gathered
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbeddingTT(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        #        freqs = torch.mul(t.view((t.shape[0], 1)), self.inv_freq.view((1, self.inv_freq.shape[0]))) # einsum free implementation
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(self):
        return self.cos_cached, self.sin_cached


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    batch_size, target_length = input_ids_shape
    mask = torch.empty(
        (target_length, target_length + past_key_values_length),
        dtype=torch.bool,
        device=device,
    )
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def dropout_add(
    x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool
) -> torch.Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class TT_functional:
    def scaled_dot_product_attention(
        Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=False, user_batch=False
    ):
        DTYPE = Q.dtype
        L, S = Q.size(-2), K.size(-2)

        if user_batch:
            assert (
                attn_mask is not None
            ), "attn_mask must be provided if user_batch is True. L, S above will not be correct"
            # query: [num_batch, users, num_heads, head_dim]
            # key: [num_batches, users, context, head_dim]
            # value: [num_batches, users, context, head_dim]

        def make_mask(L, S, DTYPE):
            attn_mask = torch.ones(L, S, dtype=DTYPE).tril(diagonal=0)
            inverted_mask = 1.0 - attn_mask
            return inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(DTYPE).min
            )

        assert (
            is_causal or attn_mask is not None
        ), "attn_mask must be provided if is_causal is False"
        assert (
            not is_causal or attn_mask is None
        ), "attn_mask must be None if is_causal is True"

        if attn_mask is None or is_causal:
            attn_mask = make_mask(L, S, DTYPE)

        # attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(Q.size(-1), dtype=DTYPE))) + attn_mask, dim=-1)
        # attn_weight = torch.dropout(attn_weight, dropout_p, train)
        ATT = Q @ K.transpose(-2, -1) / torch.tensor(Q.size(-1) ** (1 / 2), dtype=DTYPE)
        attn_weight = F.softmax(ATT + attn_mask, dim=-1, dtype=DTYPE)
        attn_weight = nn.Dropout(p=dropout_p)(attn_weight)
        return attn_weight @ V


class Attention(nn.Module):
    def __init__(self, config: RWConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.use_cache = config.use_cache

        self.did_split = False

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor

        self.query_key_value = Linear(
            self.hidden_size,
            (config.n_head_kv * 2 + config.n_head) * self.head_dim,
            bias=config.bias,
        )

        self.dense = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv = config.n_head_kv

        # self.wq = Linear(self.hidden_size, self.hidden_size, bias=config.bias)
        # self.wk = Linear(self.hidden_size, self.num_kv * self.head_dim, bias=config.bias)
        # self.wv = Linear(self.hidden_size, self.num_kv * self.head_dim, bias=config.bias)
        self.wq = None
        self.wk = None
        self.wv = None
        self.bias = config.bias

    def _split_heads(
        self, fused_qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, (config.n_head_kv * 2 + config.n_head) * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim]
            key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        batch, seq_len, _ = fused_qkv.shape
        qkv = fused_qkv.view(
            batch, seq_len, -1, self.num_heads // self.num_kv + 2, 64
        )  # [batch, seq_len, group(8), num_heads(18), head_dim(64)]
        q = qkv[:, :, :, :-2]  # [batch, seq_len, group(8), num_heads(16), head_dim(64)]
        k = qkv[:, :, :, [-2]]  # [batch, seq_len, group(8), num_heads(1), head_dim(64)]
        v = qkv[:, :, :, [-1]]  # [batch, seq_len, group(8), num_heads(1), head_dim(64)]
        k = torch.broadcast_to(k, q.shape)
        v = torch.broadcast_to(v, q.shape)

        q, k, v = [
            rearrange(
                x,
                "batch seq_len group num_heads head_dim ->\
                batch seq_len (group num_heads) head_dim",
                head_dim=self.head_dim,
            )
            for x in [q, k, v]
        ]
        return q, k, v

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimenstion

        Args:
            x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def split_qkv_weights(self):
        # breakpoint()

        # load wq, wk, wv from query_key_value
        self.wq = Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.wk = Linear(self.hidden_size, self.num_kv * self.head_dim, bias=self.bias)
        self.wv = Linear(self.hidden_size, self.num_kv * self.head_dim, bias=self.bias)

        self.wq.weight = nn.Parameter(
            self.query_key_value.weight.reshape(
                self.num_kv, self.num_heads // self.num_kv + 2, 64, -1
            )[:, :-2]
            .reshape(self.num_heads * self.head_dim, self.hidden_size)
            .clone()
        )

        self.wk.weight = nn.Parameter(
            self.query_key_value.weight.reshape(
                self.num_kv, self.num_heads // self.num_kv + 2, 64, -1
            )[:, [-2]]
            .reshape(self.num_kv * self.head_dim, self.hidden_size)
            .clone()
        )

        self.wv.weight = nn.Parameter(
            self.query_key_value.weight.reshape(
                self.num_kv, self.num_heads // self.num_kv + 2, 64, -1
            )[:, [-1]]
            .reshape(self.num_kv * self.head_dim, self.hidden_size)
            .clone()
        )

        self.did_split = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        cos=None,
        sin=None,
    ):
        batch, seq_len, _ = hidden_states.shape

        query_layer = self.wq(hidden_states).view(
            batch, seq_len, -1, self.num_heads // self.num_kv, 64
        )
        key_layer = self.wk(hidden_states).view(batch, seq_len, -1, 1, 64)
        value_layer = self.wv(hidden_states).view(batch, seq_len, -1, 1, 64)
        key_layer = torch.broadcast_to(key_layer, query_layer.shape)
        value_layer = torch.broadcast_to(value_layer, query_layer.shape)

        query_layer, key_layer, value_layer = [
            rearrange(
                x,
                "batch seq_len group num_heads head_dim ->\
                batch seq_len (group num_heads) head_dim",
                head_dim=self.head_dim,
            )
            for x in [query_layer, key_layer, value_layer]
        ]

        # ###
        # fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # #  3 x [batch_size, seq_length, num_heads, head_dim]
        # (query_layer2, key_layer2, value_layer2) = self._split_heads(fused_qkv)
        # print("q mse:", torch.mean((query_layer2 - query_layer)**2), "shape:", query_layer.shape, query_layer2.shape)
        # print("k mse:", torch.mean((key_layer2 - key_layer)**2), "shape:", key_layer.shape, key_layer2.shape)
        # print("v mse:", torch.mean((value_layer2 - value_layer)**2), "shape:", value_layer.shape, value_layer2.shape)
        # #breakpoint()
        # #query_layer, key_layer, value_layer = query_layer2, key_layer2, value_layer2
        # ###
        # qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv + 2, 64)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, q_length, self.head_dim
        )
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads,
            q_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, q_length, self.head_dim
        )

        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        kv_length = key_layer.size(1)

        if self.use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        query_layer_ = query_layer.reshape(
            batch_size, self.num_heads, -1, self.head_dim
        )
        key_layer_ = key_layer.reshape(batch_size, self.num_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(
            batch_size, self.num_heads, -1, self.head_dim
        )

        attn_output = TT_functional.scaled_dot_product_attention(
            query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True
        )

        x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)

        outputs = (output_tensor, present)

        return outputs


class MLP(nn.Module):
    def __init__(self, config: RWConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = Linear(hidden_size, 4 * hidden_size, bias=config.bias)
        self.act = nn.GELU()
        self.dense_4h_to_h = Linear(4 * hidden_size, hidden_size, bias=config.bias)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: RWConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.num_heads = config.n_head
        self.self_attention = Attention(config)

        self.mlp = MLP(config)

        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        self.hidden_dropout = config.hidden_dropout

        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        cos=None,
        sin=None,
    ):
        # breakpoint()
        ln_attn = self.ln_attn(hidden_states)
        ln_mlp = self.ln_mlp(hidden_states)

        residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            ln_attn,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            cos=cos,
            sin=sin,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(ln_mlp)

        output = dropout_add(
            mlp_output + attention_output,
            residual,
            self.config.hidden_dropout,
            training=self.training,
        )

        if self.config.use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class RWPreTrainedModel(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h.*.self_attention.scale_mask_softmax.causal_mask",
        r"lm_head.weight",
    ]
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    tt_models.
    """

    config_class = RWConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear) or isinstance(module, Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False):
        if isinstance(module, RWModel):
            module.gradient_checkpointing = value

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_rw_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )


class RWModel(RWPreTrainedModel):
    def __init__(self, config: RWConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.alibi = config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        self.use_cache = config.use_cache
        self.rotary_emb = RotaryEmbeddingTT(config.head_dim, 2048, 10000, None)
        self.batch_users = config.batch_users

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        past_key_values_length: int,
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def split_qkv_weights(self):
        for h in self.h:
            h.self_attention.split_qkv_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids=None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        presents = () if use_cache else None

        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), device=hidden_states.device
            )
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        if self.alibi:
            alibi = build_alibi_tensor(
                attention_mask, self.num_heads, dtype=hidden_states.dtype
            )
        else:
            alibi = None

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        # Invert mask
        if attention_mask is not None:
            attention_mask = 1.0 - attention_mask
            attention_mask.masked_fill_(
                attention_mask.to(torch.bool), torch.finfo(hidden_states.dtype).min
            )
            if self.batch_users:
                num_batch, users, kv_length = attention_mask.size()
                attention_mask = attention_mask.view(
                    num_batch, users, 1, kv_length
                ).expand(num_batch, users, self.head_total, kv_length)

        # Rotary Embeddings
        cos, sin = self.rotary_emb()
        cos, sin = gather_cos_sin(cos, sin, position_ids, batch_size=batch_size)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                alibi=alibi,
                cos=cos,
                sin=sin,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states] if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )


class RWForCausalLM(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h.*.self_attention.scale_mask_softmax.causal_mask",
        r"lm_head.weight",
    ]

    def __init__(self, config: RWConfig):
        super().__init__(config)
        self.transformer = RWModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to our's format if needed
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_rw_cache(past)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids=None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length),
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self,
        past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        beam_idx: torch.LongTensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(
            past, batch_size=len(beam_idx)
        )

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device)
            for layer_past in past
            for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_rw_cache(reordered_past)
