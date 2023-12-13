# port of models described in RW
# We use the bloom model as a starting point for these model.
# Please refer to the bloom models for usage instructions.

import math
import warnings
from time import time
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn import functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from tt_models.falcon40b.configuration_RW import RWConfig

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

    def reset_parameters(self):
        return


from einops import rearrange


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    divided_dim = torch.div(x.shape[-1], 2, rounding_mode="floor")
    x1 = x[..., :divided_dim]
    x2 = x[..., divided_dim:]
    return torch.cat((-x2, x1), dim=-1)


def gather_cos_sin(cos, sin, position_ids):
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


def apply_rotary_pos_emb_q(q, cos, sin):
    # cos, sin have already been gathered
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed


def apply_rotary_pos_emb_k(k, cos, sin):
    # cos, sin have already been gathered
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed


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

    def flash_decode_attention(Q, Ks, Vs, attn_masks):
        # flash attention for decoding with userbatching
        # example here is a usage where we decouple attention with kv_cache and new kv
        # Q: [batch, users, num_heads, head_dim]
        # Ks: List of Ks. For example, {k_old: [batch, users, seq_len, head_dim], k_new: [batch, users, 1, head_dim]}
        # Vs: List of Vs. For example {v_old: [batch, users, seq_len, head_dim], v_new: [batch, users, 1, head_dim]}
        # attn_masks: List of attn_masks. For example {mask_old: [batch, users, n_head, seq_len], mask_new: [batch, users, n_head, 1](None)}
        # note original flash attention do this sequentially to do kernels fusion. We do not do that here
        # We have a modified version where we first gather all stats and do another for loop for adding V matmul
        DTYPE = Q.dtype
        batch, users, n_head, head_dim = Q.size()
        # global_max_m = torch.ones(batch, users, n_head, 1)*torch.finfo(DTYPE).min
        all_local_attns = []
        all_local_max_m = []
        all_local_row_sum_l = []
        for K, mask in zip(Ks, attn_masks):
            ATT = (
                Q
                @ K.transpose(-2, -1)
                / torch.tensor(Q.size(-1) ** (1 / 2), dtype=DTYPE)
            )
            if mask is not None:
                ATT = ATT + mask
            local_max_m = torch.max(ATT, dim=-1, keepdim=True)[0]
            ATT = torch.exp(ATT - local_max_m)
            local_row_sum_l = torch.sum(ATT, dim=-1, keepdim=True)
            all_local_attns.append(ATT)
            all_local_max_m.append(local_max_m)
            all_local_row_sum_l.append(local_row_sum_l)
        output = torch.zeros_like(Q)
        scale = torch.zeros_like(all_local_row_sum_l[0])
        # find global_max_m from all_local_max_m
        global_max_m = torch.max(
            torch.cat(all_local_max_m, dim=-1), dim=-1, keepdim=True
        )[0]
        for V, ATT, local_max_m, local_row_sum_l in zip(
            Vs, all_local_attns, all_local_max_m, all_local_row_sum_l
        ):
            factor = torch.exp(local_max_m - global_max_m)
            scale += factor * local_row_sum_l
            output += factor * (ATT @ V)
        output = output / scale
        return output


class Attention(nn.Module):
    def __init__(self, config: RWConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.use_cache = config.use_cache
        self.split_mq = config.split_mq
        self.efficient_40b = config.efficient_40b
        self.flash_decode = config.flash_decode
        self.user_rows = config.user_rows

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

        self.query_key_value = None
        self.did_split = True

    def split_group_qkv_proj(self):
        group = self.num_heads // self.num_kv
        self.wq_list = nn.ModuleList(
            [
                Linear(self.hidden_size, group * self.head_dim, bias=self.bias)
                for _ in range(self.num_kv)
            ]
        )
        self.wk_list = nn.ModuleList(
            [
                Linear(self.hidden_size, self.head_dim, bias=self.bias)
                for _ in range(self.num_kv)
            ]
        )
        self.wv_list = nn.ModuleList(
            [
                Linear(self.hidden_size, self.head_dim, bias=self.bias)
                for _ in range(self.num_kv)
            ]
        )
        # TODO: load the weights from wq
        for i, module in enumerate(self.wq_list):
            module.weight = nn.Parameter(
                self.wq.weight[
                    group * i * self.head_dim : group * (i + 1) * self.head_dim, :
                ].clone()
            )
        for i, module in enumerate(self.wk_list):
            module.weight = nn.Parameter(
                self.wk.weight[i * self.head_dim : (i + 1) * self.head_dim, :].clone()
            )
        for i, module in enumerate(self.wv_list):
            module.weight = nn.Parameter(
                self.wv.weight[i * self.head_dim : (i + 1) * self.head_dim, :].clone()
            )

        self.dense_list = nn.ModuleList(
            [
                Linear(self.hidden_size, self.hidden_size // 8, bias=self.bias)
                for _ in range(self.num_kv)
            ]
        )
        for i, module in enumerate(self.dense_list):
            module.weight = nn.Parameter(
                self.dense.weight[
                    :, i * self.hidden_size // 8 : (i + 1) * self.hidden_size // 8
                ].clone()
            )

        self.wq = None
        self.wk = None
        self.wv = None
        self.dense = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos,
        sin,
        attention_mask: torch.Tensor = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        kv_read_mask=None,
        kv_write_mask=None,
    ):
        if self.efficient_40b:
            if self.flash_decode:
                return self.efficient_forward_flash_decode(
                    hidden_states,
                    cos,
                    sin,
                    attention_mask,
                    layer_past,
                    head_mask,
                    output_attentions,
                    kv_read_mask,
                    kv_write_mask,
                )
            else:
                return self.efficient_forward_split_group_qkv_proj(
                    hidden_states,
                    cos,
                    sin,
                    attention_mask,
                    layer_past,
                    head_mask,
                    output_attentions,
                    kv_read_mask,
                    kv_write_mask,
                )
        elif self.split_mq:
            if self.user_rows > 1:
                return self.split_mq_forward_users(
                    hidden_states,
                    cos,
                    sin,
                    attention_mask,
                    layer_past,
                    head_mask,
                    output_attentions,
                )
            else:
                return self.split_mq_forward(
                    hidden_states,
                    cos,
                    sin,
                    attention_mask,
                    layer_past,
                    head_mask,
                    output_attentions,
                )
        else:
            return self.normal_forward(
                hidden_states,
                cos,
                sin,
                attention_mask,
                layer_past,
                head_mask,
                output_attentions,
            )

    def normal_forward(
        self,
        hidden_states: torch.Tensor,
        cos,
        sin,
        attention_mask: torch.Tensor = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        batch, seq_len, _ = hidden_states.shape

        # # this implementations uses 5 dimension. Need to fix so that it uses 4 dimensions
        # query_layer = self.wq(hidden_states).view(batch, seq_len, -1, self.num_heads // self.num_kv, 64)
        # key_layer = self.wk(hidden_states).view(batch, seq_len, -1, 1, 64)
        # value_layer = self.wv(hidden_states).view(batch, seq_len, -1, 1, 64)
        # key_layer = torch.broadcast_to(key_layer, query_layer.shape)
        # value_layer = torch.broadcast_to(value_layer, query_layer.shape)
        # query_layer, key_layer, value_layer = [
        #     x.reshape(batch, seq_len, -1, self.head_dim)
        #     for x in [query_layer, key_layer, value_layer]
        # ]

        # # Jack's fix 1: Fails -- repeat_interleave seems not to be supported in Pybuda
        # group = self.num_heads // self.num_kv
        # query_layer = self.wq(hidden_states).view(batch, seq_len, -1, 64)   # [batch, seq_len, num_heads(128), head_dim(64)]
        # key_layer = self.wk(hidden_states).view(batch, seq_len, -1, 64).repeat_interleave(group,-2)  # [batch, seq_len, num_kv(8), head_dim(64)] -> [batch, seq_len, num_heads(128)=num_kv*group, head_dim(64)]
        # value_layer = self.wv(hidden_states).view(batch, seq_len, -1, 64).repeat_interleave(group,-2)   # [batch, seq_len, num_kv(8), head_dim(64)]

        # Jack's fix 2: repeat at last dim
        query_layer = self.wq(hidden_states)
        key_layer = self.wk(hidden_states)
        value_layer = self.wv(hidden_states)

        query_layer = query_layer.reshape(batch, seq_len, -1, self.head_dim)
        key_layer = (
            key_layer.view(batch, seq_len, -1, self.head_dim)
            .repeat(1, 1, 1, self.num_heads // self.num_kv)
            .reshape(batch, seq_len, -1, self.head_dim)
        )
        value_layer = (
            value_layer.view(batch, seq_len, -1, self.head_dim)
            .repeat(1, 1, 1, self.num_heads // self.num_kv)
            .reshape(batch, seq_len, -1, self.head_dim)
        )

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(
            1, 2
        )  # reshape(batch_size * self.num_heads, q_length, self.head_dim)
        key_layer = key_layer.transpose(
            1, 2
        )  # .reshape(batch_size * self.num_heads,q_length,self.head_dim,)
        value_layer = value_layer.transpose(
            1, 2
        )  # .reshape(batch_size * self.num_heads, q_length, self.head_dim)

        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        key_layer_ret, value_layer_ret = key_layer, value_layer

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=1)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        attn_output = TT_functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, None, 0.0, is_causal=True
        )

        x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)

        # return output_tensor, present[0], present[1]
        return output_tensor, key_layer_ret, value_layer_ret

    def split_mq_forward(
        self,
        hidden_states: torch.Tensor,
        cos,
        sin,
        attention_mask: torch.Tensor = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        batch, seq_len, _ = hidden_states.shape

        # Jack's fix 2: merge seq_len and batch dim
        query_layer = self.wq(hidden_states).reshape(
            batch, seq_len, -1, self.head_dim
        )  # [batch, seq_len, num_heads(128), head_dim(64)]
        key_layer = self.wk(hidden_states).reshape(
            batch, seq_len, -1, self.head_dim
        )  # [batch, seq_len, num_kv(8), head_dim(64)]
        value_layer = self.wv(hidden_states).reshape(
            batch, seq_len, -1, self.head_dim
        )  # [batch, seq_len, num_kv(8), head_dim(64)]

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(
            1, 2
        )  # [batch, num_heads, seq_len, head_dim]
        key_layer = key_layer.transpose(1, 2)  # [batch, num_kv, seq_len, head_dim]
        value_layer = value_layer.transpose(1, 2)  # [batch, num_kv, seq_len, head_dim]

        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        # # return boardcasted ver. for debug
        # group = self.num_heads // self.num_kv
        # key_layer_ret, value_layer_ret = key_layer.repeat_interleave(group,1), value_layer.repeat_interleave(group,1)

        key_layer_ret, value_layer_ret = key_layer, value_layer

        if layer_past is not None and layer_past[0] is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            # TODO: need to fix here. This is very likely not correct
            key_layer = torch.cat((past_key[:, :, :-1, :], key_layer), dim=-2)
            value_layer = torch.cat((past_value[:, :, :-1, :], value_layer), dim=-2)
            # key_layer = torch.cat((past_key, key_layer), dim=-2)
            # value_layer = torch.cat((past_value, value_layer), dim=-2)

        kv_length = key_layer.size(-2)

        if layer_past is not None and layer_past[0] is not None:
            assert (
                q_length == 1
            ), "Input can only have one token if we're passing in a layer_past"
            is_causal = False
            attention_mask = attention_mask.view(batch_size, 1, q_length, kv_length)
        elif attention_mask is not None:
            is_causal = False
            attention_mask = attention_mask.view(batch_size, 1, q_length, kv_length)
        else:
            is_causal = True
            attention_mask = None

        group = self.num_heads // self.num_kv
        attn_outputs = [
            TT_functional.scaled_dot_product_attention(
                query_layer[:, i * group : (i + 1) * group, :, :],
                key_layer[:, i, :, :].unsqueeze(1),
                value_layer[:, i, :, :].unsqueeze(1),
                attention_mask,
                0.0,
                is_causal=is_causal,
            )
            for i in range(self.num_kv)
        ]

        attn_output = torch.cat(attn_outputs, dim=1)

        x = attn_output.view(batch_size, self.num_heads, q_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.num_heads * self.head_dim)

        output_tensor = self.dense(attn_output)

        # outputs = (output_tensor, present)

        # return output_tensor, present[0], present[1]
        return output_tensor, key_layer_ret, value_layer_ret

    def split_mq_forward_users(
        self,
        hidden_states: torch.Tensor,
        cos,
        sin,
        attention_mask: torch.Tensor = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        num_batch, users, _ = hidden_states.shape

        # Jack's fix 2: merge seq_len and batch dim
        query_layer = self.wq(hidden_states).reshape(
            num_batch, users, -1, self.head_dim
        )  # [batch, users, num_heads(128), head_dim(64)]
        key_layer = self.wk(hidden_states).reshape(
            num_batch, users, -1, self.head_dim
        )  # [batch, users, num_kv(8), head_dim(64)]
        value_layer = self.wv(hidden_states).reshape(
            num_batch, users, -1, self.head_dim
        )  # [batch, users, num_kv(8), head_dim(64)]

        query_layer = query_layer.transpose(1, 2)  # [batch, num_heads, users, head_dim]
        key_layer = key_layer.transpose(1, 2)  # [batch, num_kv, users, head_dim]
        value_layer = value_layer.transpose(1, 2)  # [batch, num_kv, users, head_dim]

        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        # Put users in numkv dimension instead
        # key_layer = key_layer.reshape(num_batch, -1, 1, users*self.head_dim) # [batch, num_kv, 1, users*head_dim]
        # value_layer = value_layer.reshape(num_batch, -1, 1, users*self.head_dim) # [batch, num_kv, 1, users*head_dim]
        key_layer = key_layer.reshape(
            num_batch, users * self.num_kv, 1, self.head_dim
        )  # [batch, num_kv*users, 1, head_dim]
        value_layer = value_layer.reshape(
            num_batch, users * self.num_kv, 1, self.head_dim
        )  # [batch, num_kv*users, 1, head_dim]

        key_layer_ret, value_layer_ret = key_layer, value_layer

        if layer_past is not None and layer_past[0] is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            # past_key : [batch, num_kv, seqlen, users*head_dim]
            # past_value : [batch, num_kv, seqlen, users*head_dim]

            """
            Idea to improve placement of this concat op:
            1. Concat past_key[-32:-1], key_layer 
                - Performs concat with output of size 64
                - Performs sparseMM with output of size 32
            2. Concat past_key[:-32], result

            This should make the sparse matmul much smaller

            Original impl:
            1. concat past_key, key_layer
                - past_key: [batch, num_kv, seqlen(2047), users*head_dim]
                - key_layer: [batch, num_kv, seqlen(1), users*head_dim]
                - output: [batch, num_kv, seqlen(2048), users*head_dim]
                - involves
                    - concat, output size [batch, num_kv, seqlen(2080), users*head_dim]
                    - sparseMM, output size [batch, num_kv, seqlen(2048), users*head_dim]
            """
            # # key_layer = key_layer.repeat(1, 1, 32, 1) # Trick pybuda into believing there are 2080 valid datums
            # key_layer = torch.cat((past_key, key_layer), dim=-2)
            # # value_layer = value_layer.repeat(1, 1, 32, 1)
            # value_layer = torch.cat((past_value, value_layer), dim=-2)

            """
            Method: Use addition to combine past_key and key_layer
            """
            key_layer_top = (
                past_key[:, :, -32:, :] + key_layer
            )  # Should fill top 32 postions with new_k
            key_layer = torch.cat((past_key[:, :, :-32, :], key_layer_top), dim=-2)

            value_layer_top = past_value[:, :, -32:, :] + value_layer
            value_layer = torch.cat(
                (past_value[:, :, :-32, :], value_layer_top), dim=-2
            )

            # key_layer : [batch, num_kv, 1, users*head_dim] in tensor of [batch, num_kv, 32, users*head_dim][:,:,0,:]

            # key_layer_base, key_layer_top_tile = torch.split(past_key, [2048-32, 31], dim=-2)

            # key_layer_top_tile = torch.cat((key_layer_top_tile, key_layer), dim=-2)
            # key_layer = torch.cat((key_layer_base, key_layer_top_tile), dim=-2)

            # value_layer_base, value_layer_top_tile = torch.split(past_value, [2048-32, 31], dim=-2)
            # value_layer_top_tile = torch.cat((value_layer_top_tile, value_layer), dim=-2)
            # value_layer = torch.cat((value_layer_base, value_layer_top_tile), dim=-2)

        kv_length = key_layer.size(-2)

        if layer_past is not None and layer_past[0] is not None:
            is_causal = False
        else:
            is_causal = True
            attention_mask = None

        # key_layer is [batch, num_kv*users, seq, head_dim]
        # key_layer[:,i*users:(i+1)*users,:,:] : batch, users, seq, shead_dim
        group = self.num_heads // self.num_kv
        attn_outputs = [
            TT_functional.scaled_dot_product_attention(
                query_layer[:, i * group : (i + 1) * group, :, :].transpose(
                    1, 2
                ),  # batch, users, group, head_dim TODO: try and remove this double transpose from above
                key_layer[
                    :, i * users : (i + 1) * users, :, :
                ],  # want batch, users, seq, head_dim
                value_layer[:, i * users : (i + 1) * users, :, :],
                attention_mask,
                0.0,
                is_causal=is_causal,
            )
            for i in range(self.num_kv)
        ]

        attn_output = torch.cat(
            attn_outputs, dim=2
        )  # batch, users, num_heads, head_dim

        attn_output = attn_output.reshape(
            num_batch, users, self.num_heads * self.head_dim
        )

        output_tensor = self.dense(attn_output)

        return output_tensor, key_layer_ret, value_layer_ret

    def efficient_forward_split_group_qkv_proj(
        self,
        hidden_states: torch.Tensor,
        cos,
        sin,
        attention_mask: torch.Tensor = None,
        layer_past=None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        kv_read_mask=None,
        kv_write_mask=None,
    ):
        num_batch, users, _ = hidden_states.shape
        group = self.num_heads // self.num_kv

        # maintain a list of queries, keys, values
        queries = []
        keys = []
        values = []
        for i in range(self.num_kv):
            query = (
                self.wq_list[i](hidden_states)
                .reshape(num_batch, users, -1, self.head_dim)
                .transpose(1, 2)
            )  # [batch, group, users, head_dim]
            query = apply_rotary_pos_emb_q(query, cos, sin)
            queries.append(query.transpose(1, 2))

            key = self.wk_list[i](hidden_states).reshape(
                num_batch, 1, users, self.head_dim
            )  # [batch, 1, users, head_dim]
            key = apply_rotary_pos_emb_k(key, cos, sin)
            keys.append(key.transpose(1, 2))

            value = self.wv_list[i](hidden_states).reshape(
                num_batch, users, 1, self.head_dim
            )  # [batch, 1, users, head_dim]
            values.append(value)

        keys_cache = layer_past[:8]
        values_cache = layer_past[8:]

        # key_layer_ret, value_layer_ret = key_layer, value_layer

        if layer_past is not None and layer_past[0] is not None:
            keys_cache_masked = [k * kv_read_mask for k in keys_cache]
            values_cache_masked = [v * kv_read_mask for v in values_cache]

            # past_key = past_key * kv_read_mask          # zero out k/v for index where we will write current token
            # past_value = past_value * kv_read_mask

            keys_masked = [k * kv_write_mask for k in keys]
            values_masked = [v * kv_write_mask for v in values]

            # key_layer = key_layer * kv_write_mask       # broadcast current k/v to k/v cache size and mask out locations where we won't be writing
            # value_layer = value_layer * kv_write_mask

            keys_merged = [
                keys_masked[i] + keys_cache_masked[i]
                for i in range(len(keys_cache_masked))
            ]
            values_merged = [
                values_masked[i] + values_cache_masked[i]
                for i in range(len(values_cache_masked))
            ]

        if layer_past is not None and layer_past[0] is not None:
            is_causal = False
        else:
            is_causal = True
            attention_mask = None

        # key_layer is [batch, num_kv*users, seq, head_dim]
        # key_layer[:,i*users:(i+1)*users,:,:] : batch, users, seq, shead_dim
        group = self.num_heads // self.num_kv
        attn_outputs = [
            TT_functional.scaled_dot_product_attention(
                queries[i],
                keys_merged[i],
                values_merged[i],
                attention_mask,
                0.0,
                is_causal=is_causal,
            )
            for i in range(self.num_kv)
        ]

        attn_output = torch.cat(
            attn_outputs, dim=2
        )  # batch, users, num_heads, head_dim

        attn_output = attn_output.reshape(
            num_batch, users, self.num_heads * self.head_dim
        )

        output_tensor = self.dense(attn_output)

        return output_tensor, keys_merged, values_merged

    def efficient_forward_flash_decode(
        self,
        hidden_states: torch.Tensor,
        cos,
        sin,
        attention_mask: torch.Tensor = None,
        layer_past=None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        kv_read_mask=None,
        kv_write_mask=None,
    ):
        num_batch, users, _ = hidden_states.shape
        group = self.num_heads // self.num_kv

        # maintain a list of queries, keys, values
        queries = []
        keys = []
        values = []
        for i in range(self.num_kv):
            query = (
                self.wq_list[i](hidden_states)
                .reshape(num_batch, users, -1, self.head_dim)
                .transpose(1, 2)
            )  # [batch, group, users, head_dim]
            query = apply_rotary_pos_emb_q(query, cos, sin)
            queries.append(query.transpose(1, 2))  # [batch, users, group, head_dim]

            key = self.wk_list[i](hidden_states).reshape(
                num_batch, 1, users, self.head_dim
            )  # [batch, 1, users, head_dim]
            key = apply_rotary_pos_emb_k(key, cos, sin)
            keys.append(key.transpose(1, 2))  # [batch, users, 1, head_dim]

            value = self.wv_list[i](hidden_states).reshape(
                num_batch, users, 1, self.head_dim
            )  # [batch, users, 1, head_dim]
            values.append(value)

        keys_cache = layer_past[:8]
        values_cache = layer_past[8:]

        # key_layer_ret, value_layer_ret = key_layer, value_layer

        if layer_past is not None and layer_past[0] is not None:
            keys_cache_masked = [k * kv_read_mask for k in keys_cache]
            values_cache_masked = [v * kv_read_mask for v in values_cache]

            # past_key = past_key * kv_read_mask          # zero out k/v for index where we will write current token
            # past_value = past_value * kv_read_mask

            keys_masked = [k * kv_write_mask for k in keys]
            values_masked = [v * kv_write_mask for v in values]

            # key_layer = key_layer * kv_write_mask       # broadcast current k/v to k/v cache size and mask out locations where we won't be writing
            # value_layer = value_layer * kv_write_mask

            keys_merged = [
                keys_masked[i] + keys_cache_masked[i]
                for i in range(len(keys_cache_masked))
            ]
            values_merged = [
                values_masked[i] + values_cache_masked[i]
                for i in range(len(values_cache_masked))
            ]

        if layer_past is not None and layer_past[0] is not None:
            is_causal = False
        else:
            is_causal = True
            attention_mask = None
        # key_layer is [batch, num_kv*users, seq, head_dim]
        # key_layer[:,i*users:(i+1)*users,:,:] : batch, users, seq, shead_dim
        group = self.num_heads // self.num_kv
        attn_outputs = [
            TT_functional.flash_decode_attention(
                Q=queries[i],
                Ks=[keys_cache[i], keys[i]],
                Vs=[values_cache[i], values[i]],
                attn_masks=[attention_mask, None],
            )
            for i in range(self.num_kv)
        ]

        # attn_output = torch.cat(attn_outputs, dim=2) # batch, users, num_heads, head_dim

        # attn_output = attn_output.reshape(num_batch, users, self.num_heads * self.head_dim)

        attn_outputs = [
            o.reshape(num_batch, users, -1) for o in attn_outputs
        ]  # Reshape at end of each MQA
        # attn_output = torch.cat(attn_outputs, dim=-1) # batch, 1, users, num_heads*head_dim

        # output_tensor = self.dense(attn_output)
        dense_outputs = [
            self.dense_list[i](attn_outputs[i]) for i in range(self.num_kv)
        ]
        # Tree reduction
        tree_outputs = [
            dense_outputs[0] + dense_outputs[1],
            dense_outputs[2] + dense_outputs[3],
            dense_outputs[4] + dense_outputs[5],
            dense_outputs[6] + dense_outputs[7],
        ]
        tree_outputs = [
            tree_outputs[0] + tree_outputs[1],
            tree_outputs[2] + tree_outputs[3],
        ]
        output_tensor = tree_outputs[0] + tree_outputs[1]

        return output_tensor, keys_merged, values_merged


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
        self.use_cache = config.use_cache

        self.mlp = MLP(config)

        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        self.hidden_dropout = config.hidden_dropout

        self.config = config
        self.efficient_40b = config.efficient_40b

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos,
        sin,
        attention_mask=None,
        kv_read_mask=None,
        kv_write_mask=None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # breakpoint()
        ln_attn = self.ln_attn(hidden_states)
        ln_mlp = self.ln_mlp(hidden_states)

        residual = hidden_states

        # Self attention.
        attention_output, key_past, value_past = self.self_attention(
            ln_attn,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            cos=cos,
            sin=sin,
            kv_read_mask=kv_read_mask,
            kv_write_mask=kv_write_mask,
        )

        # MLP.
        mlp_output = self.mlp(ln_mlp)

        mlp_output += attention_output

        output = dropout_add(
            mlp_output, residual, self.config.hidden_dropout, training=self.training
        )

        return output, key_past, value_past  # hidden_states, present, attentions


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
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
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


class SequentialCaller(nn.Module):
    def __init__(self, layers, norm=None, lm_head=None, kv_mask_embed=None):
        super().__init__()
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.num_heads = layers[0].self_attention.num_heads
        self.hidden_size = layers[0].self_attention.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.use_cache = layers[0].use_cache
        self.efficient_40b = layers[0].efficient_40b
        self.kv_mask_embed = kv_mask_embed

    def forward(self, hidden_states, cos, sin, attention_mask=None, *past_key_values):
        result = []
        kv_read_mask = None
        kv_write_mask = None
        # breakpoint()
        if len(past_key_values) > 0 and self.efficient_40b:
            kv_mask_id = past_key_values[-1]
            past_key_values = past_key_values[:-1]
            kv_write_mask = (
                self.kv_mask_embed(kv_mask_id)
                .reshape(1, 32, 64, 2048)
                .transpose(-1, -2)
            )
            # kv_write_mask = kv_write_mask.expand(1, 32, 2048, self.head_dim)
            kv_read_mask = 1 - kv_write_mask
            kv_write_mask = 1 - kv_read_mask  # add this dummy line as buffer op
            # kv_read_mask = kv_read_mask.transpose(-1, -2)
            # kv_write_mask = kv_write_mask.transpose(-1, -2)

        for i, block in enumerate(self.layers):
            if len(past_key_values) > 0 and self.efficient_40b:
                layer_past = past_key_values[i * 16 : (i + 1) * 16]
            elif len(past_key_values) > 0:
                layer_past = past_key_values[i * 2], past_key_values[i * 2 + 1]
            else:
                layer_past = None

            output, key_past, value_past = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                kv_read_mask=kv_read_mask,
                kv_write_mask=kv_write_mask,
                cos=cos,
                sin=sin,
            )

            # TODO: return only new_key, new_value
            hidden_states = output
            if self.efficient_40b:
                result.extend(key_past + value_past)
            else:
                result.extend([key_past, value_past])
        if self.norm is not None and self.lm_head is not None:
            hidden_states = self.norm(hidden_states)
            hidden_states = self.lm_head(hidden_states)
        if type(hidden_states) == tuple or type(hidden_states) == list:
            for fracture in hidden_states[::-1]:
                result.insert(0, fracture)
        else:
            result.insert(0, hidden_states)
        return tuple(result)


class RWModel(RWPreTrainedModel):
    def __init__(self, config: RWConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.num_kv = config.n_head_kv
        self.alibi = config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        self.kv_mask_embed = nn.Embedding(
            2048, 2048
        )  # this embedding should be an identity matrix
        self.kv_mask_embed.weight.data = torch.eye(2048)

        # Transformer blocks
        self.h = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.blocks = None

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        self.use_cache = config.use_cache
        self.rotary_emb = RotaryEmbeddingTT(config.head_dim, 2048, 10000, None)
        self.batch_users = config.user_rows > 1
        self.efficient_40b = config.efficient_40b

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head

    def split_qkv_weights(self):
        for h in self.h:
            h.self_attention.split_qkv_weights()
        self.blocks = SequentialCaller(self.h)

    def split_qkv_wqkv_weights(self):
        for h in self.h:
            h.self_attention.split_qkv_weights()
            h.self_attention.split_group_qkv_proj()
        self.blocks = SequentialCaller(self.h, kv_mask_embed=self.kv_mask_embed)

    def put_lm_head_on_device(self, lm_head):
        self.blocks = SequentialCaller(
            self.h, self.ln_f, lm_head, kv_mask_embed=self.kv_mask_embed
        )

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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_mask_id=None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids=None,
        profile_runtime=False,
        on_device_lm_head=False,
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

        runtime_profile = {}
        pre_start = time()

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

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
                ).expand(num_batch, users, self.num_heads // self.num_kv, kv_length)

                # Do the same for kv read/write mask. Expand last dim to d_head
                # expand the last dim, but at the second last dimension because embedding requires id in the last dim I believe. This might not be the optimal way
                kv_mask_id = kv_mask_id.view(num_batch, users, 1).expand(
                    num_batch, users, self.hidden_size // self.num_heads
                )
                kv_mask_id = kv_mask_id.reshape(num_batch, -1)

        # Rotary Embeddings
        cos, sin = self.rotary_emb()
        cos, sin = gather_cos_sin(cos, sin, position_ids)

        pre_end = time()

        if self.efficient_40b:
            assert past_key_values is not None and attention_mask is not None
            flattened_kv = []
            for kvs in past_key_values:
                flattened_kv.extend(kvs)
            outputs = self.blocks(
                hidden_states, cos, sin, attention_mask, *flattened_kv, kv_mask_id
            )
        elif past_key_values is not None and attention_mask is not None:
            flattened_kv = []
            for k, v in past_key_values:
                flattened_kv.extend([k, v])
            outputs = self.blocks(
                hidden_states, cos, sin, attention_mask, *flattened_kv
            )
        elif past_key_values is None and attention_mask is None:
            outputs = self.blocks(hidden_states, cos, sin)
        elif past_key_values is None and attention_mask is not None:
            outputs = self.blocks(hidden_states, cos, sin, attention_mask)
        else:
            raise ValueError("XNOR past_key_values and attention_mask")

        assert (
            return_dict == self.config.use_return_dict
        ), f"Expect the default value of return_dict: {self.config.use_return_dict} but instead got: {return_dict}"

        blocks_end = time()
        if on_device_lm_head:
            if isinstance(outputs, tuple):
                outputs = (outputs[0],) + outputs[1:]
                # outputs = [o.to(torch.float32) for o in outputs] # for bf16 transformer compatibility
            else:
                outputs = (outputs,)
        else:
            if isinstance(outputs, tuple):
                outputs = (self.ln_f(outputs[0]),) + outputs[1:]
                # outputs = [o.to(torch.float32) for o in outputs] # for bf16 transformer compatibility
            else:
                outputs = (self.ln_f(outputs),)

        post_end = time()

        if profile_runtime:
            runtime_profile["pre"] = pre_end - pre_start
            runtime_profile["blocks"] = blocks_end - pre_end
            runtime_profile["post"] = post_end - blocks_end
            if isinstance(outputs, tuple):
                outputs = outputs + (runtime_profile,)
            else:
                outputs = (outputs, runtime_profile)

        return outputs  # , all_hidden_states, all_self_attentions, return_dict


class LMHeadFractured(nn.Module):
    def __init__(self, lm_head, fracture_factor):
        super().__init__()
        vocab_size, embed_dim = lm_head.weight.data.shape
        fractured_vocab_size = vocab_size // fracture_factor
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(embed_dim, fractured_vocab_size, bias=False)
                for _ in range(fracture_factor)
            ]
        )
        # load the weights
        for i in range(fracture_factor):
            self.lm_heads[i].weight.data = lm_head.weight.data[
                i * fractured_vocab_size : (i + 1) * fractured_vocab_size, :
            ]

    def forward(self, x):
        return [lm_head(x) for lm_head in self.lm_heads]


class RWForCausalLM(RWPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h.*.self_attention.scale_mask_softmax.causal_mask",
        r"lm_head.weight",
    ]

    def __init__(self, config: RWConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        self.transformer = RWModel(config)
        self.is_padded_lmh = config.padded_lmh
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def set_padded_lmh(self, padding):
        padded_lmh = nn.Linear(self.embed_dim, self.vocab_size + padding, bias=False)
        # added weights here
        padded_lmh.weight.data[:-padding, :] = self.lm_head.weight.data
        self.lm_head = padded_lmh

    def fracture_vocab(self, fracture_factor):
        assert (
            self.vocab_size % fracture_factor == 0
        ), "vocab size must be divisible by fracture factor. try padding to vocab size"
        self.lm_head = LMHeadFractured(self.lm_head, fracture_factor)

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

    def forward(self, *args, **kwargs):
        output = self.main_forward_part(*args, **kwargs)
        return self.final_forward_part(output)

    def main_forward_part(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_mask_id=None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids=None,
        profile_runtime=False,
        on_device_lm_head=False,
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
            kv_mask_id=kv_mask_id,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            position_ids=position_ids,
            profile_runtime=profile_runtime,
            on_device_lm_head=on_device_lm_head,
        )
        return transformer_outputs

    def final_forward_part(
        self,
        outputs,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = self.config.use_cache

        presents = () if use_cache else None

        all_hidden_states = () if output_hidden_states else None

        # We don't use these in decode.py so we also don't do the work to pass them around in async mode
        all_self_attentions = None
        # Use default value from config
        return_dict = self.config.use_return_dict

        hidden_states = outputs[0]

        presents = outputs[1:]

        transformer_outputs = None
        if not return_dict:
            transformer_outputs = tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        else:
            transformer_outputs = BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )

        # hidden_states = outputs[0]
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
