import json
import math
import os
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartEncoderLayer

from utils.phone import unstressed_phone_map, stressed_phone_map
from .config import LatticeBartConfig


class LatticeSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
            f"embed_dim must be divisible by num_heads (`embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _split_head(self, tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, marginal_hidden, forward_backward, lattice_mask, output_attentions=False):
        """
        :param hidden_states:       FloatTensor. [B, S, D]
        :param marginal_hidden:     FloatTensor. [B, S, D]
        :param forward_backward:    FloatTensor. [B, 1, S, S]
        :param lattice_mask:        FloatTensor. [B, 1, S, S]
        :param output_attentions: bool
        """
        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        Q = self._split_head(self.q_proj(hidden_states) * self.scaling, tgt_len, bsz)
        # get key, value proj
        K = self._split_head(self.k_proj(hidden_states), -1, bsz)
        V = self._split_head(self.v_proj(hidden_states) + marginal_hidden, -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        Q = Q.view(*proj_shape)
        K = K.view(*proj_shape)
        V = V.view(*proj_shape)

        src_len = K.size(1)
        attn_weights = torch.bmm(Q, K.transpose(1, 2))

        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len), \
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
        assert lattice_mask.shape == forward_backward.shape == (bsz, 1, tgt_len, src_len), \
            f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {lattice_mask.size()}"

        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + forward_backward + lattice_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, V)

        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim), \
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim) \
            .transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class LatticeEncoderLayer(nn.Module):
    def __init__(self, config: LatticeBartConfig):
        super().__init__()

        self.embed_dim = config.d_model
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout

        self.lattice_self_attn = LatticeSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states, marginal_hidden, forward_backward, lattice_mask, output_attentions=False):
        """
        :param hidden_states:       FloatTensor. [B, S, D]
        :param marginal_hidden:     FloatTensor. [B, S, D]
        :param forward_backward:    FloatTensor. [B, 1, S, S]
        :param lattice_mask:        FloatTensor. [B, 1, S, S]
        :param output_attentions:   bool
        """
        residual = hidden_states
        hidden_states, attn_weights = self.lattice_self_attn.forward(
            hidden_states=hidden_states,
            lattice_mask=lattice_mask,
            marginal_hidden=marginal_hidden,
            forward_backward=forward_backward,
            output_attentions=output_attentions
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training) + residual
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class LatticeEncoder(nn.Module):
    MODEL_NAME = "LatticeEncoder"
    MODEL_PT = "lattice_encoder.pt"

    def __init__(self, config: LatticeBartConfig, shared_embed):
        super().__init__()

        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if shared_embed is not None:
            self.embed_tokens = shared_embed
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = nn.Embedding(config.max_position_embeddings + 2, embed_dim, self.padding_idx)

        # Phoneme embedding
        if config.phone_embed == "none":
            self.phone_embed = None
        elif config.phone_embed == "unstressed":
            self.phone_embed = nn.Embedding(len(unstressed_phone_map), embed_dim)
        else:
            self.phone_embed = nn.Embedding(len(stressed_phone_map), embed_dim)

        self.marginal_proj = nn.Linear(1, embed_dim)
        self.lattice_layers = nn.ModuleList([LatticeEncoderLayer(config) for _ in range(config.lat_enc_layers)])

        self.layers = nn.ModuleList(
            [BartEncoderLayer(config) for _ in range(config.encoder_layers - config.lat_enc_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def prepare_embeds(self, embed: nn.Embedding, input_ids: LongTensor, appendix: dict = None,
                       batch_idx: LongTensor = None):
        inputs_embeds = embed(input_ids)

        # When distributed training, `batch_idx` will not be None
        min_idx, max_idx = 0, input_ids.shape[0] - 1
        if batch_idx is not None:
            min_idx, max_idx = batch_idx[0].item(), batch_idx[-1].item()

        # Add appendix
        if appendix is not None:
            for i, chunks in appendix.items():
                if i < min_idx or max_idx < i:
                    continue
                i -= min_idx
                for j, ids in chunks.items():
                    _ids = torch.tensor(ids, dtype=torch.long, device=input_ids.device)
                    _chunk_embed = embed(_ids).sum(dim=0)
                    inputs_embeds[i, j] = (inputs_embeds[i, j] + _chunk_embed) / (len(ids) + 1)
        inputs_embeds = inputs_embeds * self.embed_scale
        return inputs_embeds

    def _pos_embedding(self, pos: Tensor):
        pos = torch.clamp(pos + 2, 0, self.config.max_position_embeddings - 1)
        return self.embed_positions(pos)

    def forward(self, input_ids, appendix=None, phone_ids=None, phone_appendix=None, min_depth=None, max_depth=None,
                lattice_mask=None, forward_backward=None, marginals=None, output_attentions=False,
                output_hidden_states=False, return_dict=True, batch_idx: Tensor = None):
        """
        :param input_ids:       LongTensor. [B, S]
        :param appendix:        dict
        :param phone_ids:       LongTensor. [B, S]
        :param phone_appendix:  dict
        :param min_depth:       LongTensor. [B, S]
        :param max_depth:       LongTensor. [B, S]
        :param lattice_mask:        LongTensor or FloatTensor. [B, S, S]. {0, 1}
        :param forward_backward:    FloatTensor. [B, S, S]
        :param marginals:       FloatTensor. [B, S]
        :param output_attentions:   bool
        :param output_hidden_states: bool
        :param return_dict:         bool
        :param batch_idx:       LongTensor. Used when train model in distributed fashion
        """
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Word embedding
        inputs_embeds = self.prepare_embeds(embed=self.embed_tokens, input_ids=input_ids,
                                            appendix=appendix, batch_idx=batch_idx)
        if phone_ids is not None:
            phone_embeds = self.prepare_embeds(embed=self.phone_embed, input_ids=phone_ids,
                                               appendix=phone_appendix, batch_idx=batch_idx)
            inputs_embeds = inputs_embeds + phone_embeds

        # Positional embedding
        if min_depth is None and max_depth is None:
            pos_embed = torch.zeros_like(inputs_embeds)
        elif min_depth is None:
            pos_embed = self._pos_embedding(max_depth)
        elif max_depth is None:
            pos_embed = self._pos_embedding(min_depth)
        else:
            pos_embed = (self._pos_embedding(min_depth) + self._pos_embedding(max_depth)) / 2

        # Prepare hidden states
        hidden_states = inputs_embeds + pos_embed
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        B, S = input_ids.shape

        # expand lattice mask
        if lattice_mask is None:
            lattice_mask = torch.ones((B, 1, S, S), device=input_ids.device, dtype=torch.float)
        elif isinstance(lattice_mask, torch.FloatTensor):
            # probability mask
            lattice_mask = lattice_mask.unsqueeze(1)
        else:
            # binary mask
            lattice_mask = 1.0 - lattice_mask.unsqueeze(1)
            lattice_mask = lattice_mask.masked_fill(lattice_mask.bool(), torch.finfo(inputs_embeds.dtype).min)

        # forward_backward matrix
        if forward_backward is None:
            forward_backward = torch.zeros((B, 1, S, S), device=input_ids.device, dtype=torch.float)
        else:
            forward_backward = forward_backward.unsqueeze(1)

        # Marginal vector
        if marginals is None:
            marginal_hidden = torch.zeros_like(hidden_states)
        else:
            marginal_hidden = self.marginal_proj(marginals.unsqueeze(-1))

        # Lattice Encoders
        for idx, lat_enc_layer in enumerate(self.lattice_layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = lat_enc_layer.forward(
                hidden_states=hidden_states,
                lattice_mask=lattice_mask,
                marginal_hidden=marginal_hidden,
                forward_backward=forward_backward,
                output_attentions=output_attentions
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Bart Encoders
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer.forward(hidden_states, lattice_mask,
                                                  layer_head_mask=None, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=None, attentions=all_attentions
        )


class EncoderWrapper(nn.Module):
    ENC_PT_FILE = LatticeEncoder.MODEL_PT
    PROJ_PT_FILE = "proj.pt"

    def __init__(self, lattice_encoder: LatticeEncoder):
        super().__init__()

        self.encoder = lattice_encoder
        self.proj = nn.Linear(self.config.d_model, self.config.d_model)

    @property
    def config(self):
        return self.encoder.config

    def forward(self, *args, eos_id=2, pooling_method="mean", token_id_weight: dict = None, **kwargs):
        assert pooling_method in ("mean", "weight", "eos"), \
            f"pooling_method({pooling_method}) should be 'mean' or 'weight' or 'eos'."
        assert pooling_method != "weight" or (pooling_method == "weight" and token_id_weight is not None), \
            f"When pooling_method is 'weight', token_id_weight should be given."

        if "output_hidden_states" not in kwargs or not kwargs["output_hidden_states"]:
            kwargs["output_hidden_states"] = True
        outputs = self.encoder.forward(*args, **kwargs)
        eos_position = torch.where(kwargs["input_ids"].eq(eos_id))

        if pooling_method == "eos":
            last_state = outputs.last_hidden_state[eos_position]
        else:
            hidden_states = outputs.last_hidden_state
            B, S, _D = hidden_states.shape
            wt = torch.ones(B, S, dtype=hidden_states.dtype, device=hidden_states.device)
            input_ids = args[0] if len(args) > 0 else kwargs["input_ids"]

            for b, eos_pos in zip(*eos_position):
                wt[b, eos_pos + 1:] = 0
                # Assign token weight to each position
                if pooling_method == "weight":
                    for i in range(eos_pos + 1):
                        wt[b, i] = token_id_weight[input_ids[b, i].item()]

            last_state = (hidden_states * wt.unsqueeze(dim=-1)).sum(dim=-2)
            last_state = last_state / wt.sum(dim=-1).unsqueeze(dim=-1)

        last_state = self.proj(last_state)
        return last_state

    def save_pretrained(self, model_path: str):
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), join(model_path, self.ENC_PT_FILE))
        torch.save(self.proj.state_dict(), join(model_path, self.PROJ_PT_FILE))
        with open(join(model_path, "config.json"), "w", encoding="utf8") as f:
            config = self.config.to_dict()
            json.dump(config, f, indent=2, ensure_ascii=False)
        # log(f"Save {self.__class__.__name__} to {model_path}", logger)

    @classmethod
    def from_pretrained(cls, model_path: str):
        with open(join(model_path, "config.json"), encoding="utf8") as f:
            config = LatticeBartConfig.from_dict(json.load(f))
        encoder = LatticeEncoder(config, None)
        encoder.load_state_dict(torch.load(join(model_path, cls.ENC_PT_FILE), map_location=lambda st, loc: st))

        wrapper = cls(encoder)
        wrapper.proj.load_state_dict(torch.load(join(model_path, cls.PROJ_PT_FILE), map_location=lambda st, loc: st))

        # log(f"Load {cls.__class__.__name__} from {model_path}", logger)
        return wrapper
