import math
import random
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BartPretrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.bart.modeling_bart import BartDecoderLayer

from .config import LatticeBartConfig


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    Return mask of shape: [bsz, 1, tgt_len, past_key_value_len + tgt_len]
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.shape[-1])
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if (tgt_len is not None) else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class LatticeDecoder(BartPretrainedModel):
    config: LatticeBartConfig

    def _reorder_cache(self, past, beam_idx):
        raise NotImplementedError

    def __init__(self, config: LatticeBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = nn.Embedding(config.max_position_embeddings + 2, config.d_model, self.padding_idx)

        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def _pos_embedding(self, pos):
        # pos embedding with offset = 2
        pos = torch.clamp(pos + 2, 0, self.config.max_position_embeddings - 1)
        return self.embed_positions(pos)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, dtype, past_key_values_length):
        """
        Args:
            attention_mask: [bsz, tgt_len]
            input_shape:    (bsz, src_len)
            dtype:          dtype
            past_key_values_length: int
        Returns:
            mask: [bsz, 1, tgt_len, src_len]
        """
        combined_attention_mask = None
        if input_shape[-1] > 1:
            # [bsz, 1, tgt_len, src_len]
            combined_attention_mask = _make_causal_mask(input_shape, dtype, past_key_values_length).to(self.device)

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                # [bsz, tgt_len] -> [bsz, 1, tgt_len, src_len]
                expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1])
            else:
                assert len(attention_mask.shape) == 3, \
                    f"Require attention mask for lattice of shape [bsz, tgt_seq_len, src_seq_len], " \
                    f"got {attention_mask.shape}"
                # [bsz, 1, tgt_len, src_len]
                expanded_attn_mask = 1.0 - attention_mask.unsqueeze(1)
                expanded_attn_mask = expanded_attn_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

            if combined_attention_mask is None:
                combined_attention_mask = expanded_attn_mask
            else:
                combined_attention_mask += expanded_attn_mask

        return combined_attention_mask

    def forward(self, input_ids=None, attention_mask=None, min_depth=None, max_depth=None, encoder_hidden_states=None,
                encoder_attention_mask=None, head_mask=None, encoder_head_mask=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]，此处的attention_mask为self attention的mask
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds.dtype,
                                                              past_key_values_length)
        # print(attention_mask)

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # Positional embedding
        if min_depth is None and max_depth is None:
            bsz, seq_len = input_ids.shape[:2]
            pos = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long,
                               device=input_ids.device)
            positions = self._pos_embedding(pos)
        elif min_depth is None:
            positions = self._pos_embedding(max_depth)
        elif max_depth is None:
            positions = self._pos_embedding(min_depth)
        else:
            positions = (self._pos_embedding(min_depth) + self._pos_embedding(max_depth)) / 2

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layers), \
                f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    print("`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                          "`use_cache=False`...", file=sys.stderr)
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                from torch.utils.checkpoint import checkpoint

                layer_outputs = checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
