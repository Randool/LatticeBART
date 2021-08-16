import json
import logging
import math
import os
from os.path import join
from typing import Optional, Iterable, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch.nn import CrossEntropyLoss
from transformers import BartConfig, BartForConditionalGeneration, BeamSearchScorer
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput, Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import BartEncoderLayer, BartDecoder, BartPretrainedModel


class LatticeSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
            f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
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
    def __init__(self, config: BartConfig):
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

    def __init__(self, config: BartConfig, shared_embed):
        super().__init__()

        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.depth_clip = config.depth_clip

        if shared_embed is not None:
            self.embed_tokens = shared_embed
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.positional_embed = nn.Embedding(self.depth_clip, embed_dim)
        self.marginal_proj = nn.Linear(1, embed_dim)
        self.lattice_layer = LatticeEncoderLayer(config)

        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def _prepare_word_embeds(self, input_ids: LongTensor, appendix: dict = None, batch_idx: LongTensor = None):
        inputs_embeds = self.embed_tokens(input_ids)

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
                    _chunk_embed = self.embed_tokens(_ids).sum(dim=0)
                    inputs_embeds[i, j] = (inputs_embeds[i, j] + _chunk_embed) / (len(ids) + 1)
        inputs_embeds = inputs_embeds * self.embed_scale
        return inputs_embeds

    def forward(self, input_ids, appendix=None, min_depth=None, max_depth=None, lattice_mask=None,
                forward_backward=None, marginals=None, output_attentions=False, output_hidden_states=False,
                return_dict=True, batch_idx: Tensor = None):
        """
        :param input_ids:       LongTensor. [B, S]
        :param appendix:        dict
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
        inputs_embeds = self._prepare_word_embeds(input_ids, appendix, batch_idx=batch_idx)

        # Positional embedding
        if min_depth is None and max_depth is None:
            pos_embed = torch.zeros_like(inputs_embeds)
        elif min_depth is None:
            pos_embed = self.positional_embed(torch.clamp(max_depth, 0, self.depth_clip - 1))
        elif max_depth is None:
            pos_embed = self.positional_embed(torch.clamp(min_depth, 0, self.depth_clip - 1))
        else:
            pos_embed = (self.positional_embed(torch.clamp(min_depth, 0, self.depth_clip - 1)) +
                         self.positional_embed(torch.clamp(max_depth, 0, self.depth_clip - 1))) / 2

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

        # Encode
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        layer_outputs = self.lattice_layer.forward(
            hidden_states=hidden_states,
            lattice_mask=lattice_mask,
            marginal_hidden=marginal_hidden,
            forward_backward=forward_backward,
            output_attentions=output_attentions
        )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

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

    def forward(self, *args, eos_id=2, **kwargs):
        if "output_hidden_states" not in kwargs or not kwargs["output_hidden_states"]:
            kwargs["output_hidden_states"] = True
        outputs = self.encoder.forward(*args, **kwargs)
        eos_position = torch.where(kwargs["input_ids"].eq(eos_id))
        last_state = outputs.last_hidden_state[eos_position]
        last_state = self.proj(last_state)
        return last_state

    def save_pretrained(self, model_path: str):
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.encoder.state_dict(), join(model_path, self.ENC_PT_FILE))
        torch.save(self.proj.state_dict(), join(model_path, self.PROJ_PT_FILE))
        with open(join(model_path, "config.json"), "w", encoding="utf8") as f:
            config = self.config.to_dict()
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info(f"Save {self.__class__.__name__} to {model_path}")

    @classmethod
    def from_pretrained(cls, model_path: str):
        with open(join(model_path, "config.json"), encoding="utf8") as f:
            config = BartConfig.from_dict(json.load(f))
        encoder = LatticeEncoder(config, None)
        encoder.load_state_dict(torch.load(join(model_path, cls.ENC_PT_FILE), map_location=lambda st, loc: st))

        wrapper = cls(encoder)
        wrapper.proj.load_state_dict(torch.load(join(model_path, cls.PROJ_PT_FILE), map_location=lambda st, loc: st))

        logging.info(f"Load {cls.__class__.__name__} from {model_path}")
        return wrapper


class LatticeBartBase(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.encoder = LatticeEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    # LatticeBartBase
    def forward(self, input_ids=None, appendix=None, min_depth=None, max_depth=None, lattice_mask=None,
                forward_backward=None, marginals=None, enc_dec_attention_mask=None, encoder_outputs=None,
                past_key_values=None, decoder_input_ids=None, decoder_attention_mask=None, use_cache=True,
                output_attentions=False, output_hidden_states=False, return_dict=True, batch_idx: Tensor = None):
        # Encoder forward
        if encoder_outputs is None:
            encoder_outputs = self.encoder.forward(
                input_ids=input_ids,
                appendix=appendix,
                min_depth=min_depth,
                max_depth=max_depth,
                lattice_mask=lattice_mask,
                forward_backward=forward_backward,
                marginals=marginals,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                batch_idx=batch_idx
            )

            if enc_dec_attention_mask is None:
                enc_dec_attention_mask = input_ids.ne(self.config.pad_token_id).long()

        # Decoder forward
        decoder_outputs = self.decoder.forward(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=enc_dec_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions
        )


class LatticeBart(BartPretrainedModel):
    MODEL_PT = "lattice_bart.pt"

    def __init__(self, config: BartConfig):
        super().__init__(config)

        self.model = LatticeBartBase(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # LatticeBart
    def forward(self, input_ids=None, appendix=None, min_depth=None, max_depth=None, lattice_mask=None,
                forward_backward=None, marginals=None, enc_dec_attention_mask=None, encoder_outputs=None,
                past_key_values=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None,
                use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True,
                batch_idx: Tensor = None, reduction="mean"):
        outputs = self.model.forward(
            input_ids=input_ids,
            appendix=appendix,
            min_depth=min_depth,
            max_depth=max_depth,
            lattice_mask=lattice_mask,
            forward_backward=forward_backward,
            marginals=marginals,
            enc_dec_attention_mask=enc_dec_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            batch_idx=batch_idx
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction=reduction)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
        )

    def save_pretrained(self, model_path: str):
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.state_dict(), join(model_path, self.MODEL_PT))
        with open(join(model_path, "config.json"), "w", encoding="utf8") as f:
            config = self.config.to_dict()
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info(f"Save {self.__class__.__name__} to {model_path}")

    @classmethod
    def from_pretrained(cls, model_path: str, *model_args, **kwargs):
        with open(join(model_path, "config.json"), encoding="utf8") as f:
            config = BartConfig.from_dict(json.load(f))
        lattice_bart = cls(config)
        lattice_bart.load_state_dict(torch.load(join(model_path, cls.MODEL_PT), map_location=lambda st, loc: st))
        logging.info(f"Load {cls.__class__.__name__} from {model_path}")
        return lattice_bart

    @classmethod
    def from_bart(cls, bart_path: str, depth_clip=100):
        bart = BartForConditionalGeneration.from_pretrained(bart_path)
        config = bart.config.to_dict()
        config["depth_clip"] = depth_clip
        lattice_bart = cls(BartConfig.from_dict(config))

        # Move parameters from pretrained bart
        bart_param = dict(bart.named_parameters())

        for key, param in lattice_bart.named_parameters():
            if key not in bart_param:
                logging.debug(f"[x] {key}\t no found")
                continue
            if param.shape != bart_param[key].shape:
                logging.debug(f"[x] {key}\t shape mismatch between bart and lattice bart")
                continue
            param.data = bart_param[key].data
            logging.debug(f"[v] {key}")

        return lattice_bart

    @torch.no_grad()
    def generate(
            self, input_ids, appendix, min_depth, max_depth, lattice_mask,
            forward_backward=None, marginals=None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            num_beam_groups: Optional[int] = None,
            diversity_penalty: Optional[float] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            batch_idx: Tensor = None,
            **model_kwargs,
    ):
        # set init values
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict_in_generate = return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate

        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["enc_dec_attention_mask"] = input_ids.ne(pad_token_id).long()

        model_kwargs["encoder_outputs"] = self.model.encoder.forward(
            input_ids=input_ids, appendix=appendix, min_depth=min_depth, max_depth=max_depth,
            lattice_mask=lattice_mask, forward_backward=forward_backward, marginals=marginals,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, batch_idx=batch_idx
        )

        # set input_ids as decoder_input_ids
        if "decoder_input_ids" in model_kwargs:
            input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            input_ids = input_ids.new(input_ids.shape[0], 1).fill_(bos_token_id)

        if input_ids.shape[-1] >= max_length:
            logging.warning(
                f"Input length of decoder_input_ids is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # determine generation mode
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
        is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # set model_kwargs
        model_kwargs["use_cache"] = use_cache

        # get distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=0,
            encoder_input_ids=None,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
        )

        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # get probability distribution wrapper
            logits_wrapper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, num_return_sequences, **model_kwargs,
            )

            # sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_wrapper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(input_ids, num_beams, **model_kwargs)

            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            logits_wrapper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams
            )

            batch_size = input_ids.shape[0] * num_return_sequences

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # interleave with `num_beams * num_return_sequences`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, num_beams * num_return_sequences, **model_kwargs,
            )

            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_wrapper,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            batch_size = input_ids.shape[0]

            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping

            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            diverse_beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # interleave with `num_beams`
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, num_beams, **model_kwargs
            )
            return self.group_beam_search(
                input_ids,
                diverse_beam_scorer,
                logits_processor=logits_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

    def prepare_inputs_for_generation(self, decoder_input_ids, past=None, enc_dec_attention_mask=None,
                                      head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "enc_dec_attention_mask": enc_dec_attention_mask,
            "use_cache": use_cache  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids,
            expand_size: int = 1,
            attention_mask: torch.LongTensor = None,
            encoder_outputs: ModelOutput = None,
            **model_kwargs
    ):
        expanded_return_idx = torch.arange(input_ids.shape[0]).view(-1, 1) \
            .repeat(1, expand_size).view(-1).to(input_ids.device)

        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if "enc_dec_attention_mask" in model_kwargs:
            enc_dec_attention_mask = model_kwargs["enc_dec_attention_mask"]
            model_kwargs["enc_dec_attention_mask"] = enc_dec_attention_mask.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        # if is_encoder_decoder:
        assert encoder_outputs is not None, "encoder_outputs is None"
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
        )
        model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
