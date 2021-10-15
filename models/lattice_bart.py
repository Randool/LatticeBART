import json
import os
from os.path import join
from typing import Optional, Iterable, Callable, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BartForConditionalGeneration, BeamSearchScorer
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqModelOutput, Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import BartPretrainedModel

from utils.utils import log
from .config import LatticeBartConfig
from .decoder import LatticeDecoder
from .encoder import LatticeEncoder


class LatticeBartBase(BartPretrainedModel):
    def _reorder_cache(self, past, beam_idx):
        raise NotImplementedError

    def __init__(self, config: LatticeBartConfig):
        super().__init__(config)

        self.shared = nn.Embedding(config.vocab_size, config.d_model, config.pad_token_id)
        self.encoder = LatticeEncoder(config, self.shared)
        self.decoder = LatticeDecoder(config, self.shared)

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
    def forward(self, input_ids=None, min_depth=None, max_depth=None, lattice_mask=None, forward_backward=None,
                marginals=None, enc_dec_attention_mask=None, encoder_outputs=None, past_key_values=None,
                decoder_input_ids=None, decoder_attention_mask=None, decoder_min_depth=None, decoder_max_depth=None,
                use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True):
        # Encoder forward
        if encoder_outputs is None:
            encoder_outputs = self.encoder.forward(
                input_ids=input_ids,
                min_depth=min_depth,
                max_depth=max_depth,
                lattice_mask=lattice_mask,
                forward_backward=forward_backward,
                marginals=marginals,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if enc_dec_attention_mask is None:
                enc_dec_attention_mask = input_ids.ne(self.config.pad_token_id).long()

        # Decoder forward
        decoder_outputs = self.decoder.forward(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            min_depth=decoder_min_depth,
            max_depth=decoder_max_depth,
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

    @property
    def device(self):
        return next(self.parameters()).device


class LatticeBart(BartPretrainedModel):
    MODEL_PT = "lattice_bart.pt"
    config: LatticeBartConfig

    def __init__(self, config: LatticeBartConfig):
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
    def forward(self, input_ids=None, min_depth=None, max_depth=None, lattice_mask=None, forward_backward=None,
                marginals=None, enc_dec_attention_mask=None, encoder_outputs=None, past_key_values=None,
                decoder_input_ids=None, decoder_attention_mask=None, decoder_min_depth=None, decoder_max_depth=None,
                labels=None, use_cache=True, output_attentions=False, output_hidden_states=False, return_dict=True,
                reduction="mean"):
        outputs = self.model.forward(
            input_ids=input_ids,
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
            decoder_min_depth=decoder_min_depth,
            decoder_max_depth=decoder_max_depth,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias

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

    @property
    def device(self):
        return next(self.parameters()).device

    def save_pretrained(self, model_path: str):
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.state_dict(), join(model_path, self.MODEL_PT))
        with open(join(model_path, "config.json"), "w", encoding="utf8") as f:
            config = self.config.to_dict()
            json.dump(config, f, indent=2, ensure_ascii=False)
        # log(f"Save {self.__class__.__name__} to {model_path}", logger)

    @classmethod
    def from_pretrained(cls, model_path: str, *model_args, **kwargs):
        with open(join(model_path, "config.json"), encoding="utf8") as f:
            config = LatticeBartConfig.from_dict(json.load(f))
        lattice_bart = cls(config)
        lattice_bart.load_state_dict(torch.load(join(model_path, cls.MODEL_PT), map_location=lambda st, loc: st))
        # log(f"Load {cls.__class__.__name__} from {model_path}", logger)
        return lattice_bart

    @classmethod
    def from_bart(cls, bart_path: str, lat_enc_layers: int):
        bart = BartForConditionalGeneration.from_pretrained(bart_path)
        config = bart.config.to_dict()

        assert bart.config.encoder_layers >= lat_enc_layers, \
            f"lat_enc_layers({lat_enc_layers}) larger than " \
            f"bart.config.encoder_layers({bart.config.encoder_layers})"

        config["lat_enc_layers"] = lat_enc_layers
        lattice_bart = cls(LatticeBartConfig.from_dict(config))

        # Move parameters from pretrained bart
        params = dict(bart.named_parameters())

        # Offset Lattice Encoder's layers
        for i in range(bart.config.encoder_layers):
            if i < lat_enc_layers:
                r"""
                Apply replacements
                    layers => lattice_layers
                    self_attn => lattice_self_attn 
                on:
                    model.encoder.layers.\d+.self_attn.[kqv]_proj.(weight|bias)
                    model.encoder.layers.\d+.self_attn_layer_norm.(weight|bias)
                    model.encoder.layers.\d+.fc1.(weight|bias)
                """
                prefix = f"model.encoder.layers.{i}"
                for key in list(params.keys()):
                    if key.startswith(prefix):
                        new_key = key.replace(".layers.", ".lattice_layers.") \
                            .replace(".self_attn.", ".lattice_self_attn.")
                        params[new_key] = params[key]
                        del params[key]
            else:
                prefix1 = f"model.encoder.layers.{i}"
                prefix2 = f"model.encoder.layers.{i - lat_enc_layers}"
                for key in list(params.keys()):
                    if key.startswith(prefix1):
                        key_ = key.replace(prefix1, prefix2)
                        params[key_] = params[key]

        # Set parameters
        for key, param in lattice_bart.named_parameters():
            if key not in params:
                log(f"[x] {key}\t no found", None, "error")
                continue
            if param.shape != params[key].shape:
                log(f"[x] {key}\t shape mismatch between bart and lattice bart", None, "error")
                continue
            param.data = params[key].data
            # logging.debug(f"[v] {key}")

        return lattice_bart

    @torch.no_grad()
    def generate(
            self, input_ids, min_depth, max_depth, lattice_mask,
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
            input_ids=input_ids, min_depth=min_depth, max_depth=max_depth, lattice_mask=lattice_mask,
            forward_backward=forward_backward, marginals=marginals, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # set input_ids as decoder_input_ids
        if "decoder_input_ids" in model_kwargs:
            input_ids = model_kwargs.pop("decoder_input_ids")
        else:
            input_ids = input_ids.new(input_ids.shape[0], 1).fill_(bos_token_id)

        if input_ids.shape[-1] >= max_length:
            log(
                f"Input length of decoder_input_ids is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}."
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``.",
                None, "warning"
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

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
