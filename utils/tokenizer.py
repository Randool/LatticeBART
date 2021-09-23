from typing import List

import torch
from transformers import BartTokenizer

from .phone import unstressed_phone_map, stressed_phone_map, pronunciation


def word_based_tokenize(srcs: List[str], tokenizer: BartTokenizer, lower=False, return_attention=True,
                        to_tensor=True, padding=True, phone_embed="none") -> dict:
    """ Tokenizer会将词拆为bpe，而在序列标注问题中，最小建模单元需要为word. """
    with_phone, stressed = (phone_embed != "none"), (phone_embed == "stressed")
    phone_map = stressed_phone_map if stressed else unstressed_phone_map

    input_ids, appendix = [], {}
    phone_ids, phone_appendix = ([], {}) if with_phone else (None, None)

    special_tokens = tokenizer.special_tokens_map.values()
    bos, eos = tokenizer.bos_token, tokenizer.eos_token

    for i, src in enumerate(srcs):
        tokens = src.lower().split() if lower else src.split()
        tokens = [bos] + tokens + [eos] if (tokens[0] != bos) else tokens

        # map tokens to sequence of tokens (or phones) chunks
        chunks = []
        word_start, word_end = [], {}

        phone_chunks = [] if with_phone else None
        phone_start, phone_end = ([], {}) if with_phone else (None, None)

        for token in tokens:
            if token in special_tokens:
                chunks.append(tokenizer.encode(token, add_special_tokens=False))
                if with_phone:
                    phone_chunks.append([phone_map[token]])
            else:
                chunks.append(tokenizer.encode(token, add_prefix_space=True, add_special_tokens=False))
                if with_phone:
                    phone_chunks.append([phone_map[_p] for _p in pronunciation(token, stressed)])

        for j, chunk in enumerate(chunks):
            word_start.append(chunk[0])
            if len(chunk) > 1:
                word_end[j] = chunk[1:]

        input_ids.append(word_start)
        if len(word_end) != 0:
            appendix[i] = word_end

        if with_phone:
            for j, chunk in enumerate(phone_chunks):
                phone_start.append(chunk[0])
                if len(chunk) > 1:
                    phone_end[j] = chunk[1:]

            phone_ids.append(phone_start)
            if len(phone_end) != 0:
                phone_appendix[i] = phone_end

    attention_mask = [] if return_attention else None

    if to_tensor or padding:
        max_len = max(map(len, input_ids))
        padding_id = tokenizer.pad_token_id

        for i, ids in enumerate(input_ids):
            L = len(ids)
            if return_attention:
                attention_mask.append([1] * L + [0] * (max_len - L))
            input_ids[i] = ids + (max_len - L) * [padding_id]
            if with_phone:
                phone_ids[i] = phone_ids[i] + (max_len - L) * [padding_id]

    inputs = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long) if to_tensor else input_ids,
        "appendix": appendix
    }
    if return_attention:
        inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long) if to_tensor else attention_mask,

    if with_phone:
        inputs["phone_ids"] = torch.tensor(phone_ids, dtype=torch.long) if to_tensor else phone_ids
        inputs["phone_appendix"] = phone_appendix

    return inputs
