from typing import List

import torch
from transformers import BartTokenizer


def word_based_tokenize(srcs: List[str], tokenizer: BartTokenizer, lower=False, return_attention=True,
                        to_tensor=False) -> dict:
    """ Tokenizer会将词拆为bpe，而在序列标注问题中，最小建模单元需要为word. """
    input_ids, appendix = [], {}

    special_tokens = tokenizer.special_tokens_map.values()
    bos, eos = tokenizer.bos_token, tokenizer.eos_token

    for i, src in enumerate(srcs):
        tokens = src.lower().split() if lower else src.split()
        if tokens[0] != bos:
            tokens = [bos] + tokens + [eos]
        chunks = []
        for token in tokens:
            if token in special_tokens:
                chunks.append(tokenizer.encode(token, add_special_tokens=False))
            else:
                chunks.append(tokenizer.encode(token, add_prefix_space=True, add_special_tokens=False))

        word_start, word_end = [], {}

        for j, chunk in enumerate(chunks):
            word_start.append(chunk[0])
            if len(chunk) > 1:
                word_end[j] = chunk[1:]

        input_ids.append(word_start)
        if len(word_end) != 0:
            appendix[i] = word_end

    max_len = max(map(len, input_ids))
    attention_mask = [] if return_attention else None
    padding_id = tokenizer.pad_token_id

    for i, ids in enumerate(input_ids):
        L = len(ids)
        if return_attention:
            attention_mask.append([1] * L + [0] * (max_len - L))
        input_ids[i] = ids + (max_len - L) * [padding_id]

    inputs = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long) if to_tensor else input_ids,
        "appendix": appendix
    }
    if return_attention:
        inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long) if to_tensor else attention_mask,

    return inputs
