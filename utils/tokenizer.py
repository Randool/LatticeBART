import random
import re
from collections import defaultdict
from typing import List, Tuple

import cmudict
import torch
from g2p_en import G2p
from transformers import BartTokenizer


class HomophoneReplacer:
    def __init__(self):
        self.CMU_dict = {k.lower(): tuple(v) for k, v in cmudict.entries()}
        self.CMU_dict_unstressed = {}
        self.g2p = G2p()  # Grapheme to phone guesser

        self.pron_to_tokens = defaultdict(list)
        for k, v in self.CMU_dict.items():
            self.pron_to_tokens[v].append(k)
            self.pron_to_tokens[self.unstressed(v)].append(k)

    def _pronunciation(self, token: str) -> Tuple[str]:
        """ Look up and cache pronunciation """
        if token in self.CMU_dict:
            return self.CMU_dict[token]
        pron = self.CMU_dict[token] = tuple(self.g2p(token))
        self.pron_to_tokens[pron].append(token)
        return pron

    @staticmethod
    def unstressed(pron: Tuple[str]) -> Tuple[str]:
        return tuple(p[:-1] if p[-1].isdigit() else p for p in pron)

    def find_homophones(self, token: str) -> List[str]:
        token = token.lower()
        pron = self._pronunciation(token)
        homophones = self.pron_to_tokens.get(pron)

        if homophones is None:  # Miss and Loop up for the first time
            unstressed_pron = self.unstressed(pron)
            homophones = self.pron_to_tokens[pron] = self.pron_to_tokens[unstressed_pron]

        try:
            homophones.remove(token)
        except ValueError:
            pass

        return homophones

    def process_sent(self, sent: str, ratio=0.1) -> str:
        assert 0 <= ratio <= 1, f"The value of ratio ({ratio}) is out of range"
        tokens = re.split("\\s+", re.sub("[.!?]", "", sent))

        for i, token in enumerate(tokens):
            if random.random() <= ratio:
                homos = self.find_homophones(token)
                if len(homos) == 0:
                    continue
                homo = random.choice(homos)
                tokens[i] = homo

        return " ".join(tokens)


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
