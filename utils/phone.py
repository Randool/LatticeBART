import random
import re
from collections import defaultdict
from random import random, choice
from typing import Tuple, List

import cmudict
from g2p_en import G2p

# About CMU dict: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
CMU_dict = {k.lower(): tuple(v) for k, v in cmudict.entries()}
g2p = G2p()  # Grapheme to phone guesser

# Build phone to index map
unstressed_phone_map = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
stressed_phone_map = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

phones = cmudict.phones()

for _p, phone_type in phones:  # type: str, list
    unstressed_phone_map[_p] = len(unstressed_phone_map)
    if phone_type[0] == "vowel":
        for _i in range(3):
            stressed_phone_map[f"{_p}{_i}"] = len(stressed_phone_map)
    else:
        stressed_phone_map[_p] = len(stressed_phone_map)


def unstressed_pron(pron: Tuple[str]) -> Tuple[str]:
    return tuple(pn[:-1] if pn[-1].isdigit() else pn for pn in pron)


def pronunciation(token: str, stressed=False) -> Tuple[str]:
    token = token.lower()
    if token in CMU_dict:
        pron = CMU_dict[token]
    else:
        guessed_pron = [p for p in tuple(g2p(token)) if p in stressed_phone_map]
        pron = CMU_dict[token] = tuple(guessed_pron)

    pron = pron if stressed else unstressed_pron(pron)
    return pron


class HomophoneReplacer:
    def __init__(self):
        self.pron_to_tokens = defaultdict(list)
        for k, v in CMU_dict.items():
            self.pron_to_tokens[v].append(k)
            self.pron_to_tokens[unstressed_pron(v)].append(k)

    def find_homophones(self, token: str) -> List[str]:
        pron = pronunciation(token)
        homophones = self.pron_to_tokens.get(pron)

        # Miss and Loop up for the first time
        if homophones is None:
            homophones = self.pron_to_tokens[pron] = self.pron_to_tokens[unstressed_pron(pron)]

        try:
            homophones.remove(token)
        except ValueError:
            pass

        return homophones

    def process_sent(self, sent: str, ratio=0.1) -> str:
        assert 0 <= ratio <= 1, f"The value of ratio ({ratio}) is out of range"
        tokens = re.split("\\s+", re.sub("[.!?]", "", sent))

        for i, token in enumerate(tokens):
            if random() <= ratio:
                homos = self.find_homophones(token)
                if len(homos) == 0:
                    continue
                homo = choice(homos)
                tokens[i] = homo

        return " ".join(tokens)
