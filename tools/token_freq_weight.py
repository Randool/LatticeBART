import argparse
import json
from math import log10
from os.path import join
from pathlib import Path

from tqdm import tqdm
from transformers import BartTokenizer


def parse_args():
    home = str(Path.home())
    _tokenizer_path = join(home, "Dataset/transformers/bart-base")

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", "-i", type=str, nargs="+", help="text files")
    parser.add_argument("--field", type=int, nargs="+", default=[1], help="Start token positions")
    parser.add_argument("--tokenizer", "-t", type=str, default=_tokenizer_path, help="Bart Tokenizer path")
    parser.add_argument("--save", "-o", type=str, required=True)
    args = parser.parse_args()

    if len(args.field) > 1:
        if len(args.files) != len(args.field):
            raise ValueError(f"Len. of field ({len(args.field)}) should be 1 or equal to files({len(args.files)})")
        args.field = args.field * len(args.files)

    print(args)
    return args


def tokenizer_file(filename: str, field: int, tokenizer, token_id_freq: dict):
    print(f"> {filename}")
    with open(filename, encoding="utf8") as f:
        for line in tqdm(f):
            if line.isspace() or len(line) == 0:
                continue
            tokens = line.split()[field:]
            sent = " ".join(tokens)
            for _id in tokenizer.encode(sent, add_special_tokens=True):
                token_id_freq[_id] += 1


def main():
    args = parse_args()
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer)
    token_id_freq = {_id: 1 for _token, _id in tokenizer.get_vocab().items()}

    # Get frequent
    for file, field in zip(args.files, args.field):
        tokenizer_file(file, field, tokenizer, token_id_freq)

    # Calculate weights
    for token_id, freq in token_id_freq.items():
        token_id_freq[token_id] = 1 / log10(freq + 2)

    # Dump
    with open(args.save, "w", encoding="utf8") as f:
        json.dump(token_id_freq, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
