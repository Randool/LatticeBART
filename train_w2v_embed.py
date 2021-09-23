import argparse
from typing import List

from gensim.models import Word2Vec
from tqdm import tqdm

from dataset.dataset import TextDataset
from utils.phone import pronunciation


def parse_sentences(dataset: TextDataset, args: argparse.Namespace):
    sentences: List[List[str]] = []
    stressed = args.token == "stressed"

    for key, line in tqdm(dataset.items.items(), desc="Parse sent"):
        if args.token == "token":
            sentences.append(line.split())
        else:
            sentences.append([phn for token in line.split() for phn in pronunciation(token, stressed)])

    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--text", type=str, nargs="+", required=True, help="Text file")
    parser.add_argument("-o", "--save", type=str, required=True, help="Save file")
    parser.add_argument("-t", "--token", type=str, choices=("token", "stressed", "unstressed"), default="stressed")
    parser.add_argument("--vector_size", type=int, default=768)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("-w", "--workers", type=int, default=4)
    args = parser.parse_args()

    if args.text is None:
        raise ValueError("`--text` should be given.")

    sentences: List[List[str]] = []
    for text in args.text:
        print(f"Parse: {text}")
        text_dataset = TextDataset(text)
        sentences.extend(parse_sentences(text_dataset, args))

    print("Training...")
    w2v = Word2Vec(sentences=sentences, vector_size=args.vector_size, window=args.window,
                   min_count=args.min_count, workers=args.workers, epochs=5)
    wv = w2v.wv
    print(f"Save to {args.save}")
    wv.save(args.save)  # Load: from gensim.models import KeyedVectors; wv = KeyedVectors.load("xxx")


if __name__ == "__main__":
    main()
