import argparse

import torch.cuda
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TextDataset, dict_to_device
from models.lstm_lm import Vocab, RNNModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-m", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--sent_file", "-i", type=str, required=True)
    parser.add_argument("--write_file", "-o", type=str, required=True)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = RNNModel.from_pretrained(args.ckpt).to(device).eval()
    vocab = Vocab.from_pretrained(args.vocab)

    dataset = TextDataset(args.sent_file)
    collate_fn = vocab.collate_fn

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    logprob = []

    for batch in tqdm(dataloader, desc=""):
        batch = dict_to_device(batch, device)
        _logprob = model.calculate_logprob(**batch)
        logprob.extend(_logprob)

    with open(args.write_file, "w", encoding="utf8") as f:
        lines = [f"{x:.5f}" for x in logprob]
        f.write("\n".join(lines))


if __name__ == '__main__':
    main()
