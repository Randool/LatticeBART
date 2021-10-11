import json
import os
from os.path import join
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class Vocab:
    def __init__(self, text_file: str = None):
        self.i2s = ["<pad>", "<s>", "</s>", "<unk>"]
        self.s2i = {token: i for i, token in enumerate(self.i2s)}
        self.pad = "<pad>"
        self.bos = "<s>"
        self.eos = "</s>"
        self.unk = "<unk>"
        self.pad_id = self.s2i[self.pad]
        self.bos_id = self.s2i[self.bos]
        self.eos_id = self.s2i[self.eos]
        self.unk_id = self.s2i[self.unk]

        if text_file is not None:
            self.extend_vocab(text_file)

    def extend_vocab(self, file: str):
        with open(file) as f:
            for line in f:
                tokens = line.strip().split()[1:]
                for token in tokens:
                    if token not in self.s2i:
                        self.s2i[token] = len(self.s2i)
                        self.i2s.append(token)

    def __len__(self):
        return len(self.i2s)

    def _pad_seqs(self, seqs: List[List[int]]):
        max_len = max(map(len, seqs))
        for i, seq in enumerate(seqs):
            seqs[i] = seq + (max_len - len(seq)) * [self.pad_id]
        return torch.tensor(seqs, dtype=torch.long)

    def save_pretrained(self, path):
        with open(join(path, "vocab.json"), "w", encoding="utf8") as f:
            json.dump(self.i2s, f)

    @classmethod
    def from_pretrained(cls, path):
        vocab = cls()
        with open(join(path, "vocab.json"), encoding="utf8") as f:
            i2s = json.load(f)
        vocab.i2s = i2s
        vocab.s2i = {t: i for i, t in enumerate(i2s)}
        return vocab

    def collate_fn(self, batch_sent: List[str]) -> dict:
        input_ids = []
        length = []
        for sent in batch_sent:
            ids = [self.bos_id]
            tokens = sent.strip().split()
            for token in tokens:
                if token not in self.s2i:
                    token = "<unk>"
                ids.append(self.s2i[token])
            ids.append(self.eos_id)
            input_ids.append(ids)
            length.append(len(ids) - 1)

        input_ids = self._pad_seqs(input_ids)
        inputs = {
            "input_ids": input_ids[:, :-1].contiguous(),
            "targets": input_ids[:, 1:].contiguous(),
            "length": length
        }
        return inputs


class RNNModel(nn.Module):
    PT_FILE = "RNNModel.pt"

    def __init__(self, vocab_size, embed_size, hidden_size, n_layers, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.layers = nn.LSTM(embed_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(hidden_size, vocab_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, targets=None) -> dict:
        embed = self.embed(input_ids)
        hidden_states, _ = self.layers(embed)
        hidden_states = self.proj(hidden_states)
        if targets is not None:
            bsz, seq_len, dim = hidden_states.shape
            loss = F.cross_entropy(hidden_states.reshape(bsz * seq_len, dim), targets.reshape(-1))
            return {"loss": loss}
        else:
            return {"hidden_states": hidden_states}

    @torch.no_grad()
    def calculate_logprob(self, input_ids, targets, length):
        hidden_states = self.forward(input_ids)["hidden_states"]
        bsz, seq_len, dim = hidden_states.shape
        logprob = - F.cross_entropy(hidden_states.reshape(bsz * seq_len, dim), targets.reshape(-1), reduction="none")
        logprob = logprob.view(bsz, seq_len)
        logprob = [logprob[i][:length[i]].mean().item() for i in range(bsz)]
        return logprob

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), join(path, self.PT_FILE))
        with open(join(path, "config.json"), "w", encoding="utf8") as f:
            config = {
                "vocab_size": self.vocab_size, "embed_size": self.embed_size,
                "hidden_size": self.hidden_size, "n_layers": self.n_layers,
                "dropout": self.dropout,
            }
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, path: str):
        with open(join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(join(path, cls.PT_FILE), map_location=lambda st, loc: st))
        return model
