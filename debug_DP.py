import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO,
                    datefmt='%H:%M:%S', format='[%(asctime)s] %(levelname)s - %(message)s')


class ToyDataset(Dataset):
    def __init__(self, n):
        self.Xs = [i % 10 for i in range(n)]
        self.Ys = [random.randint(0, 9) for _ in range(n)]

    def __getitem__(self, idx):
        return self.Xs[idx], self.Ys[idx]

    def __len__(self):
        return len(self.Xs)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 10)
        self.layers = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

    def forward(self, x, target=None, reduction='mean'):
        logging.info(f"In model: {x}")
        x = self.layers(self.embed(x))
        outputs = {"hidden": x}
        if target is not None:
            outputs["loss"] = F.cross_entropy(x, target, reduction=reduction)
        return outputs


def main():
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    model = ToyModel().to(device)
    model = nn.DataParallel(model)

    dataset = ToyDataset(10000)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=2)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    for idx, (inputs, targets) in enumerate(dataloader):
        print(inputs)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model.forward(inputs, targets, reduction="mean")
        outputs2 = model.forward(inputs, targets, reduction="none")
        print(outputs["loss"])
        print(outputs["loss"].mean())
        print(outputs2["loss"].mean())
        loss = outputs["loss"]
        # loss = criterion(hidden, targets)
        print(loss)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        if idx >= 2:
            break


if __name__ == '__main__':
    main()
