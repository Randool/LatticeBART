import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 10)
        self.layers = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

    def forward(self, x):
        print(f"ToyModel: rank={dist.get_rank()}, word_size={dist.get_world_size()}")
        print(x)
        return self.layers(self.embed(x))


class ToyDataset(Dataset):
    def __init__(self, n):
        self.arr = [i for i in range(n)]

    def __getitem__(self, idx):
        return self.arr[idx]

    def __len__(self):
        return len(self.arr)


def example(rank, world_size, inputs, labels):
    print(f"rank={rank}, world_size={world_size}")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    inputs = torch.randint(0, 10, (5,)).to(rank)
    labels = torch.randn(5, 10).to(rank)
    outputs = ddp_model(inputs)

    loss_fn(outputs, labels).backward()
    optimizer.step()
    cleanup()


def main():

    mp.spawn(fn=example, args=(world_size,), nprocs=world_size)


if __name__ == '__main__':
    world_size = 2
    dist.init_process_group(backend="nccl", world_size=world_size)

    model = ToyModel().to(rank)
    main()
