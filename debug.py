import logging
import os
import random

import torch

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')

if os.name == "nt":
    bart_base_path = r"D:\data\trained_net\transformers\bart-base"
    bart_tiny_path = r"D:\data\trained_net\transformers\tinier_bart"
else:
    bart_base_path = "/home/randool/Dataset/transformers/bart-base"
    bart_tiny_path = "/home/randool/Dataset/transformers/tinier-bart"

dataset = "librispeech"
lattice_path = f"debug/{dataset}/pruned.lat.[1].*.txt"
text_path = f"debug/{dataset}/text"


def display_lattice():
    from utils.dataset import parse_lattice_file, scoring_line_graph
    from utils.graph_visual import display_line_graph

    lattices = parse_lattice_file(f"debug/{dataset}/pruned.lat.1.txt", to_line_graph=True, lower_case=True)
    for lat_id, lat in lattices.items():
        print(lat_id)
        lat = scoring_line_graph(lat)
        display_line_graph(lat, lat_id)
        # break


def display_length_distribution():
    import matplotlib.pyplot as plt
    from utils.dataset import LatticeTextDataset

    lat2text = LatticeTextDataset(lattice_pattern=lattice_path, text_pattern=text_path)
    len_node = []
    len_text = []
    for _id, G, text in lat2text.items:
        n_node = len(G)
        n_word = len(text.split())
        len_node.append(n_node)
        len_text.append(n_word)
        if n_node < n_word:
            print(f"{_id}, #node={n_node}, #text={n_word}")
    plt.scatter(len_text, len_node)
    plt.xlabel("#word")
    plt.ylabel("#node")
    plt.tight_layout()
    plt.show()


def show_lattice_item():
    from utils.dataset import LatticeTextDataset
    lat2text = LatticeTextDataset(lattice_pattern=lattice_path, text_pattern=text_path)
    for _id, lat, text, in lat2text:
        words = lat.words
        marginal = [f"{m:.1e}" for m in lat.marginal]
        # print(lat)
        print(_id, text)
        print(list(zip(words.split(), marginal)))
        _ = input()


def dummy_dataset(min_nodes=1, max_nodes=100, **kwargs):
    from utils.dataset import LatticeTextDataset
    lat2text = LatticeTextDataset(lattice_pattern=lattice_path, text_pattern=text_path,
                                  min_nodes=min_nodes, max_nodes=max_nodes, **kwargs)
    lat2text.sort_by_length()
    return lat2text


def dummy_dataloader(model_path=bart_tiny_path, batch_size=4, min_nodes=10, max_nodes=100, n_lat=1, **kwargs):
    from transformers import BartTokenizer
    from torch.utils.data.dataloader import DataLoader

    lat2text = dummy_dataset(min_nodes=min_nodes, max_nodes=max_nodes, **kwargs)

    tokenizer = BartTokenizer.from_pretrained(model_path)
    dataloader = DataLoader(lat2text, batch_size=batch_size,
                            collate_fn=lat2text.partial_collate_fn(tokenizer, n_lat=n_lat))

    return dataloader


def test_dataloader():
    for idx, inputs in enumerate(dummy_dataloader(batch_size=2, use_min_depth=False, n_lat=2)):
        for key, val in inputs.items():
            if isinstance(val, torch.Tensor):
                print(key, val.shape)
            else:
                print(key, val)
        if idx > 2:
            break
        print("===")


def test_model_param_match(model_path=bart_tiny_path):
    from models.lattice_bart import LatticeBart
    model = LatticeBart.from_bart(model_path)
    return model


def test_model_forward(model_path=bart_tiny_path):
    model = test_model_param_match(model_path)

    for i, inputs in enumerate(dummy_dataloader(model_path)):
        print(inputs.keys())
        outputs = model.forward(**inputs, output_attentions=True, reduction="mean")
        loss1 = outputs["loss"]
        print(outputs.keys())
        print(loss1)
        for key in outputs.keys():
            if isinstance(outputs[key], torch.Tensor):
                print(key, outputs[key].shape)
        print()
        if i > 3:
            break


def test_model_train(model_path=bart_tiny_path, **kwargs):
    from transformers import AdamW

    dataloader = dummy_dataloader(model_path, 16, 0, 200, **kwargs)

    model = test_model_param_match(model_path).train()
    optimizer = AdamW(model.parameters(), lr=1e-1, weight_decay=1e-4)

    for i, inputs in enumerate(dataloader, start=1):
        outputs = model.forward(**inputs)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        print(f"{i}/{len(dataloader)}\tloss={loss.item():.2e}")
        optimizer.step()
        optimizer.zero_grad()


def _test_bart_generate(model_path=bart_tiny_path):
    from transformers import BartForConditionalGeneration, BartTokenizer

    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)

    input_context = ['My cute dog', 'His cute dog is small']
    bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
    inputs = tokenizer.batch_encode_plus(input_context, padding=True, return_tensors='pt')

    outputs = model.generate(
        **inputs,
        # num_beams=2,
        # max_length=50,
        # do_sample=True,
        bad_words_ids=bad_words_ids
    )
    print(tokenizer.batch_decode(outputs))


def test_lattice_bart_generate(model_path=bart_tiny_path):
    from transformers import BartTokenizer

    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = test_model_param_match(model_path).eval()

    bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]

    for inputs in dummy_dataloader(model_path):
        inputs.pop("decoder_input_ids", None)
        inputs.pop("decoder_attention_mask", None)
        inputs.pop("labels", None)
        print(inputs.keys())
        outputs = model.generate(
            **inputs,
            num_beams=2,
            # max_length=50,
            do_sample=True,
            bad_words_ids=bad_words_ids
        )
        print(tokenizer.batch_decode(outputs))
        break


def test_bleu():
    from typing import List
    from utils.metric import BLEU
    hypo = [
        'unfortunately his work never does get good',
        'he eats and sleeps very steadily replied the new king',
        "he doesn't work all"
    ]
    refer = [
        'only unfortunately his own work never does get good',
        'he eats and sleeps very steadily replied the new king',
        "he doesn't work at all"
    ]
    hypo = [s.split() for s in hypo]  # type: List[List[str]]
    refer = [s.split() for s in refer]  # type: List[List[str]]
    print(BLEU(refer, hypo))


def test_wer():
    from utils.metric import WER

    refer = '1 2 3 4 5 6 7 8 9'.split()
    hypo = "2 3 4 5 6 7 9 8".split()

    a = WER([refer], [hypo])
    print(a)


def test_num_workers():
    import time
    import threading
    from torch.utils.data import Dataset, DataLoader

    def get_thread_info():
        return f"{os.getpid()} - {threading.current_thread().name} - {threading.get_ident()}"

    class DummyDataset(Dataset):
        def __init__(self, n):
            self.arr = [idx for idx in range(n)]

        def __getitem__(self, idx):
            print(f"{get_thread_info()} __getitem__: {idx}")
            return self.arr[idx]

        def __len__(self):
            return len(self.arr)

    def collate_fn(batch_data):
        print(f"{get_thread_info()} collate_fn: {batch_data}")
        return batch_data

    loader = DataLoader(DummyDataset(1000), batch_size=4, num_workers=4, collate_fn=collate_fn)

    for i, batch in enumerate(loader):
        print(batch)
        time.sleep(1)
        if i >= 1:
            break


def test_distributed_scatter():
    from torch.nn.parallel.scatter_gather import scatter_kwargs

    inputs = {
        "appendix": {1: [1, 1], 2: [2, 2], 3: [3, 3]},
        "args": "Haha",
        "list": [[1, 1], [2, 2], [3, 3]]
    }

    def func(*a, **kwargs):
        print(scatter_kwargs(a, kwargs, target_gpus=["cuda:0", "cuda:1"]))

    func(**inputs)


def test_model_init():
    from models.lattice_bart import LatticeBart
    from utils.utils import model_init

    model = LatticeBart.from_bart(bart_tiny_path)
    model_init(model, std=model.config.init_std, except_modules=(model.get_encoder().embed_tokens,))


##### 常驻 #####
def test_args():
    import argparse
    from utils.args import add_args_for_model_construct, add_args_for_contrastive_learning
    parser = argparse.ArgumentParser()
    add_args_for_model_construct(parser)
    add_args_for_contrastive_learning(parser)
    args = parser.parse_args()
    print(args)


##########


def test_forward_twice():
    from inspect import signature
    from models.lattice_bart import LatticeBart

    model = LatticeBart.from_bart(bart_tiny_path)
    sig = signature(model.get_encoder().forward).parameters
    enc_param = sig
    print(enc_param)

    for i, inputs in enumerate(dummy_dataloader()):
        print(inputs.keys())
        # Main forward
        outputs = model.forward(**inputs, output_attentions=True, reduction="mean")
        print(outputs.keys())

        # Enc forward
        enc_inputs = {k: v for k, v in inputs.items() if k in enc_param}
        enc_outputs = model.get_encoder().forward(**enc_inputs)
        print(enc_outputs.keys())

        loss = outputs["loss"]
        print(loss)

        if i > 2:
            break
        print()


def show_pred_distribution():
    from torch.utils.data import DataLoader
    from transformers import BartTokenizer
    from models.lattice_bart import EncoderWrapper
    from train_lattice_encoder import cos_similarity_mtx, contrastive_loss
    from utils.dataset import LatticeTextDataset
    from utils.utils import model_init
    from transformers.trainer_callback import TrainerState

    lat2text = LatticeTextDataset(lattice_pattern=lattice_path, text_pattern=None, min_nodes=0, max_nodes=100)
    lat2text.sort_by_length()

    path = "./exp/lattice_encoder/bart_path-bart-base,init_model,lr-0.001,optim-lamb,seed-1024,train_batch-32,contrast_method-dropout,gpu-2/"
    ckpt_path = os.path.join(path, "epoch_3")
    tokenizer = BartTokenizer.from_pretrained(path)
    wrapper = EncoderWrapper.from_pretrained(ckpt_path).train()
    model_init(wrapper, std=wrapper.config.init_std)
    # print(wrapper)

    dataloader = DataLoader(lat2text, batch_size=32, collate_fn=lat2text.partial_collate_fn(tokenizer))

    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            outs1 = wrapper.forward(**inputs, eos_id=tokenizer.eos_token_id)
            outs2 = wrapper.forward(**inputs, eos_id=tokenizer.eos_token_id)
            print(outs1)
            print(outs2)
            mtx = cos_similarity_mtx(outs1, outs2)
            print(mtx)
            loss = contrastive_loss(mtx, reduction='none')
            print(loss)
            if i >= 0:
                break



if __name__ == '__main__':
    show_pred_distribution()
