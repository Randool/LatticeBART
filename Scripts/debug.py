import logging
import os

import torch

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')

if os.name == "nt":
    bart_base_path = r"D:\data\trained_net\transformers\bart-base"
    bart_tiny_path = r"D:\data\trained_net\transformers\tinier_bart"
else:
    bart_base_path = "/home/randool/Dataset/transformers/bart-base"
    bart_tiny_path = "/home/randool/Dataset/transformers/tinier-bart"

dataset = "librispeech"
lattice_path = f"debug/data/{dataset}/pruned.lat.[1].*.txt"
lattice_ark_path = f"debug/data/{dataset}/train.ark"
text_path = f"debug/data/{dataset}/text"


#### DATA ####

def test_parse_lattice_file():
    from dataset.parse import parse_lattice_file

    lattices = parse_lattice_file(f"debug/data/{dataset}/pruned.lat.1.m200.b4.a1.l1.w0.txt",
                                  to_line_graph=True, lower_case=True)
    for lat_id, lat in lattices.items():
        print(lat_id)
        yield lat

    print(len(lattices))


def test_new_line_graph_process():
    from transformers import BartTokenizer
    from dataset.parse import line_graph_process

    tokenizer = BartTokenizer.from_pretrained(bart_base_path)

    for lat in test_parse_lattice_file():
        info = line_graph_process(lat, tokenizer)
        print(info)
        break


def test_dummy_lattice():
    import copy
    from math import log
    from collections import namedtuple
    import networkx as nx
    from dataset.utils import dump_graph
    from dataset.parse import lattice_to_line_graph, scoring_line_graph
    from dataset.graph_visual import display_line_graph

    T = namedtuple("T", ("u_for_edge", "v_for_edge", "word", "log_f"))
    edges = [
        (0, 1, "quick", 0.4),
        (0, 1, "quit", 0.4),
        (0, 2, "wick", 0.2),
        (1, 3, "brown", 0.6),
        (1, 4, "crown", 0.4),
        (2, 4, "gown", 1),
        (3, 5, "fast", 0.2),
        (3, 5, "fox", 0.8),
        (4, 5, "ox", 1),
        (5, 6, "</s>", 1)
    ]

    G = nx.MultiDiGraph()
    for edge in edges:
        edge = edge[:-1] + (log(edge[-1]),)
        edge = dict(T(*edge)._asdict())
        print(edge)
        G.add_edge(**dict(edge))

    # print(dump_graph(G))
    # display_lattice(G)

    L = lattice_to_line_graph(copy.deepcopy(G))
    L = scoring_line_graph(L)
    print(dump_graph(L))
    topo = nx.topological_sort(L)
    for x in topo:
        print(x, L.nodes[x])
    display_line_graph(L)


def test_display_lattice():
    from dataset.parse import parse_lattice_file, scoring_line_graph
    from dataset.graph_visual import display_line_graph

    lattices = parse_lattice_file(f"debug/data/{dataset}/pruned.lat.1.m200.b4.a1.l1.w0.txt", to_line_graph=True,
                                  lower_case=True)
    for lat_id, lat in lattices.items():
        print(lat_id)
        lat = scoring_line_graph(lat)
        display_line_graph(lat, lat_id)
        # break


def display_length_distribution():
    import matplotlib.pyplot as plt
    from dataset.dataset import LatticeTextDataset

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
    from dataset.dataset import LatticeTextDataset
    lat2text = LatticeTextDataset(lattice_pattern=lattice_path, text_pattern=text_path)
    for _id, lat, text, in lat2text:
        words = lat.words
        marginal = [f"{m:.1e}" for m in lat.marginal]
        # print(lat)
        print(_id, text)
        print(list(zip(words.split(), marginal)))
        _ = input()


def dummy_dataset(text_pattern=text_path):
    from dataset.dataset import LatticeTextDataset
    lat2text = LatticeTextDataset(lattice_pattern=lattice_path, text_pattern=text_pattern)
    return lat2text


def dummy_ark_dataset(text_pattern=text_path):
    from dataset.dataset import LatticeTextDataset
    lat2text = LatticeTextDataset(lattice_ark_pattern=lattice_ark_path, text_pattern=text_pattern)
    return lat2text


def dummy_dataloader(model_path=bart_tiny_path, return_text=True, use_ark=False, batch_size=4, n_lat=1,
                     decoder_input="lattice"):
    from transformers import BartTokenizer
    from torch.utils.data.dataloader import DataLoader
    from dataset.dataset import CollateFn

    text_pattern = text_path if return_text else None
    lat2text = dummy_ark_dataset(text_pattern) if use_ark else dummy_dataset(text_pattern)

    tokenizer = BartTokenizer.from_pretrained(model_path)
    collate_fn = CollateFn(tokenizer=tokenizer, n_lat=n_lat, decoder_input=decoder_input,
                           process_cfg=lat2text.process_cfg).collate_fn
    dataloader = DataLoader(lat2text, batch_size=batch_size,
                            collate_fn=collate_fn)

    return dataloader


def test_dataloader(use_ark=False, return_text=True):
    import torch
    for idx, inputs in enumerate(dummy_dataloader(return_text=return_text, batch_size=2, use_ark=use_ark, n_lat=2)):
        for key, val in inputs.items():
            # if key == "input_ids":
            #     print(val)
            if isinstance(val, torch.Tensor):
                print(key, val.shape)
            else:
                print(key, val)
        if idx >= 0:
            break
        print("===")


def _match_lattice_id():
    def get_keys(filename: str, bias=0) -> set:
        keys = set()
        with open(filename) as f:
            for line in f:
                key, *_ = line.split()
                keys.add(key[bias:])
        print(filename, "\t", len(keys))
        return keys

    # 来自rm
    fisher_keys = get_keys("/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher/fisher.txt")
    eval2000_keys = get_keys("/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher/eval2000.txt")
    rt03_keys = get_keys("/mnt/lustre/sjtu/home/lfd98/Dataset/mr_fisher/rt03.txt")

    # 来自zzk
    fisher_swbd_train_keys = get_keys("/mnt/lustre/sjtu/home/lfd98/Dataset/fisher_swbd/train/text", 6)
    fisher_swbd_test_keys = get_keys("/mnt/lustre/sjtu/home/lfd98/Dataset/fisher_swbd/test/text", 6)
    fisher_swbd_dev_keys = get_keys("/mnt/lustre/sjtu/home/lfd98/Dataset/fisher_swbd/dev/text", 6)

    print(len(fisher_keys & eval2000_keys))
    print(len(fisher_keys & rt03_keys))
    print()
    print(len(fisher_keys & fisher_swbd_train_keys))
    print(len(fisher_keys & fisher_swbd_test_keys))
    print(len(fisher_keys & fisher_swbd_dev_keys))
    print()
    print(len(fisher_swbd_train_keys & fisher_swbd_test_keys))
    print(len(fisher_swbd_train_keys & fisher_swbd_dev_keys))


def use_lattice_id_ark():
    from dataset.parse import parse_lattice_from_address

    id_map = {}
    with open("debug/librispeech.ark", encoding="utf8") as f:
        for line in f:
            key, *items = line.split()
            id_map[key] = items

    for key, file_poss in id_map.items():
        for file_pos in file_poss:
            lattice_id, G = parse_lattice_from_address(file_pos)
            print(lattice_id)
        break


#### MODEL ####

def show_model_param_with_shape(model_path=bart_tiny_path, lat_enc_layers=1):
    from models.lattice_bart import LatticeBart
    from transformers import BartForConditionalGeneration

    bart = BartForConditionalGeneration.from_pretrained(model_path)
    lattice_bart = LatticeBart.from_bart(model_path, lat_enc_layers=lat_enc_layers)

    for name, param in bart.named_parameters():
        print(param.shape, "\t", name)

    print()

    for name, param in lattice_bart.named_parameters():
        print(param.shape, "\t", name)


def test_model_init():
    from models.lattice_bart import LatticeBart
    from models.model_utils import model_init

    model = LatticeBart.from_bart(bart_tiny_path, 2)
    model_init(model, std=model.config.init_std, except_modules=(model.get_encoder().embed_tokens,))


def test_model_param_match(model_path=bart_tiny_path, lat_enc_layers=1):
    from models.lattice_bart import LatticeBart
    model = LatticeBart.from_bart(model_path, lat_enc_layers=lat_enc_layers)
    return model


def test_specific_layer_wise_lr():
    from torch.optim import AdamW
    from models.model_utils import config_layer_wise_lr

    model = test_model_param_match()
    tmp = config_layer_wise_lr(model, 1e-3, 0.95)

    optim = AdamW(tmp, lr=1e-5)
    optim.step()
    optim.zero_grad()
    lrs = [f"{group['lr']:.2e}" for group in optim.param_groups]
    if len(lrs) > 2:
        lrs = [lrs[0], lrs[-1]]
    lrs = " ~ ".join(lrs)
    print(optim)
    print(lrs)


def test_model_forward(model_path=bart_tiny_path):
    import torch

    model = test_model_param_match(model_path)
    # print(model)
    for i, inputs in enumerate(dummy_dataloader(model_path, return_text=True)):
        print("==== inputs ====")
        for key, val in inputs.items():
            if torch.is_tensor(val):
                print(key, val.shape)
            else:
                print(key, val)
        outputs = model.forward(**inputs, output_attentions=True, reduction="mean")
        loss1 = outputs["loss"]
        print("==== outputs ====")
        print("loss:", loss1)
        for key, val in outputs.items():
            if torch.is_tensor(val):
                print(key, val.shape)
            else:
                print(key, type(val))
        if i >= 0:
            break
        print()


def test_model_train(model_path=bart_tiny_path, device="cuda:0"):
    import torch
    from transformers import AdamW
    from dataset.dataset import dict_to_device
    from models.model_utils import config_layer_wise_lr

    dataloader = dummy_dataloader(model_path, use_ark=False, batch_size=8)

    model = test_model_param_match(model_path).train().to(device)
    optimizer = AdamW(config_layer_wise_lr(model, init_lr=1e-4, lr_decay=0.95), weight_decay=1e-5)

    for i, inputs in enumerate(dataloader, start=1):
        inputs = dict_to_device(inputs, device)
        outputs = model.forward(**inputs)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        print(f"{i}/{len(dataloader)}\tloss={loss.item():.2e}")
        optimizer.step()
        optimizer.zero_grad()


def test_model_decode_lattice(model_path=bart_tiny_path, device="cuda:0"):
    import torch
    from transformers import AdamW
    from dataset.dataset import dict_to_device
    from models.model_utils import config_layer_wise_lr

    # device = "cpu"

    dataloader = dummy_dataloader(model_path, return_text=True, use_ark=True, batch_size=4, decoder_input="lattice")
    model = test_model_param_match(model_path).train().to(device)
    optimizer = AdamW(config_layer_wise_lr(model, init_lr=1e-3, lr_decay=0.95), weight_decay=1e-5)

    show_keys = True

    for i, inputs in enumerate(dataloader, start=1):
        # if inputs["input_ids"].shape[1] >= 16:
        #     continue
        # if inputs["min_depth"].eq(inputs["max_depth"]).all():
        #     continue
        # print(inputs["lattice_mask"])
        if show_keys:
            show_keys = False
            print(inputs.keys())
        inputs = dict_to_device(inputs, device)
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


def test_early_stop():
    from utils.utils import EarlyStopping
    mtc = EarlyStopping(2, decrease=False)
    for v in (0.683, 0.662, 0.663, 0.688, 0.681, 0.678, 0.674):
        print(v, mtc.is_better(v), mtc.record(v), mtc.is_stop)


def test_homonym_replace():
    from dataset.dataset import TextDataset
    from utils.phone import HomophoneReplacer

    text_dataset = TextDataset("../debug/data/librispeech/text")
    hr = HomophoneReplacer()

    for key in text_dataset.keys():
        sent = text_dataset[key]
        print(sent.strip())
        print(hr.process_sent(sent, ratio=0.91))
        print()


def show_sim_mtx():
    import random
    import torch
    import matplotlib.pyplot as plt
    from utils.metric import show_similarity_mtx
    B = 64
    mtx = torch.randn(B, B)
    labels = [random.randint(10000, 200000) for _ in range(B)]
    show_similarity_mtx(mtx, include_values=False, display_labels=labels, xticks_rotation=70)
    plt.show()


def use_cmudict():
    from utils.phone import pronunciation, unstressed_phone_map, stressed_phone_map

    print(unstressed_phone_map.keys())
    print(stressed_phone_map.keys())

    sent1 = "I wreck a nice beach"
    sent2 = "I recognize speech"
    ph1 = [pronunciation(token, False) for token in sent1.split()]
    ph2 = [pronunciation(token, False) for token in sent2.split()]
    print(ph1)
    print(ph2)


def test_rescore_lm():
    from models.lstm_lm import RNNModel
    model = RNNModel(vocab_size=1000, embed_size=64, hidden_size=64, n_layers=2, dropout=0).eval()
    inputs = {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "targets": torch.randint(0, 100, (4, 10)),
        "length": torch.randint(2, 10, (4,))
    }
    print(model.calculate_logprob(**inputs))


if __name__ == '__main__':
    pass
