from itertools import product
from logging import Logger
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu

from utils.utils import log


def precision(pred, refer) -> float:
    """
    二分类任务
    precision = TP / (TP + FP)
    """
    if isinstance(pred, (np.ndarray, torch.Tensor)):
        pred = pred.tolist()
    if isinstance(refer, (np.ndarray, torch.Tensor)):
        refer = refer.tolist()

    TP = sum(a * b for a, b in zip(pred, refer))
    FP = sum(a * (1 - b) for a, b in zip(pred, refer))
    return TP / (TP + FP)


def recall(pred, refer) -> float:
    """
    二分类任务
    recall = TP / (TP + FN)
    """
    if isinstance(pred, (np.ndarray, torch.Tensor)):
        pred = pred.tolist()
    if isinstance(refer, (np.ndarray, torch.Tensor)):
        refer = refer.tolist()

    TP = sum(a * b for a, b in zip(pred, refer))
    FN = sum((1 - a) * b for a, b in zip(pred, refer))
    return TP / (TP + FN)


def F1(pred, refer) -> float:
    """
    二分类任务
    F1 = 2 * precision * recall / (precision + recall)
    """
    if isinstance(pred, (np.ndarray, torch.Tensor)):
        pred = pred.tolist()
    if isinstance(refer, (np.ndarray, torch.Tensor)):
        refer = refer.tolist()

    p = precision(pred, refer)
    r = recall(pred, refer)
    return 2 * p * r / (p + r)


def sentence_accuracy(pred: List[str], refer: List[str]) -> float:
    assert len(pred) == len(refer), f"Length mismatch, pred=({len(pred)}) while refer=({len(refer)})"
    if len(pred) == 0:
        return 0
    cmp = [s1.strip() == s2.strip() for s1, s2 in zip(pred, refer)]
    return sum(cmp) / len(cmp)


def BLEU(refer: List[List[str]], hypo: List[List[str]]) -> float:
    """
    :param refer:   [sent1, sent2, ...] where sent is List[str]
    :param hypo:    [sent1, sent2, ...] where sent is List[str]
    :return:    The corpus-level BLEU score.
    """
    assert len(refer) == len(hypo)
    # TODO: BLEU应该用什么smoothing函数？
    refer_list = [[s] for s in refer]
    bleu = corpus_bleu(refer_list, hypo)
    return bleu


class ErrorState:
    def __init__(self, ins_num=0, del_num=0, sub_num=0, total_cost=0):
        self.ins_num = ins_num
        self.del_num = del_num
        self.sub_num = sub_num
        self.total_cost = total_cost

    def __str__(self):
        return f"<I={self.ins_num},D={self.del_num},S={self.sub_num},T={self.total_cost}>"

    def copy_state(self, other):
        other.ins_num = self.ins_num
        other.del_num = self.del_num
        other.sub_num = self.sub_num
        other.total_cost = self.total_cost


def levenshtein_edit_distance(ref: List[str], hyp: List[str]) -> ErrorState:
    e = [ErrorState() for _ in range(len(ref) + 1)]
    cur_e = [ErrorState() for _ in range(len(ref) + 1)]

    for i, state in enumerate(e):
        state.sub_num = state.ins_num = 0
        state.total_cost = state.del_num = i

    for hyp_index in range(1, len(hyp) + 1):
        e[0].copy_state(cur_e[0])
        cur_e[0].ins_num += 1
        cur_e[0].total_cost += 1

        for ref_index in range(1, len(ref) + 1):
            ins_err = e[ref_index].total_cost + 1
            del_err = cur_e[ref_index - 1].total_cost + 1
            sub_err = e[ref_index - 1].total_cost

            if hyp[hyp_index - 1] != ref[ref_index - 1]:
                sub_err += 1

            if sub_err < ins_err and sub_err < del_err:
                e[ref_index - 1].copy_state(cur_e[ref_index])
                if hyp[hyp_index - 1] != ref[ref_index - 1]:
                    cur_e[ref_index].sub_num += 1
                cur_e[ref_index].total_cost = sub_err

            elif del_err < ins_err:
                cur_e[ref_index - 1].copy_state(cur_e[ref_index])
                cur_e[ref_index].total_cost = del_err
                cur_e[ref_index].del_num += 1

            else:
                e[ref_index].copy_state(cur_e[ref_index])
                cur_e[ref_index].total_cost = ins_err
                cur_e[ref_index].ins_num += 1

        for i in range(len(e)):
            cur_e[i].copy_state(e[i])

    return e[len(e) - 1]


def WER(refers: List[List[str]], hypos: List[List[str]], is_wer=True, logger: Logger = None) -> float:
    assert len(refers) == len(hypos)

    num_words = 0
    ins_num = del_num = sub_num = error_words = 0

    for ref, hyp in zip(refers, hypos):
        num_words += len(ref)
        state = levenshtein_edit_distance(ref, hyp)
        ins_num += state.ins_num
        del_num += state.del_num
        sub_num += state.sub_num
        error_words += state.total_cost

    if num_words == 0:
        log("zero word in sentences", logger)
        num_words += 1e-9

    wer = error_words / num_words

    leader = "WER" if is_wer else "CER"
    res = f"{leader} = {wer:.2%}, " \
          f"INS = {ins_num} ({ins_num / num_words:.2%}), " \
          f"DEL = {del_num} ({del_num / num_words:.2%}), " \
          f"SUB={sub_num} ({sub_num / num_words:.2%})"
    log(res, logger)
    return wer


def cos_similarity_mtx(vectors1, vectors2):
    """ vectors[1|2] of shape [B, D], return [B, B] """
    n1, n2 = vectors1.norm(dim=1, keepdim=True), vectors2.norm(dim=1, keepdim=True)
    eps = torch.finfo(n1.dtype).eps
    norm_1 = vectors1 / torch.clamp(n1, min=eps)
    norm_2 = vectors2 / torch.clamp(n2, min=eps)
    sim_mtx = torch.mm(norm_1, norm_2.transpose(0, 1))
    return sim_mtx


def show_similarity_mtx(mtx, include_values=False, display_labels: list = None, cmap='viridis',
                        xticks_rotation=60, values_format: str = None):
    """
    Args:
        mtx:            similarity matrix of shape [n_classes, n_classes]
        include_values: include values in the graph
        display_labels: display labels on x-tick and y-tick
        cmap:           see: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        xticks_rotation:    x-ticks rotation degree
        values_format:  values format
    """
    import matplotlib.pyplot as plt

    if torch.is_tensor(mtx):
        mtx = mtx.detach().cpu().numpy()

    n_classes = mtx.shape[0]

    fig, ax = plt.subplots()
    W = 8 * n_classes / 32
    H = 6 * n_classes / 32
    fig.set_size_inches(W, H)  # W, H
    im_ = ax.imshow(mtx, interpolation='nearest', cmap=cmap)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(mtx, dtype=object)
        thresh = (mtx.max() + mtx.min()) / 2.0

        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if mtx[i, j] < thresh else cmap_min

            if values_format is None:
                text_cm = format(mtx[i, j], '.2g')
                if mtx.dtype.kind != 'f':
                    text_d = format(mtx[i, j], 'd')
                    if len(text_d) < len(text_cm):
                        text_cm = text_d
            else:
                text_cm = format(mtx[i, j], values_format)

            text_[i, j] = ax.text(j, i, text_cm, ha="center", va="center", color=color)

    if display_labels is None:
        display_labels = np.arange(n_classes)

    fig.colorbar(im_, ax=ax)
    ax.set(xticks=np.arange(n_classes), yticks=np.arange(n_classes),
           xticklabels=display_labels, yticklabels=display_labels)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.xticks(rotation=xticks_rotation)

    return fig


def contrastive_loss(sim_mtx, reduction="mean", improved=False):
    assert sim_mtx.shape[-2] == sim_mtx.shape[-1]

    if improved:
        """
        Contrastive loss improved by `Simpler, Faster, Stronger:
            Breaking The log-K Curse On Contrastive Learners With FlatNCE`
        """
        exp_mtx = torch.exp(sim_mtx)
        diag_val = torch.diagonal(exp_mtx)
        loss = torch.log(exp_mtx.sum(dim=1) - diag_val) - diag_val
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
    else:
        index = torch.arange(0, sim_mtx.shape[-1], dtype=torch.long, device=sim_mtx.device)
        loss = F.cross_entropy(sim_mtx, index, reduction=reduction)

    return loss
