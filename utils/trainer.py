import time
from logging import Logger
from typing import Union, Optional, Callable, Dict, List, Tuple

import torch
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from transformers import Seq2SeqTrainer, PreTrainedModel, TrainingArguments, DataCollator, PreTrainedTokenizerBase, \
    TrainerCallback, EvalPrediction, TrainerState, TrainerControl, PrinterCallback, BartTokenizer

from dataset import dict_to_device
from models import LatticeBart
from utils.metric import BLEU, WER


class LoggerCallback(TrainerCallback):
    def __init__(self, logger: Logger = None):
        self.logger = logger

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: dict = None, **kwargs):
        if logs is None:
            return
        _msg = []
        for k in sorted(logs.keys()):
            if k == "epoch":
                _msg.append(f"Ep: {logs[k]:.2f}")
            elif k in ("learning_rate", "loss", "eval_loss", "test_loss"):
                _msg.append(f"{k}: {logs[k]:.2e}")
            elif k in ("train_samples_per_second", "train_runtime"):
                _msg.append(f"{k}: {logs[k]:.2f}")
            else:
                _msg.append(f"{k}: {logs[k]}")
        msg = " | ".join(_msg)
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        setattr(self, "epoch_tic", time.time())

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        tic = getattr(self, "epoch_tic")
        self.logger.info(f"Epoch {round(state.epoch)} finished. Cost {(time.time() - tic):.2f} s")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        setattr(self, "train_tic", time.time())

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        tic = getattr(self, "train_tic")
        self.logger.info(f"Training finished. Cost {(time.time() - tic):.2f} s")


def test_model(model: Union[LatticeBart, DP, DDP], tokenizer: BartTokenizer, dataloader: DataLoader,
               min_len: int, max_len: int, num_beams: int, logger: Logger, save_to: str = None) -> dict:
    """ Calculate BLEU, CER, WER """
    _state = model.training
    model.eval()
    device = model.device if isinstance(model, LatticeBart) else model.module.device

    refers, hypos = [], []
    keys = []

    for inputs in dataloader:  # type: dict
        inputs.pop("decoder_input_ids")
        inputs.pop("decoder_attention_mask")

        key = inputs.pop("key", None)
        if key is not None:
            keys.extend(key)

        labels = inputs.pop("labels")
        batch_refers = tokenizer.batch_decode(labels, skip_special_tokens=True)
        refers.extend(batch_refers)

        inputs = dict_to_device(inputs, device)
        if isinstance(model, DP):
            outputs = model.module.generate(**inputs, max_length=max_len, min_length=min_len, num_beams=num_beams)
        else:
            outputs = model.generate(**inputs, max_length=max_len, min_length=min_len, num_beams=num_beams)

        batch_hypos = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        hypos.extend(batch_hypos)

    word_refers = [s.split() for s in refers]
    word_hypos = [s.split() for s in hypos]
    bleu = BLEU(word_refers, word_hypos)
    wer = WER(word_refers, word_hypos, is_wer=True, logger=logger)

    if save_to is not None:
        with open(save_to, "w", encoding="utf8") as f:
            if len(keys) != 0:
                for key, hypo in zip(keys, word_hypos):
                    f.write(f"{key}\t{' '.join(hypo)}\n")
            else:
                for hypo in word_hypos:
                    f.write(f"{' '.join(hypo)}\n")

    # char_refers = [list(s) for s in refers]
    # char_hypos = [list(s) for s in hypos]
    # cer = WER(char_refers, char_hypos, is_wer=False, logger=logger)

    if _state:
        model.train()

    if logger:
        logger.info(f"Eval | BLEU: {bleu:.3f}")
    else:
        print(f"Eval | BLEU: {bleu:.3f}")

    return {"bleu": bleu, "wer": wer}


class LatticeBARTTrainer(Seq2SeqTrainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module] = None,
            args: TrainingArguments = None,
            train_collator: Optional[DataCollator] = None,
            test_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):
        super().__init__(model, args, test_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers)
        self.train_collator = train_collator
        self.test_collator = test_collator

        # Remove PrinterCallback
        self.callback_handler.remove_callback(PrinterCallback)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.train_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
