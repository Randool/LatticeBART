import glob
import json
from argparse import Namespace
from logging import Logger
from os.path import join
from typing import List

from torch.utils.data import DataLoader
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, set_seed, BartTokenizer

from dataset import DataProcessCfg, LatticeTextDataset
from dataset.dataset import CollateFn
from models.config import get_model_config, get_tokenizer
from models.lattice_bart import LatticeBart
from utils.args import ModelConstructArgs, DataArgs
from utils.trainer import LatticeBARTTrainer, LoggerCallback, test_model
from utils.utils import get_logger


def prepare_datasets(args: Namespace, logger: Logger):
    process_cfg = DataProcessCfg(
        min_nodes=args.min_nodes, max_nodes=args.max_nodes, lower_case=True, ac_scale=args.ac_scale,
        lm_scale=args.lm_scale, use_marginal=args.use_marginal, use_min_depth=args.use_min_depth,
        use_max_depth=args.use_max_depth, use_forward_backward=args.use_forward_backward,
        use_lattice_mask=args.use_lattice_mask, homophone_replace=args.homophone_replace,
        token_mask=args.token_mask, depth_disturb=args.depth_disturb,
    )
    process_cfg_for_test = DataProcessCfg(
        min_nodes=args.min_nodes, max_nodes=args.max_nodes, lower_case=True, ac_scale=args.ac_scale,
        lm_scale=args.lm_scale, use_marginal=1 if args.use_marginal else 0,
        use_min_depth=1 if args.use_min_depth else 0,
        use_max_depth=1 if args.use_max_depth else 0,
        use_forward_backward=1 if args.use_forward_backward else 0,
        use_lattice_mask=1 if args.use_lattice_mask else 0,
        homophone_replace=0, token_mask=0, depth_disturb=0,
    )

    trainset, devset, testsets = None, None, []

    if not (args.train_lat is None and args.train_lat_ark is None) and (args.train_text is not None):
        trainset = LatticeTextDataset(lattice_pattern=args.train_lat,
                                      lattice_ark_pattern=args.train_lat_ark,
                                      text_pattern=args.train_text,
                                      process_cfg=process_cfg)

    if not (args.eval_lat is None and args.eval_lat_ark is None) and (args.eval_text is not None):
        devset = LatticeTextDataset(lattice_pattern=args.eval_lat,
                                    lattice_ark_pattern=args.eval_lat_ark,
                                    text_pattern=args.eval_text,
                                    process_cfg=process_cfg_for_test)

    if len(args.test_lat) == 0 and len(args.test_lat_ark) == 0:
        if logger:
            logger.error("No test data found.")

    if len(args.test_lat) > 0 and len(args.test_lat_ark) > 0:
        raise ValueError(f"Only one of `--test_lat` and `--test_lat_ark` of can be set. "
                         f"Got: test_lat: {args.test_lat}, test_lat_ark: {args.test_lat_ark}")

    test_lat = [None] * len(args.test_lat_ark) if len(args.test_lat) == 0 else args.test_lat
    test_lat_ark = [None] * len(args.test_lat) if len(args.test_lat_ark) == 0 else args.test_lat_ark
    if len(args.test_text) == 1:
        text_patterns = [args.test_text[0]] * len(test_lat)
    else:
        assert len(args.test_text) == len(test_lat)
        text_patterns = args.test_text

    for i, (_lat, _lat_ark, text) in enumerate(zip(test_lat, test_lat_ark, text_patterns)):
        testset = LatticeTextDataset(lattice_pattern=_lat, lattice_ark_pattern=_lat_ark, text_pattern=text,
                                     process_cfg=process_cfg_for_test)
        testsets.append(testset)

    return trainset, devset, testsets


def main(_args: List[str] = None):
    # ===== Parse arguments =====
    parser = HfArgumentParser([Seq2SeqTrainingArguments, ModelConstructArgs, DataArgs])
    train_args, model_args, data_args = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(__name__, exp_dir=train_args.output_dir)

    for _log_name, _logger in logger.manager.loggerDict.items():
        if _log_name.startswith("transformers.trainer"):
            _logger.addHandler(logger.handlers[0])

    logger.info(f"==== Train Arguments ==== {train_args.to_json_string()}")
    logger.info(f"==== Model Arguments ==== {model_args.to_json_string()}")
    logger.info(f"==== Data Arguments ==== {data_args.to_json_string()}")

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get model =====
    if model_args.default_config is not None:
        config = get_model_config(model_args.default_config)
        model = LatticeBart(config)
        tokenizer = get_tokenizer()
    elif model_args.lat_bart_path is not None:
        model = LatticeBart.from_pretrained(model_args.lat_bart_path)
        tokenizer = BartTokenizer.from_pretrained(model_args.lat_bart_path)
    elif model_args.bart_path is not None:
        model = LatticeBart.from_bart(model_args.bart_path)
        tokenizer = BartTokenizer.from_pretrained(model_args.bart_path)
    else:
        raise ValueError("One of `--default_config`, `lat_bart_path` or `--bart_path` should be given.")

    # ===== Get datasets =====
    train_set, eval_set, test_sets = prepare_datasets(data_args, logger)
    train_collator = CollateFn(tokenizer, decoder_input=data_args.decoder_input, process_cfg=train_set.process_cfg)
    test_collator = CollateFn(tokenizer, decoder_input=data_args.decoder_input, process_cfg=eval_set.process_cfg)

    # ===== Callbacks =====
    callbacks = [LoggerCallback(logger)]

    # ===== Trainer =====
    if train_args.do_train:
        trainer = LatticeBARTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            train_collator=train_collator.collate_fn,
            test_collator=test_collator.collate_fn,
            train_dataset=train_set,
            eval_dataset=eval_set,
            callbacks=callbacks,
        )
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

    if train_args.do_predict:
        logger.info("*** Do prediction ***")

        max_step = max([-1] + [int(s.split("-")[-1]) for s in glob.glob(f"{train_args.output_dir}/checkpoint-*")])
        if max_step == -1:
            logger.warning("No checkpoints found. Skip testing")
            return

        with open(join(f"{train_args.output_dir}/checkpoint-{max_step}/trainer_state.json")) as f:
            ckpt = json.load(f)["best_model_checkpoint"]
            logger.info(f"Load ckpt from {ckpt}")

        model = LatticeBart.from_pretrained(ckpt).eval()
        if not train_args.no_cuda:
            model.to("cuda:0")

        for i, test_set in enumerate(test_sets):
            logger.info(f"Test {i}")
            dataloader = DataLoader(
                test_set, batch_size=train_args.per_device_eval_batch_size,
                collate_fn=test_collator.collate_fn, num_workers=train_args.dataloader_num_workers
            )
            test_model(
                model=model, dataloader=dataloader, tokenizer=tokenizer, logger=logger, min_len=2,
                max_len=train_args.generation_max_length, num_beams=train_args.generation_num_beams,
                save_to=f"{train_args.output_dir}/predict_{i}.txt"
            )


if __name__ == '__main__':
    main()
