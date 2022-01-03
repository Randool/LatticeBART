import argparse
import logging

from dataset.dataset import LatticeTextDataset, DataProcessCfg
from utils.args import *


def parse_args_and_setup_hyper_param() -> Namespace:
    psr = argparse.ArgumentParser()
    add_args_for_data_path(psr)
    add_args_for_data_process(psr)
    args = psr.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    args = parse_args_and_setup_hyper_param()

    ######## Dataset ########
    process_cfg = DataProcessCfg(
        min_nodes=args.min_nodes, max_nodes=args.max_nodes, lower_case=True, ac_scale=args.ac_scale,
        lm_scale=args.lm_scale, use_marginal=args.use_marginal, use_min_depth=args.use_min_depth,
        use_max_depth=args.use_max_depth, use_forward_backward=args.use_forward_backward,
        use_lattice_mask=args.use_lattice_mask, homophone_replace=args.homophone_replace
    )

    if (args.train_lat is not None) or (args.train_text is not None):
        cache_train = join(args.cache_dir, "cache_train")
        trainset = LatticeTextDataset(lattice_pattern=args.train_lat, text_pattern=args.train_text,
                                      cache_dir=cache_train, process_cfg=process_cfg)
        print(trainset)

    if (args.eval_lats is not None) or (args.eval_text is not None):
        cache_dev = join(args.cache_dir, "cache_dev")
        devset = LatticeTextDataset(lattice_pattern=args.eval_lat, text_pattern=args.eval_text,
                                    cache_dir=cache_dev, process_cfg=process_cfg)
        print(devset)

    if (args.test_lat is not None) and (args.test_text is not None):
        cache_test = join(args.cache_dir, "cache_test")
        testset = LatticeTextDataset(lattice_pattern=args.test_lat, text_pattern=args.test_text,
                                     cache_dir=cache_test, process_cfg=process_cfg)
        print(testset)


if __name__ == '__main__':
    main()
