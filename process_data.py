import argparse
import logging
from utils.args import *
from utils.dataset import LatticeTextDataset


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
    data_process = {
        "lower_case": True,
        "sort_by_len": args.sort_by_len == 1,
        "lm_scale": args.lm_scale,
        "ac_scale": args.ac_scale,
        "use_marginal": args.use_marginal == 1,
        "use_min_depth": args.use_min_depth == 1,
        "use_max_depth": args.use_max_depth == 1,
        "use_forward_backward": args.use_forward_backward == 1,
        "use_lattice_mask": args.use_lattice_mask == 1,
        "binary_mask": args.use_binary_mask == 1
    }

    if (args.train_lats is not None) or (args.train_text is not None):
        cache_train = join(args.cache_dir, "cache_train")
        trainset = LatticeTextDataset(lattice_pattern=args.train_lats, text_pattern=args.train_text,
                                      cache_dir=cache_train, min_nodes=args.min_nodes,
                                      max_nodes=args.max_nodes, **data_process)
        print(trainset)

    if (args.eval_lats is not None) or (args.eval_text is not None):
        cache_dev = join(args.cache_dir, "cache_dev")
        devset = LatticeTextDataset(lattice_pattern=args.eval_lats, text_pattern=args.eval_text,
                                    cache_dir=cache_dev, min_nodes=0, max_nodes=args.max_nodes,
                                    **data_process)
        print(devset)

    if (args.test_lats is not None) and (args.test_text is not None):
        cache_test = join(args.cache_dir, "cache_test")
        testset = LatticeTextDataset(lattice_pattern=args.test_lats, text_pattern=args.test_text,
                                     cache_dir=cache_test, min_nodes=0, max_nodes=args.max_nodes,
                                     **data_process)
        print(testset)



if __name__ == '__main__':
    main()
