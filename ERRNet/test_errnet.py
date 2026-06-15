import argparse
import sys
from os.path import join

import torch.backends.cudnn as cudnn

import data.reflect_dataset as datasets
from engine import Engine
from options.errnet.train_options import TrainOptions


EVAL_DATASETS = {
    "ceilnet_table2": {
        "dataset_name": "testdata_table2",
        "path": "testdata_CEILNET_table2",
        "save_subdir": "CEILNet_table2",
    },
    "real20": {
        "dataset_name": "testdata_real",
        "path": "real20",
        "save_subdir": "real20",
        "max_long_edge": 512,
    },
    "postcard": {
        "dataset_name": "testdata_postcard",
        "path": "postcard",
        "save_subdir": "postcard",
    },
    "objects": {
        "dataset_name": "testdata_objects",
        "path": "objects",
        "save_subdir": "objects",
    },
    "wild": {
        "dataset_name": "testdata_wild",
        "path": "wild",
        "save_subdir": "wild",
    },
    "sir2_withgt": {
        "dataset_name": "testdata_sir2",
        "path": "sir2_withgt",
        "save_subdir": "sir2_withgt",
    },
    "official_ceilnet_table2": {
        "dataset_name": "official_ceilnet_table2",
        "path": "/mnt/a/ljz/DIP/ERRNet/datasets/processed_data/testdata_CEILNET_table2",
        "save_subdir": "official_ceilnet_table2",
    },
    "official_real20": {
        "dataset_name": "official_real20",
        "path": "/mnt/a/ljz/DIP/XReflection/data/sirs/test/real20_420",
        "save_subdir": "official_real20",
        "max_long_edge": 512,
    },
    "official_sir2_objects": {
        "dataset_name": "official_sir2_objects",
        "path": "/mnt/a/ljz/DIP/XReflection/data/sirs/test/SIR2/SolidObjectDataset",
        "save_subdir": "official_sir2_objects",
    },
    "official_sir2_postcard": {
        "dataset_name": "official_sir2_postcard",
        "path": "/mnt/a/ljz/DIP/XReflection/data/sirs/test/SIR2/PostcardDataset",
        "save_subdir": "official_sir2_postcard",
    },
    "official_sir2_wild": {
        "dataset_name": "official_sir2_wild",
        "path": "/mnt/a/ljz/DIP/XReflection/data/sirs/test/SIR2/WildSceneDataset",
        "save_subdir": "official_sir2_wild",
    },
}

TEST_DATASETS = {
    "internet": {
        "path": None,
        "save_subdir": "internet",
    },
}


def parse_test_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(list(EVAL_DATASETS.keys()) + list(TEST_DATASETS.keys()) + ["custom"]),
        help="dataset to run",
    )
    parser.add_argument(
        "--data_root",
        default="./datasets/processed_data",
        help="root directory for processed datasets",
    )
    parser.add_argument(
        "--input_dir",
        default="./datasets/raw_data/CEILNet/testdata_reflection_real",
        help="input directory for test-only datasets such as internet/custom images",
    )
    parser.add_argument(
        "--result_dir",
        default="./results",
        help="directory for saved outputs",
    )
    parser.add_argument(
        "--save_subdir",
        default=None,
        help="override output subdirectory name under result_dir",
    )
    parser.add_argument(
        "--max_long_edge",
        type=int,
        default=None,
        help="resize test images so the longest edge does not exceed this value",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def build_eval_dataloader(opt, data_root, dataset_key, max_long_edge=None):
    spec = EVAL_DATASETS[dataset_key]
    dataset = datasets.CEILTestDataset(
        join(data_root, spec["path"]),
        max_long_edge=max_long_edge if max_long_edge is not None else spec.get("max_long_edge"),
    )
    dataloader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=not opt.no_pin_memory,
    )
    return spec, dataloader


def build_test_dataloader(opt, dataset_key, input_dir, max_long_edge=None):
    if dataset_key == "custom":
        dataset = datasets.RealDataset(input_dir, max_long_edge=max_long_edge)
        save_subdir = "custom"
    else:
        spec = TEST_DATASETS[dataset_key]
        dataset = datasets.RealDataset(
            input_dir if spec["path"] is None else join(input_dir, spec["path"]),
            max_long_edge=max_long_edge,
        )
        save_subdir = spec["save_subdir"]

    dataloader = datasets.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=not opt.no_pin_memory,
    )
    return save_subdir, dataloader


def main():
    cli_args = parse_test_args()
    option_parser = TrainOptions()
    option_parser.isTrain = False
    opt = option_parser.parse()
    opt.isTrain = False

    cudnn.benchmark = len(opt.gpu_ids) > 0
    opt.no_log = True
    opt.display_id = 0
    opt.verbose = False

    engine = Engine(opt)

    if cli_args.dataset in EVAL_DATASETS:
        spec, dataloader = build_eval_dataloader(
            opt,
            cli_args.data_root,
            cli_args.dataset,
            max_long_edge=cli_args.max_long_edge,
        )
        save_subdir = cli_args.save_subdir or spec["save_subdir"]
        res = engine.eval(
            dataloader,
            dataset_name=spec["dataset_name"],
            savedir=join(cli_args.result_dir, save_subdir),
        )
        print(res)
    else:
        default_save_subdir, dataloader = build_test_dataloader(
            opt,
            cli_args.dataset,
            cli_args.input_dir,
            max_long_edge=cli_args.max_long_edge,
        )
        save_subdir = cli_args.save_subdir or default_save_subdir
        engine.test(
            dataloader,
            savedir=join(cli_args.result_dir, save_subdir),
        )


if __name__ == "__main__":
    main()
