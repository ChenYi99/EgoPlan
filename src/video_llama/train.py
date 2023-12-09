"""
Adapted from salesforce@LAVIS and Vision-CAIR@MiniGPT-4. Below is the original copyright:
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate


def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]



import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import sys

import src.video_llama.video_llama.tasks as tasks
from src.video_llama.video_llama.common.config import Config
from src.video_llama.video_llama.common.dist_utils import get_rank, init_distributed_mode
from src.video_llama.video_llama.common.logger import setup_logger
from src.video_llama.video_llama.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from src.video_llama.video_llama.common.registry import registry
from src.video_llama.video_llama.common.utils import now

# imports modules for registration
from src.video_llama.video_llama.datasets.builders import *
from src.video_llama.video_llama.models import *
from src.video_llama.video_llama.processors import *
from src.video_llama.video_llama.runners import *
from src.video_llama.video_llama.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)

    # datasets['webvid']['train'][0]
    # datasets
    model = task.build_model(cfg)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
