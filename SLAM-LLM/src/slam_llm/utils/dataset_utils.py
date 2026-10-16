# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import Sampler

import logging
logger = logging.getLogger(__name__)

class MixedWeightedSampler(Sampler):
    """70% weighted by cluster sampling_weight, 30% uniform over the full set."""
    def __init__(self, weights, num_samples, mix_ratio=0.7, generator=None):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.mix_ratio = mix_ratio
        self.generator = generator

    def __iter__(self):
        n_weighted = int(self.num_samples * self.mix_ratio)
        n_uniform = self.num_samples - n_weighted

        ####
        print("Debugging\n")
        print("sum:", self.weights.sum())
        print("min:", self.weights.min())
        print("max:", self.weights.max())
        print("nonzero:", (self.weights > 0).sum())
        print("shape:", self.weights.shape)
        print(torch.isnan(self.weights).any())
        print(torch.isinf(self.weights).any())
        ####

        weighted_idx = torch.multinomial(
            self.weights, n_weighted, replacement=True, generator=self.generator
        )
        uniform_idx = torch.randint(
            high=len(self.weights), size=(n_uniform,), generator=self.generator
        )
        idx = torch.cat([weighted_idx, uniform_idx])
        idx = idx[torch.randperm(len(idx), generator=self.generator)]
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples


def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        logger.info(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:

    return get_custom_dataset(
        dataset_config,
        tokenizer,
        split,
    )
