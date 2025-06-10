from inspect import getargs
import inspect

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, Callable, List, Union, Any
from tabulate import tabulate

from .. import ThunkCache, Thunker

from tensor_mosaic import Mosaic

def get_args(fn: Callable) -> list[str]:
    try:
        return list(inspect.signature(fn).parameters.keys())
    except (ValueError, TypeError):
        # Fallback for lambdas or other callables
        return list(fn.__code__.co_varnames[:fn.__code__.co_argcount])

class IdentityThunker(Thunker):
    def encode(self, value):
        return value

    def decode(self, x):
        return x


from typing import Dict, Any, Callable, Union, List
import torch

class FnSpace:
    def __init__(self, dim=1, device='cpu', mosaic_kwargs=None):
        self.device = torch.device(device)
        self.mosaic = Mosaic(dim=dim, **(mosaic_kwargs or {}))
        self.thunks: Dict[str, Any] = {}
        self.subspaces: Dict[str, str] = {}  # name -> subspace name in mosaic
        self.fns: Dict[str, Callable] = {}
        self.eval_order: List[str] = []

    def add_type(self, **kwargs):
        for name, thunk in kwargs.items():
            # Use .width if available, else len(thunk)
            if hasattr(thunk, "width"):
                width = thunk.width
            elif hasattr(thunk, "__len__"):
                width = len(thunk)
            else:
                raise ValueError(f"Thunker '{name}' must have .width or __len__ defined")
            # Allocate a region in the mosaic
            self.mosaic.add(name, shape=width)
            self.thunks[name] = thunk
            self.subspaces[name] = name

    def add_fn(self, name: str, fn: Callable):
        if name in self.fns:
            raise ValueError(f"Function '{name}' already defined.")
        fn_args = get_args(fn)
        for arg in fn_args:
            if arg not in self.thunks:
                raise ValueError(f"Parameter '{arg}' not instantiated.")
        self.fns[name] = fn
        self.eval_order.append(name)

    def __call__(
        self,
        x: torch.Tensor,
        subFn: Union[List[str], str, None] = None,
        types: Union[List[str], str, None] = None,
        return_types: bool = True,
    ) -> Dict[str, Any]:
        # 1. Decode variables (types)
        if types is None:
            type_names = list(self.thunks)
        elif isinstance(types, str):
            type_names = [types]
        else:
            type_names = list(types)
        values = {}
        for name in type_names:
            thunk = self.thunks[name]
            region = self.mosaic[name]
            # 1D/ND support for region:
            values[name] = thunk.decode(x[..., *region])

        results = dict(values)
        # 2. Evaluate functions
        evals = [subFn] if isinstance(subFn, str) else subFn or self.eval_order
        for fname in evals:
            fn = self.fns[fname]
            args = {arg: results[arg] for arg in get_args(fn)}
            results[fname] = fn(**args)

        if return_types:
            return results
        else:
            return {fname: results[fname] for fname in evals}

    def encode(self, instances: list):
        """
        Given a list of dataclass instances (e.g., SE or DE), encode them to logits,
        placing each encoded field into the correct mosaic subspace for each sample.
        
        Returns:
            logits: [batch_size, ...mosaic.shape]
        """
        batch_size = len(instances)
        shape = (batch_size, *self.mosaic.shape)
        logits = torch.zeros(shape, device=self.device)

        for i, inst in enumerate(instances):
            # For each thunker, encode the corresponding value (if present in dataclass)
            for name, thunker in self.thunks.items():
                region = self.mosaic[name]
                # Use getattr if possible, else skip
                if hasattr(inst, name):
                    value = getattr(inst, name)
                    enc = thunker.encode(value)
                    # Write logits for this sample to correct subspace
                    logits[i][region] = enc
                else:
                    # Optionally, skip or zero-fill for this field
                    pass

        return logits

    def to(self, device):
        device = torch.device(device)
        self.device = device
        self.mosaic.device = device  # assuming mosaic supports device changes

    def __setitem__(self, name: str, thunk):
        self.add_type(**{name: thunk})

    def __setattr__(self, name, value):
        if name in {"device", "mosaic", "thunks", "subspaces", "fns", "eval_order"}:
            super().__setattr__(name, value)
        else:
            self.__setitem__(name, value)

    def __getattr__(self, name):
        if name in self.thunks:
            return self.thunks[name]
        elif name in self.fns:
            return self.fns[name]
        raise AttributeError(f"'FnSpace' object has no attribute '{name}'")

    def pretty_print(self):
        # Print variable info
        print("\nFnSpace structure:")
        for name in self.thunks:
            region = self.mosaic[name]
            print(f"{name}: thunker={type(self.thunks[name]).__name__}, mosaic region={region}")
        # Print functions
        print("\nFunctions:")
        for fname, fn in self.fns.items():
            args = list(get_args(fn))
            fn_type = type(fn).__name__ if not hasattr(fn, "__name__") else fn.__name__
            print(f"{fname}: ({', '.join(args)}) -> {fn_type}")

