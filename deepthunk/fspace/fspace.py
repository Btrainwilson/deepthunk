from inspect import getargs
import inspect

import torch
from torch import nn
from abc import ABC, abstractmethod
from typing import Dict, Callable, List, Union, Any
from tabulate import tabulate

from .. import ThunkCache, Thunker

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


class FnSpace:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.thunks = ThunkCache(device=self.device)
        self.type_specs = {}        # name -> thunk
        self.type_indices = {}      # name -> tensor of indices
        self.fns = {}
        self.eval_order = []
        self.width = 0

    def add_type(self, **kwargs):
        for name, spec in kwargs.items():
            if isinstance(spec, tuple):
                thunk, dim = spec
            elif isinstance(spec, int):
                thunk = IdentityThunker()
                dim = spec
            elif hasattr(spec, '__len__') or hasattr(spec, 'width'):
                thunk = spec
                dim = len(spec) if hasattr(spec, '__len__') else spec.width
            else:
                raise ValueError(f"Type for '{name}' must be int, (Thunker, dim), or a Thunker with __len__/width.")

            indices = torch.arange(self.width, self.width + dim, device=self.device)
            self.width += dim
            self.type_specs[name] = thunk
            self.type_indices[name] = indices
            self.thunks.add(name, indices, thunk)

    def add_fn(self, name: str, fn: Callable):
        if name in self.fns:
            raise ValueError(f"Function '{name}' already defined.")
        fn_args = get_args(fn)
        for arg in fn_args:
            if arg not in self.type_specs:
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
        """
        Args:
            x: input tensor
            subFn: functions to evaluate (default: all in eval_order)
            types: subset of types/variables to decode (default: all)
            return_types: if True, return all decoded variables + function results;
                        if False, return only the function results.

        Returns:
            Dict[str, Any]
        """
        # 1. Decode variables (types)
        if types is None:
            type_names = list(self.type_specs)
        elif isinstance(types, str):
            type_names = [types]
        else:
            type_names = list(types)
        values = {}
        for name in type_names:
            thunk = self.type_specs[name]
            indices = self.type_indices[name]
            values[name] = thunk.decode(x.index_select(-1, indices))

        results = dict(values)

        # 2. Evaluate functions
        evals = [subFn] if isinstance(subFn, str) else subFn or self.eval_order
        for name in evals:
            fn = self.fns[name]
            args = {arg: results[arg] for arg in get_args(fn)}
            results[name] = fn(**args)

        if return_types:
            return results
        else:
            return {name: results[name] for name in evals}

    def to(self, device):
        device = torch.device(device)
        self.device = device
        for k in self.type_indices:
            self.type_indices[k] = self.type_indices[k].to(device)
        self.thunks.to(device)

    def __setitem__(self, name: str, obj):
        if isinstance(obj, int):
            self.add_type(**{name: obj})
        elif isinstance(obj, tuple) and len(obj) == 2:
            thunk, width = obj
            if not callable(getattr(thunk, 'decode', None)):
                raise TypeError("First item of tuple must be a Thunker with .decode")
            if not isinstance(width, int):
                raise TypeError("Second item of tuple must be the width (int)")
            self.add_type(**{name: (thunk, width)})
        elif hasattr(obj, "__len__") or hasattr(obj, "width"):
            width = len(obj) if hasattr(obj, "__len__") else obj.width
            self.add_type(**{name: (obj, width)})
        elif callable(obj):
            self.add_fn(name, obj)
        else:
            raise TypeError(f"Unsupported object for FnSpace: {obj}")

    def __setattr__(self, name, value):
        if name in {"device", "thunks", "type_specs", "type_indices", "fns", "eval_order", "width"}:
            super().__setattr__(name, value)
        else:
            self.__setitem__(name, value)

    def __getattr__(self, name):
        if name in self.type_specs:
            return self.type_specs[name]
        elif name in self.fns:
            return self.fns[name]
        raise AttributeError(f"'FnSpace' object has no attribute '{name}'")

    def pretty_print(self):
        # Print type specs (variables)
        rows = []
        for name in self.type_specs:
            indices = self.type_indices[name]
            indices_str = str(indices.tolist())
            thunk = self.type_specs[name]
            thunk_str = type(thunk).__name__
            rows.append((name, indices_str, thunk_str))
        print("\nVariables:")
        print(tabulate(rows, headers=["Name", "Indices", "Thunker Type"], tablefmt="fancy_grid"))

        # Print function specs
        fn_rows = []
        for fname, fn in self.fns.items():
            args = list(get_args(fn))
            fn_type = type(fn).__name__ if not hasattr(fn, "__name__") else fn.__name__
            fn_rows.append((fname, fn_type, ", ".join(args)))
        print("\nFunctions:")
        print(tabulate(fn_rows, headers=["Name", "Type", "Args"], tablefmt="fancy_grid"))

