import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Any, Callable, Optional, Union


class LambdaWrapper(TypeFn):
    def __init__(self, fn: Callable, args: List[str]):
        super().__init__(width=0)
        self.fn = fn
        self.args = args

    def decode(self, **kwargs):
        return self.fn(**{k: kwargs[k] for k in self.args})

    def encode(self, *args, **kwargs):
        raise NotImplementedError("LambdaWrapper does not support encode().")

