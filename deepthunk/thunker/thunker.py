from abc import ABC, abstractmethod
from typing import Any, Union, List, Optional
import torch

class Thunker:
    """
        Abstract base class for logit/token functions with explicit torch index tensor.
    """

    @abstractmethod
    def decode(self, x: torch.Tensor) -> Any:
        ...

    @abstractmethod
    def encode(self, value: Union[Any, List[Any]], device: Optional[torch.device] = None) -> torch.Tensor:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class LogitThunker(Thunker):
    def __init__(self, size:int):
        self.size = size

    def __len__(self):
        return self.size

class SpaceThunker(Thunker):
    def __init__(self, subspace: Union[torch.Tensor, list, tuple], device='cpu'):
        self.device = torch.device(device)
        self._subspace = self.normalize(subspace)

    @property
    def subspace(self):
        return self._subspace

    @subspace.setter
    def subspace(self, val):
        self._subspace = self.normalize(val)

    def normalize(self, value: Any) -> torch.Tensor:
        # Accept torch.Tensor, list, tuple, numpy array, etc.
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.as_tensor(value, device=self.device)

    def decode(self, x: torch.Tensor) -> Any:
        """
        x: input tensor (1D or batched), returns x indexed at self.subspace
        """
        idx = self._subspace
        if x.ndim == 1:
            return x.index_select(0, idx)
        else:
            return x.index_select(-1, idx)

    def encode(self, value: Union[Any, List[Any]]) -> torch.Tensor:
        raise NotImplemented

    def to(self, **kwargs):
        self._subspace = self._subspace.to(**kwargs)
        self.device = self._subspace.device
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}(subspace={self._subspace.tolist()}, device={self.device})"

class LinearLogitEmbedding(torch.nn.Module, Thunker):

    def __init__(self, inspace: torch.Tensor, outspace: torch.Tensor, device: Optional[torch.device] = None):

        torch.nn.Module.__init__(self)

        self.outspace = outspace.to(self.device).long()

        self.inshape = tuple(self.space.shape)
        self.outshape = tuple(self.outspace.shape)

        self.in_features = self.space.numel()
        self.out_features = self.outspace.numel()
        self.linear = torch.nn.Linear(self.in_features, self.out_features, bias=True).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Accepts x of shape [batch, *inshape] or [*inshape]
        orig_shape = x.shape
        if x.ndim == len(self.inshape):
            x = x.unsqueeze(0)  # Add batch dim if needed

        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)  # [batch, in_features]
        y_flat = self.linear(x_flat)   # [batch, out_features]
        y = y_flat.reshape(batch, *self.outshape)
        return y if orig_shape == x.shape else y.squeeze(0)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        # Accepts y of shape [batch, *outshape] or [*outshape]
        orig_shape = y.shape
        if y.ndim == len(self.outshape):
            y = y.unsqueeze(0)

        batch = y.shape[0]
        y_flat = y.reshape(batch, -1)      # [batch, out_features]
        weight = self.linear.weight        # [out_features, in_features]
        bias = self.linear.bias            # [out_features]
        y_flat = y_flat - bias             # [batch, out_features]
        W_pinv = torch.linalg.pinv(weight) # [in_features, out_features]
        x_flat = torch.matmul(y_flat, W_pinv.T)  # [batch, in_features]
        x = x_flat.reshape(batch, *self.inshape)
        return x if orig_shape == y.shape else x.squeeze(0)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in={self.in_features}, out={self.out_features}, "
                f"inshape={self.inshape}, outshape={self.outshape}, device={self.device})")

