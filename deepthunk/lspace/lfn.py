from abc import ABC, abstractmethod
from typing import Any, Union, List, Optional
import torch

class LogitThunker(ABC):
    """Abstract base class for logit/token functions with explicit torch index tensor."""

    def __init__(self, space: torch.Tensor, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.space = space.to(self.device).long()
        self.width = self.space.numel()

    def __len__(self):
        return self.width

    @abstractmethod
    def decode(self, x: torch.Tensor) -> Any:
        ...

    @abstractmethod
    def encode(self, value: Union[Any, List[Any]], device: Optional[torch.device] = None) -> torch.Tensor:
        ...

    def __call__(self, x: torch.Tensor) -> Any:
        return self.decode(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(width={self.width})"


