import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Any, Callable, Optional, Union


class TypeFn(ABC):
    """Abstract base class for token/logit functions."""

    def __init__(self, width: int):
        self.width = width

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


class LambdaWrapper(TypeFn):
    def __init__(self, fn: Callable, args: List[str]):
        super().__init__(width=0)
        self.fn = fn
        self.args = args

    def decode(self, **kwargs):
        return self.fn(**{k: kwargs[k] for k in self.args})

    def encode(self, *args, **kwargs):
        raise NotImplementedError("LambdaWrapper does not support encode().")


class TokenDecoder(TypeFn):
    def __init__(self, width: int, prep_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__(width)
        self.prep_fn = prep_fn or (lambda x: x)

    def decode(self, logits: torch.Tensor) -> Any:
        raise NotImplementedError

    def encode(self, value: Union[Any, List[Any]], device: Optional[torch.device] = None) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement encode().")


class VocabDecoder(TypeFn):
    def __init__(self, choices: List[Any], temp: float = 1.0):
        if temp <= 0:
            raise ValueError("Temperature must be positive.")
        super().__init__(width=len(choices))
        self.choices = choices
        self.temp = temp

    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x / self.temp, dim=-1)

    def decode(self, x: torch.Tensor) -> Any:
        probs = self._probs(x)
        idx = torch.multinomial(probs, num_samples=1)
        if x.ndim == 1:
            return self.choices[idx.item()]
        return [self.choices[i] for i in idx.squeeze(-1)]

    def encode(self, value: Union[Any, List[Any]], device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or "cpu"
        values = [value] if not isinstance(value, list) else value
        logits = torch.full((len(values), self.width), -1e9, dtype=torch.float32, device=device)
        for i, v in enumerate(values):
            if v not in self.choices:
                raise ValueError(f"Value '{v}' not in vocabulary {self.choices}")
            logits[i, self.choices.index(v)] = 1e9
        return logits if isinstance(value, list) else logits[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(width={self.width}, choices={self.choices})"


class TokenSampler(TokenDecoder):
    def __init__(self, temp: float, width: int):
        if temp <= 0:
            raise ValueError("Temperature must be positive.")
        super().__init__(width)
        self.temp = temp

    def decode(self, x: torch.Tensor) -> Union[int, torch.Tensor]:
        probs = F.softmax(x / self.temp, dim=-1)
        index = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return index.item() if index.ndim == 0 else index

    def encode(self, value: Union[int, List[int]], device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or "cpu"
        values = [value] if not isinstance(value, list) else value
        logits = torch.full((len(values), self.width), -1e9, dtype=torch.float32, device=device)
        for i, v in enumerate(values):
            if not (0 <= v < self.width):
                raise ValueError(f"Index {v} out of bounds for width {self.width}")
            logits[i, v] = 1e9
        return logits if isinstance(value, list) else logits[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(width={self.width}, temp={self.temp})"

