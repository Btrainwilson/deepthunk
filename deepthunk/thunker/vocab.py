import torch
import torch.nn.functional as F
from typing import Any, List, Optional, Union

from .thunker import Thunker, SpaceThunker

class VocabDecoder(Thunker):

    def __init__(self, choices: List[Any], temp: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)

        if temp is not None and temp <= 0:
            raise ValueError("Temperature must be positive.")

        self.choices = choices
        self.temp = temp
        self.width = len(choices)

    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x / self.temp, dim=-1)

    def idx(self, x:torch.Tensor):
        if self.temp is None:
            idx = torch.argmax(x, dim=-1).to(torch.long)
        else:
            probs = self._probs(x)
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.long)

        return idx

    def decode(self, x: torch.Tensor) -> Any:
        idx = self.idx(x)

        if x.ndim == 1:
            return self.choices[idx.item()]
        return [self.choices[i] for i in idx]

    def encode(self, value: Union[Any, List[Any]], device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or "cpu"
        values = [value] if not isinstance(value, list) else value
        logits = torch.zeros((len(values), self.width), dtype=torch.float32, device=device)
        for i, v in enumerate(values):
            if v not in self.choices:
                raise ValueError(f"Value '{v}' not in vocabulary {self.choices}")
            logits[i, self.choices.index(v)] = 1
        return logits if isinstance(value, list) else logits[0]

    def __repr__(self):
        return f"{self.__class__.__name__}(width={self.width}, choices={self.choices}, temp={self.temp})"

    def __len__(self):
        return len(self.choices)

class OneHotIntDecoder(Thunker):
    def __init__(self, size: int, temp: float, offset: int = 0, **kwargs):
        super().__init__(**kwargs)
        if temp <= 0:
            raise ValueError("Temperature must be positive.")
        self.size = size
        self.temp = temp
        self.offset = int(offset)

    def __len__(self):
        return self.size

    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x / self.temp, dim=-1)

    def decode(self, x: torch.Tensor) -> Union[int, List[int]]:
        """
        Returns sampled integer(s) from logits x after softmax/temperature.
        Handles both batched and non-batched.
        """
        probs = self._probs(x)
        # Unbatched: [size], Batched: [batch, size]
        if x.dim() == 1:
            return torch.multinomial(probs, num_samples=1).item() + self.offset
        else:
            return torch.multinomial(probs, num_samples=1).squeeze(-1).tolist() + self.offset

    def encode(self, value: Union[int, List[int]]) -> torch.Tensor:
        """
        Encode a single int or a list of ints as a one-hot tensor (batched if list).
        """
        one_hot = torch.zeros(self.size, dtype=torch.float32)
        one_hot[value - self.offset] = 1.0
        return one_hot

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, temp={self.temp}, subspace={self.subspace.tolist()})"
