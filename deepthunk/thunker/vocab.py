import torch
import torch.nn.functional as F
from typing import Any, List, Optional, Union

from .thunker import Thunker, SpaceThunker

from typing import List, Any, Optional, Union
import torch
import torch.nn.functional as F


class VocabDecoder(Thunker):
    """
    Maps between vocabulary values and logits.

    Args:
        choices (List[Any]):  the vocabulary.
        temp    (float|None): sampling temperature.  If None, use arg-max.
    """

    def __init__(self, choices: List[Any], temp: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)

        if temp is not None and temp <= 0:
            raise ValueError("Temperature must be positive.")

        self.choices = choices
        self.temp    = temp
        self.width   = len(choices)

    # ---------- internal helpers ---------- #

    @staticmethod
    def _probs(x: torch.Tensor, temp: float) -> torch.Tensor:
        """Softmax along last dim with temperature."""
        return F.softmax(x / temp, dim=-1)

    def idx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return indices selected from each last-dim slice of *x*.

        Handles arbitrary batch/sequence dimensions.
        """
        if self.temp is None:                       # arg-max mode (deterministic)
            return torch.argmax(x, dim=-1, keepdim=False).to(torch.long)

        # ----- temperature sampling (multinomial) -----
        probs       = self._probs(x, self.temp)     # same shape as x
        orig_shape  = probs.shape[:-1]
        flat_probs  = probs.reshape(-1, self.width) # 2-D for multinomial
        flat_idx    = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        return flat_idx.view(*orig_shape)           # back to (*batch_dims)

    # ---------- public API ---------- #

    def decode(
        self,
        x: torch.Tensor,
        *,
        one_hot: bool = False,
    ) -> Union[Any, List[Any], torch.Tensor]:
        """
        Decode *x* into vocabulary entries **or** one-hot vectors.

        * If ``one_hot`` is False (default)  →  Python values (same nesting).
        * If ``one_hot`` is True            →  float tensor with shape
                                              (*x.shape[:-1], width*).
        """
        idx = self.idx(x)  # shape: x.shape[:-1]

        if one_hot:
            # add width dim back at the end → (*batch_dims, width)
            return F.one_hot(idx, num_classes=self.width).to(torch.float32)

        # ---- map indices → choices with correct nesting ----
        def _map(vals):
            if isinstance(vals, list):
                return [_map(v) for v in vals]
            return self.choices[vals]

        return _map(idx.tolist())

    def encode(
        self,
        value: Union[Any, List[Any]],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Convert value(s) to one-hot logits (float32) with shape
        (*value.shape, width*).

        Works with scalars, lists, or arbitrarily nested Python lists
        matching the batch structure you want.
        """
        device = device or "cpu"

        def _encode(v):
            if isinstance(v, list):
                return torch.stack([_encode(item) for item in v])
            if v not in self.choices:
                raise ValueError(f"Value '{v}' not in vocabulary {self.choices}")
            oh = torch.zeros(self.width, dtype=torch.float32, device=device)
            oh[self.choices.index(v)] = 1.0
            return oh

        return _encode(value)

    # ---------- misc ---------- #

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"width={self.width}, choices={self.choices}, temp={self.temp})"
        )

    def __len__(self):
        return self.width

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
