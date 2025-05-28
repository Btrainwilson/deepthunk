import torch
import torch.nn.functional as F

from typing import List, Any, Callable, Optional
from typing import List, Any


class VectorFn:

    def __init__(self, width: int):
        self.width = width

    def decode(self, probs: torch.Tensor):
        raise NotImplementedError("Subclasses must implement the decode method.")

    def __len__(self):
        return self.width

    def __call__(self, probs: torch.Tensor):
        return self.decode(x)


class TokenChoice:

    def __init__(self, choices: List[Any]):
        self.choices = choices

    def __getitem__(self, idx_slice):
        if not isinstance(idx_slice, slice):
            raise ValueError("Indices must be a slice, e.g., processor[lb:ub] = fn")
        return self.choices[idx_slice]

    def __setitem__(self, idx_slice, objs):
        if not isinstance(idx_slice, slice):
            raise ValueError("Indices must be a slice, e.g., processor[lb:ub] = fn")
        self.choices[idx_slice] = objs

    def __len__(self):
        return len(self.choices)

    def __call__(self, pdist: torch.Tensor):
        index = pdist.argmax(dim=-1)
        return [self.choices[i] for i in index.view(-1)] if index.ndim > 0 else self.choices[index.item()]



class TokenSamplerSMax(TokenDecoder):

    def __init__(self, temp: float, logit_width: int):
        super().__init__(logit_width)
        if temp <= 0:
            raise ValueError("Temperature must be greater than 0.")
        self.temp = temp

    def probs(self, x):
        scaled_logits = x / self.temp
        return F.softmax(scaled_logits, dim=-1)

    def decode(self, x: torch.Tensor):

        probs = self.probs(x)

        if x.ndim == 1:
            index = torch.multinomial(probs, num_samples=1)
            return index.item()
        else:
            index = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return index

class TokenSampler(TokenDecoder):

    def __init__(self, temp: float, logit_width: int):
        super().__init__(logit_width)
        if temp <= 0:
            raise ValueError("Temperature must be greater than 0.")
        self.temp = temp

    def decode(self, x: torch.Tensor):
        scaled_logits = x / self.temp
        probs = F.softmax(scaled_logits, dim=-1)

        if x.ndim == 1:
            index = torch.multinomial(probs, num_samples=1)
            return index.item()
        else:
            index = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return index

class TokenDecoder:
    def __init__(self, logit_width: int, prep_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.prep_fn = prep_fn or (lambda x: x)  # Identity by default

    def __len__(self):
        return logit_width 

    def __getitem__(self, idx_slice):
        return self.choices[idx_slice]

    def __setitem__(self, idx_slice, objs):
        self.choices[idx_slice] = objs

    def decode(self, probs: torch.Tensor) -> List[Any]:
        """Override this in subclasses to change decoding behavior."""
        index = probs.argmax(dim=-1)
        return [self.choices[i] for i in index.view(-1)] if index.ndim > 0 else self.choices[index.item()]

    def __call__(self, logits: torch.Tensor) -> Any:
        prepped = self.prep_fn(logits)
        return self.decode(prepped)
