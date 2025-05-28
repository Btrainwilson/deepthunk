import torch
from typing import List, Any, Callable, Optional

class LogitSpace:

    def __init__(self):
        self.processors: List[tuple[slice, Callable]] = []
        self.named_slices: dict[str, slice] = {}
        self.current_index: int = 0
        self._auto_name_counter: int = 0

    def __setitem__(self, idx, processor_fn: Callable):
        if isinstance(idx, str):
            sl = self.named_slices[idx]
        elif isinstance(idx, slice):
            sl = idx
        else:
            raise ValueError("Index must be a slice or a registered name")
        self.processors.append((sl, processor_fn))

    def __call__(self, logits: torch.Tensor, subspace: Optional[str] = None):
        if subspace is not None:
            if subspace not in self.named_slices:
                raise ValueError(f"Unknown subspace '{subspace}'")
            sl = self.named_slices[subspace]

            # Find the associated function
            # I know it's not good but this works for now.
            for s, fn in self.processors:
                if s == sl:
                    sub_logits = logits[:, sl] if logits.ndim > 1 else logits[sl]
                    return [fn(sub_logits)]
            raise ValueError(f"No processor found for subspace '{subspace}'")

        # Default: apply all processors
        results = []
        for sl, fn in self.processors:
            sub_logits = logits[:, sl] if logits.ndim > 1 else logits[sl]
            results.append(fn(sub_logits))
        return results

    def add(self, fn: Callable, name: Optional[str] = None, width: Optional[int] = None):
        if width is None:
            if hasattr(fn, '__len__'):
                width = len(fn)
            else:
                raise ValueError("Width must be specified if fn has no __len__")

        if name is None:
            name = f"field_{self._auto_name_counter}"
            self._auto_name_counter += 1

        sl = slice(self.current_index, self.current_index + width)
        self.named_slices[name] = sl
        self.processors.append((sl, fn))
        self.current_index += width

    def __getattr__(self, name):
        if name in self.named_slices:
            return self.named_slices[name]
        raise AttributeError(f"'LogitSpace' object has no attribute '{name}'")

    def print_layout(self):
        for name, sl in self.named_slices.items():
            print(f"{name:10} -> {sl.start}:{sl.stop}")

    def names(self):
        return list(self.named_slices.keys())



class TokenSample(TokenChoice):

    def __init__(self, choices: List[Any], temperature: float = 1.0):
        super().__init__(choices)
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor):
        if self.temperature <= 0:
            raise ValueError("Temperature must be > 0")

        # Apply temperature scaling and softmax
        scaled_logits = logits / self.temperature
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)

        if logits.ndim == 1:
            index = torch.multinomial(probs, num_samples=1)
            return self.choices[index.item()]
        else:
            index = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return [self.choices[i] for i in index]
