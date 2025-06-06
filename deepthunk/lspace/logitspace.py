import torch
from typing import Dict, Union

class LogitSpace:
    def __init__(self, device='cpu', cache=True):
        self.device = torch.device(device)
        self.cache = cache
        self.alias_map: Dict[str, slice] = {}      # alias -> slice
        self.tensor_map: Dict[str, torch.Tensor] = {}  # alias -> cached tensor (if cache=True)
        self.width = 0

    def __len__(self):
        return self.width

    def add(self, alias: str, size: int):
        if alias in self.alias_map:
            raise ValueError(f"{alias} already defined in LogitSpace.")
        sl = slice(self.width, self.width + size)
        self.alias_map[alias] = sl
        if self.cache:
            self.tensor_map[alias] = torch.arange(sl.start, sl.stop, device=self.device)
        self.width += size

    def __setattr__(self, name, value):
        if name in {"device", "cache", "alias_map", "tensor_map", "width"}:
            super().__setattr__(name, value)
        elif isinstance(value, int):
            self.add(name, value)
        else:
            raise TypeError("Only integer allocation is supported.")

    def __getattr__(self, name) -> Union[slice, torch.Tensor]:
        if self.cache and name in self.tensor_map:
            return self.tensor_map[name]
        elif name in self.alias_map:
            return self.alias_map[name]
        raise AttributeError(f"'LogitSpace' has no attribute '{name}'")

    def __getitem__(self, name) -> Union[slice, torch.Tensor]:
        return self.__getattr__(name)

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def pretty_print(self):
        from tabulate import tabulate
        rows = []
        for alias, idx in self.alias_map.items():
            idx_str = f"{idx.start}:{idx.stop}"
            tensor_str = str(self.tensor_map[alias].tolist()) if self.cache else "-"
            rows.append((alias, idx_str, tensor_str))
        print("\nRegistered LogitSpace Entries:")
        print(tabulate(rows, headers=["Alias", "Slice", "Tensor Indices"], tablefmt="fancy_grid"))

    def slice_view(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if self.cache and name in self.tensor_map:
            return x.index_select(-1, self.tensor_map[name])
        else:
            sl = self.alias_map[name]
            return x[..., sl]

# Usage Example:

if __name__ == "__main__":
    lspace = LogitSpace(device='cpu', cache=True)
    lspace.STATE = 10
    lspace.REWARD = 5

    print("STATE indices:", lspace.STATE)    # tensor([0, 1, ..., 9])
    print("REWARD indices:", lspace.REWARD)  # tensor([10, 11, ..., 14])

    x = torch.arange(lspace.width)
    print("STATE view:", lspace.slice_view(x, "STATE"))
    print("REWARD view:", lspace.slice_view(x, "REWARD"))

    lspace.pretty_print()

