import torch
from typing import Any, Union

class SpaceCache:
    def __init__(self, device: Union[str, torch.device] = "cpu"):
        self.__dict__["device"] = torch.device(device)
        self.__dict__["_cache"] = {}

    def normalize(self, value: Any) -> torch.Tensor:
        # Accept torch.Tensor, list/tuple/numpy array, etc.
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.as_tensor(value, device=self.device)

    def __setattr__(self, name: str, value: Any):
        if name in {"device", "_cache"}:
            self.__dict__[name] = value
        else:
            self._cache[name] = self.normalize(value)

    def __getattr__(self, name: str) -> torch.Tensor:
        if name in self._cache:
            return self._cache[name]
        raise AttributeError(f"No space named '{name}' in cache.")

    def __setitem__(self, name: str, value: Any):
        self._cache[name] = self.normalize(value)

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._cache[name]

    def __contains__(self, name: str) -> bool:
        return name in self._cache

    def __delitem__(self, name: str):
        del self._cache[name]

    def to(self, device: Union[str, torch.device]):
        device = torch.device(device)
        self.device = device
        for k in self._cache:
            self._cache[k] = self._cache[k].to(device)

    def clear(self):
        self._cache.clear()

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def items(self):
        return self._cache.items()

