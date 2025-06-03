import torch
from typing import Any, Dict, Optional, List, Union
from abc import abstractmethod

from .. import SpaceCache

from .thunker import Thunker

class ThunkCache:
    """
    Stores named Thunkers, each linked to a named slice in a SliceCache.
    Supports both explicit naming and auto-naming for unnamed thunks.
    The space/tensor is NOT stored on the thunker, only in the cache.
    """
    def __init__(self, device: Union[str, torch.device] = "cpu"):
        self.slices = SpaceCache(device=device)
        self.thunks: Dict[str, Thunker] = {}
        self._auto_counter = 0
        self._space_to_name = {}

    def _gen_name(self) -> str:
        name = f"thunk_{self._auto_counter}"
        self._auto_counter += 1
        return name

    def add(self, name: str, space: Any, thunk: Thunker):
        self.slices[name] = space
        self.thunks[name] = thunk
        self._space_to_name[self._tensor_key(self.slices[name])] = name

    def __setitem__(self, key: Any, value: Thunker):
        """
        If key is a string, treat as named; if tensor/space, autoname.
        """
        if isinstance(key, str):
            name = key
            # Must already have slices[name] set elsewhere, or set it after
            self.thunks[name] = value
        else:
            # key is a tensor or list representing the indices/space
            name = self._gen_name()
            self.slices[name] = key
            self.thunks[name] = value
            self._space_to_name[self._tensor_key(self.slices[name])] = name

    def __getitem__(self, key: Any) -> Thunker:
        if isinstance(key, str):
            return self.thunks[key]
        # else assume it's a space
        name = self._space_to_name.get(self._tensor_key(key))
        if name is None:
            raise KeyError("Thunker with this space not found.")
        return self.thunks[name]

    def _tensor_key(self, t: Any) -> str:
        return str(torch.as_tensor(t).cpu().numpy().tolist())

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str):
            return key in self.thunks
        return self._tensor_key(key) in self._space_to_name

    def __delitem__(self, key: Any):
        if isinstance(key, str):
            t = self.slices[key]
            tk = self._tensor_key(t)
            del self.thunks[key]
            del self.slices[key]
            self._space_to_name.pop(tk, None)
        else:
            tk = self._tensor_key(key)
            name = self._space_to_name.get(tk)
            if name:
                del self.thunks[name]
                del self.slices[name]
                del self._space_to_name[tk]

    def to(self, device: Union[str, torch.device]):
        self.slices.to(device)
        # Nothing to do for thunks, as they no longer own a space tensor

    def keys(self):
        return self.thunks.keys()

    def values(self):
        return self.thunks.values()

    def items(self):
        return self.thunks.items()

    def clear(self):
        self.thunks.clear()
        self.slices.clear()
        self._space_to_name.clear()
        self._auto_counter = 0
