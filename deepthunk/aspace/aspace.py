import torch
from typing import Dict, List, Any, Callable, Optional, Union

from ..fspace import FnSpace, get_args
from ..thunker.vocab import VocabDecoder

import torch
from typing import Dict, List, Any, Callable, Optional, Union

from tensor_mosaic import Mosaic

class ActionSpace:
    def __init__(self, device='cpu'):
        self.device = device
        self._type_defs: Dict[str, Any] = {}              # name -> type object (Thunker-like)
        self.actions: Dict[str, Dict[str, str]] = {}      # action_name -> {param: type_name}
        self.handlers: Dict[str, Callable] = {}           # action_name -> function
        self.action_list = ["START", "STOP"]
        self.action_decoder = None
        self.fspace: Optional[FnSpace] = None             # Will be built at compile

    def add_type(self, **kwargs):
        self._type_defs.update(kwargs)

    def add_action(self, name: str, param_defs: Dict[str, str], handler: Optional[Callable] = None):
        if name in self.actions:
            raise ValueError(f"Action '{name}' already exists.")
        self.actions[name] = param_defs
        self.handlers[name] = handler
        self.action_list.append(name)

    def compile(self):
        from ..fspace import FnSpace, VocabDecoder  # Lazy import to avoid circulars
        self.action_decoder = VocabDecoder(self.action_list, temp=1.0)
        fspace = FnSpace(device=self.device)
        # Add types
        for name, typ in self._type_defs.items():
            fspace.add_type(**{name: typ})
        # Add action decoder as a type
        fspace.add_type(action=self.action_decoder)
        self.fspace = fspace

    @property
    def width(self):
        return self.fspace.width

    def __call__(self, x: torch.Tensor, subFn: Union[List[str], str, None] = None) -> List[Dict[str, Any]]:
        print(x)
        self.fspace.pretty_print()
        values = self.fspace(x)
        print(values)
        batch_size = x.shape[0] if x.ndim > 1 else 1
        results = []

        actions = values["action"]
        if batch_size == 1:
            actions = actions.unsqueeze(0)

        for b in range(batch_size):
            act = actions[b]
            if isinstance(act, torch.Tensor) and act.numel() == 1:
                act = act.item()
            if isinstance(act, torch.Tensor) and act.dtype == torch.int64:
                act = self.action_list[act]
            if act in ["START", "STOP"]:
                continue

            param_defs = self.actions[act]
            params = {}
            for pname, ptype in param_defs.items():
                val = values[ptype][b] if batch_size > 1 else values[ptype]
                params[pname] = val
            results.append({"type": act, "params": params})
        return results

    def encode(self, sequence: List[Dict[str, Any]]) -> torch.Tensor:
        rows = []
        for step in sequence:
            act = step["type"]
            pvals = step["params"]

            row = torch.zeros(self.width, dtype=torch.float32, device=self.device)
            row_slice = self.fspace.type_indices["action"]
            row[row_slice] = self.action_decoder.encode(act, device=self.device)

            for pname, pval in pvals.items():
                ptype = self.actions[act][pname]
                sl = self.fspace.type_indices[ptype]
                row[sl] = self.fspace.type_specs[ptype].encode(pval, device=self.device)
            rows.append(row)
        return torch.stack(rows, dim=0)

    def __setitem__(self, name: str, obj):
        if callable(obj):
            self.handlers[name] = obj
        else:
            self.add_type(**{name: obj})

    def __setattr__(self, name, value):
        if name in {"_type_defs", "actions", "handlers", "action_list", "action_decoder", "fspace", "device"}:
            super().__setattr__(name, value)
        else:
            self.__setitem__(name, value)

    def __getattr__(self, name):
        if self.fspace and name in self.fspace.type_specs:
            return self.fspace.type_specs[name]
        if name in self.handlers:
            return self.handlers[name]
        raise AttributeError(f"ActionSpace has no attribute '{name}'")

    def run(self, sequence: List[Dict[str, Any]]):
        for step in sequence:
            act = step["type"]
            params = step["params"]
            fn = self.handlers.get(act)
            if not fn:
                raise ValueError(f"No handler for action '{act}'")
            fn(**params)

    def print_run(self, sequence: List[Dict[str, Any]]):
        for i, step in enumerate(sequence):
            a = step["type"]
            pstr = ", ".join(f"{k}={v}" for k, v in step["params"].items())
            print(f"[{i}] {a}({pstr})")
            self.handlers[a](**step["params"])

