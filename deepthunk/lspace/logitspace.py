import torch
import shutil

from collections import OrderedDict
from typing import Callable, Dict, Union, Iterable, List, Tuple, Any
from tabulate import tabulate
from colorama import Fore, Style, init
init(autoreset=True)

from collections import defaultdict

from .. import SpaceCache

class LogitSpace:
    def __init__(self, device='cpu', cache=True):
        self.slice_space = SpaceCache(device=device, cache=cache)
        self.alias_map: Dict[str, Union[slice, torch.Tensor]] = {}   # alias → index
        self.decoders: Dict[str, Callable] = OrderedDict()           # alias → processor

    def add(self, alias: str, idx: Union[slice, torch.Tensor, Iterable[int]], processor: Callable):
        if not callable(processor):
            raise TypeError(f"Processor for alias '{alias}' must be callable.")

        self.alias_map[alias] = idx
        self.decoders[alias] = processor
        self.slice_space[idx] = processor  # forward to SliceSpace

    def to(self, device: Union[str, torch.device]):
        self.slice_space.to(device)

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_outputs = self.slice_space(x)
        return {alias: raw_outputs[self.slice_space._key(idx)] for alias, idx in self.alias_map.items()}

    def pretty_print(self):
        rows = []
        for alias, idx in self.alias_map.items():
            if isinstance(idx, slice):
                idx_str = f"{idx.start}:{idx.stop}:{idx.step or 1}"
            else:
                idx_tensor = torch.tensor(idx) if not isinstance(idx, torch.Tensor) else idx
                idx_str = str(idx_tensor.tolist())

            fn = self.decoders.get(alias, None)
            fn_name = getattr(fn, "__name__", type(fn).__name__) if fn else "None"
            rows.append((alias, idx_str, fn_name))

        print(Fore.CYAN + Style.BRIGHT + "\nRegistered LogitSpace Entries:\n")
        print(tabulate(rows, headers=["Alias", "Indices", "Processor"], tablefmt="fancy_grid"))

    def pretty_print_logits(self, x: torch.Tensor, use_probs: bool = True):
        term_width = shutil.get_terminal_size().columns
        bar_max_width = min(50, term_width - 40)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        print(Fore.CYAN + Style.BRIGHT + "\nDecoded LogitSpace Values:\n")

        for alias, idx in self.alias_map.items():
            sl = self.slice_space.normalize_index(idx, x.shape[-1])
            decoder = self.decoders[alias]
            sub_x = x.index_select(-1, sl)

            for b, logit in enumerate(sub_x):
                probs = torch.softmax(logit, dim=-1) if use_probs else logit
                decoded = decoder(logit)

                print(f"{Fore.YELLOW}{alias} [sample {b}]: {Fore.RESET}{decoded}")
                for i, p in enumerate(probs):
                    bar_len = int(p.item() * bar_max_width)
                    bar = Fore.GREEN + "█" * bar_len + Fore.RESET
                    print(f"{i:2d}: {bar}")
                print()

        print(Style.DIM + "-" * term_width)

