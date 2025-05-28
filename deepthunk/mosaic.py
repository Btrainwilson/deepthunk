import torch
from typing import Callable, Optional, Dict, Tuple, Any, List

class Mosaic:
    def __init__(self):
        self.operations: Dict[Tuple[slice, ...], Callable] = {}
        self.named_slices: Dict[str, Tuple[slice, ...]] = {}
        self._auto_name_counter: int = 0

    def __setitem__(self, idx: Any, operation: Callable):
        if isinstance(idx, str):
            slices = self.named_slices[idx]
        elif isinstance(idx, tuple):
            slices = idx
        else:
            slices = (idx,)
        self.operations[slices] = operation

    def __call__(self, tensor: torch.Tensor, subspace: Optional[str] = None) -> torch.Tensor:
        result = tensor

        if subspace:
            if subspace not in self.named_slices:
                raise ValueError(f"Unknown subspace '{subspace}'")
            slices = self.named_slices[subspace]
            operation = self.operations.get(slices)
            if operation is None:
                raise ValueError(f"No operation defined for subspace '{subspace}'")
            result[slices] = operation(result[slices])
            return result

        for slices, operation in self.operations.items():
            result[slices] = operation(result[slices])
        return result

    def add_named_slice(self, name: Optional[str], slices: Tuple[slice, ...]):
        if name is None:
            name = f"slice_{self._auto_name_counter}"
            self._auto_name_counter += 1
        self.named_slices[name] = slices

    def print_layout(self):
        for name, slices in self.named_slices.items():
            slice_str = ', '.join(f"{s.start}:{s.stop}" for s in slices)
            print(f"{name:15} -> [{slice_str}]")

    def names(self) -> List[str]:
        return list(self.named_slices.keys())

    def visualize(self, shape: Tuple[int, ...]):
        if len(shape) == 1:
            line = ["." for _ in range(shape[0])]
            for name, slices in self.named_slices.items():
                s = slices[0]
                for i in range(s.start, s.stop):
                    line[i] = name[0].upper()
            print("1D Layout:")
            print("".join(line))

        elif len(shape) == 2:
            grid = [["." for _ in range(shape[1])] for _ in range(shape[0])]
            for name, slices in self.named_slices.items():
                s0, s1 = slices[:2]
                for i in range(s0.start, s0.stop):
                    for j in range(s1.start, s1.stop):
                        grid[i][j] = name[0].upper()
            print("2D Layout:")
            for row in grid:
                print(" ".join(row))
        else:
            print("Visualization only supported for 1D or 2D tensors.")

# Example usage:
if __name__ == "__main__":
    mspace = Mosaic()
    mspace.add_named_slice("row1", (slice(0, 2),))
    mspace.add_named_slice("row2", (slice(5, 8),))
    mspace.visualize((10,))

    mspace2 = Mosaic()
    mspace2.add_named_slice("top", (slice(0, 2), slice(0, 3)))
    mspace2.add_named_slice("bottom", (slice(3, 5), slice(2, 5)))
    mspace2.visualize((6, 6))
