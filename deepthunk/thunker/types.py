import torch
from .thunker import SpaceThunker
from .simplex import greedy_fill_left
from typing import Union, List

def randSimplex(mean, dj):
    """
        Based on 
            D = sum_j dj*pj 
            1 = sum_j pj*dj/D = sum_j q_j
            -> pj = qj*D/dj
    """
    qj = torch.nn.functional.softmax(torch.randn(dj.shape), dim=-1)
    return mean / dj * qj


def randBiasedSimplex(mean, dj):
    """
        Based on 
            D = sum_j dj*pj 
            1 = sum_j pj*dj/D = sum_j q_j
            -> pj = qj*D/dj
    """
    qj = torch.nn.functional.softmax(-torch.randn(dj.shape) / dj, dim=-1)
    return mean / dj * qj


def rand_weighted_simplex(mean, dj):
    # Sample points uniformly from the simplex {x: sum_i x_i = 1, x_i >= 0}
    u = torch.sort(torch.cat([torch.zeros(1), torch.rand(len(dj)-1), torch.ones(1)]))[0]
    w = u[1:] - u[:-1]
    # Now solve sum_j d_j * p_j = D, so p_j = D * w_j / (sum_j d_j * w_j)
    denom = torch.dot(dj, w)
    pj = mean * w / denom
    return pj

class WeightedSum(SpaceThunker):
    def __init__(self, mass: torch.Tensor, encoding_strategy="fill_left", **kwargs):
        super().__init__(**kwargs)

        if tuple(mass.shape) != tuple(self.subspace.shape):
            raise ValueError(f"mass.shape {mass.shape} must match subspace.shape {self.subspace.shape}")

        self.mass = mass.float()
        self.mass = self.mass.to(self.device)
        self.encoding_strategy = encoding_strategy

    def to(self, **kwargs):
        self.mass = self.mass.to(**kwargs)

    @property
    def shape(self):
        return self.mass.shape

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        mass = self.mass
        # Check that input matches the trailing dims
        if mass.shape != x.shape[-mass.dim():]:
            raise ValueError(f"Shape mismatch: logits shape {x.shape}, mass shape {mass.shape}")
        reduction_dims = tuple(range(-mass.dim(), 0))
        return (x * mass).sum(dim=reduction_dims)

    def encode(self, value: Union[float, List[float], torch.Tensor]) -> torch.Tensor:
        mass = self.mass
        value = torch.as_tensor(value, dtype=mass.dtype, device=mass.device)

        if self.encoding_strategy == "fill_left":
            # Greedy allocation along the mass axis
            # mass: shape (..., n), value: (...,) or scalar
            orig_shape = value.shape
            mass_shape = mass.shape

            # Flatten batch dims if any
            batch_shape = value.shape if value.dim() > 0 else (1,)
            mass_ = mass.expand(batch_shape + mass_shape)
            value_ = value.expand(batch_shape)
            value_ = value_.reshape(-1)
            mass_ = mass_.reshape(len(value_), -1)  # (batch, n)
            
            # Run greedy_fill_left
            alloc, _ = greedy_fill_left(mass_, value_)
            alloc = alloc.view(batch_shape + mass_shape)
            return alloc

        # Default: simple proportional allocation
        normed = mass / (mass.sum(dim=-1, keepdim=True) + 1e-8)
        if value.shape == ():  # scalar
            return value * normed
        n_trailing = mass.dim()
        view_shape = value.shape + (1,) * n_trailing
        return value.view(view_shape) * normed

    def __repr__(self):
        return f"{self.__class__.__name__}(mass_shape={tuple(self.mass.shape)}, subspace={self.subspace.tolist()})"


class Mean(SpaceThunker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=-1)

    def encode(self, value: Union[float, List[float], torch.Tensor]) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=pmass.dtype)
        if value.shape == ():  # scalar
            return value * normed
        # Insert trailing singleton dims to value to match pmass shape for broadcasting
        n_trailing = pmass.dim()
        view_shape = value.shape + (1,) * n_trailing
        return value.view(view_shape) * normed

    def __repr__(self):
        return f"{self.__class__.__name__}(pmass_shape={tuple(self.pmass.shape)}, subspace={self.subspace.tolist()})"


#
class BitStringIntDecoder(WeightedSum):
    def __init__(self, size: int, **kwargs):
        # Bit weights: 2**0, 2**1, ..., 2**(size-1)
        pmass = torch.tensor([2 ** i for i in range(size)], dtype=torch.float32)
        super().__init__(pmass=pmass, **kwargs)
        if self.subspace.numel() != size:
            raise ValueError(f"size={size} but subspace has {subspace.numel()} elements")
        self.size = size

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given x (logits or bits), decode to integer by rounding logits to 0/1 and applying bit weights.
        Supports both 1D ([size]) and batched ([batch, size]) inputs.
        """
        # If logits: use sigmoid to get [0,1], then threshold at 0.5
        if not x.dtype.is_floating_point:
            x = x.float()
        if x.ndim == 1:
            bits = (torch.sigmoid(x) > 0.5).float()
        else:
            bits = (torch.sigmoid(x) > 0.5).float()
        return super().decode(bits).long()

    def encode(self, value: Union[int, List[int], torch.Tensor]) -> torch.Tensor:
        """
        Encode integer(s) as bit vectors (as logits 0/1).
        Supports single int or a list/tensor of ints.
        """
        device = self.pmass.device
        # Convert value(s) to binary representation
        def int_to_bits(val):
            return torch.tensor([(val >> i) & 1 for i in range(self.size)], dtype=torch.float32, device=device)
        if isinstance(value, int):
            return int_to_bits(value)
        elif isinstance(value, (list, torch.Tensor)):
            # Batched: shape [batch, size]
            vals = torch.as_tensor(value, dtype=torch.long, device=device).flatten()
            return torch.stack([int_to_bits(int(v)) for v in vals], dim=0)
        else:
            raise TypeError("Unsupported type for encode: {}".format(type(value)))
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, subspace={self.subspace.tolist()})"


#class WeightedSum(SpaceThunker):
#    def __init__(self, pmass:torch.Tensor, **kwargs):
#        super().__init__(**kwargs)
#        if temp <= 0:
#            raise ValueError("Temperature must be positive.")
#        self.size = size
#        self.pmass = pmass.float()
#
#
#    def mean(self, probs: torch.Tensor):
#        return torch.sum(probs * self.pmass, dim=-1)
#
#    def variance(self, probs: torch.Tensor):
#        mean = self.mean(probs).unsqueeze(-1)
#        return torch.sum(((self.pmass - mean) ** 2) * probs, dim=-1)
#
#    def skewness(self, probs: torch.Tensor):
#        mean = self.mean(probs).unsqueeze(-1)
#        var = self.variance(probs).unsqueeze(-1)
#        return torch.sum(((self.pmass - mean) ** 3) * probs, dim=-1) / (var.squeeze(-1).clamp(min=1e-6) ** 1.5)
#
#    def kurtosis(self, probs: torch.Tensor):
#        mean = self.mean(probs).unsqueeze(-1)
#        var = self.variance(probs).unsqueeze(-1)
#        return torch.sum(((self.pmass - mean) ** 4) * probs, dim=-1) / (var.squeeze(-1).clamp(min=1e-6) ** 2)


# Floats


# Integers
class OneHotIntDecoder(SpaceThunker):
    def __init__(self, size: int, temp: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if temp <= 0:
            raise ValueError("Temperature must be positive.")
        self.size = size
        self.temp = temp

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
            return torch.multinomial(probs, num_samples=1).item()
        else:
            return torch.multinomial(probs, num_samples=1).squeeze(-1).tolist()

    def encode(self, value: Union[int, List[int]]) -> torch.Tensor:
        """
        Encode a single int or a list of ints as a one-hot tensor (batched if list).
        """
        one_hot = torch.zeros(self.size, dtype=torch.float32, device=self.device)
        one_hot[value] = 1.0
        return one_hot

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, temp={self.temp}, subspace={self.subspace.tolist()})"

