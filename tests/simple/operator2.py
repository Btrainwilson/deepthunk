
from deepthunk import ActionSpace, FnSpace
from deepthunk.thunker import OneHotIntDecoder, VocabDecoder
from tensor_mosaic import Mosaic
from dataclasses import dataclass
import torch

# Helper to get function argument names
@dataclass
class DE:
    q1: int
    q2: int
    q3: int
    q4: int
    a: float

@dataclass
class SE:
    q1: int
    q2: int
    a: float

# --- Usage Example ---
# Suppose you have a set of thunker decoders, e.g. OneHotIntDecoder, VocabDecoder, etc.
actions = ["SE", "DE"]
angle_vals = [2**(-i) / 320 for i in range(-5, 5)]
actionThunker = VocabDecoder(actions, 1.0)
angleThunker = VocabDecoder(angle_vals, 1.0)
q = {i: OneHotIntDecoder(10, 1.0) for i in range(4)}

fnspace = FnSpace(dim=1)
fnspace.add_type(
    action=actionThunker,
    q1=q[0], q2=q[1], q3=q[2], q4=q[3],
    a=angleThunker,
)

def decode_fn(action, q1, q2, q3, q4, a):
    # Dummy
    return list(zip(action, q1, q2, q3, q4, a))

fnspace.add_fn("as_struct", decode_fn)

x = torch.randn(4, *fnspace.mosaic.shape)
decoded = fnspace(x, return_types=True)
print(decoded)

batch = [
    SE(q1=1, q2=2, a=angle_vals[0]),
    DE(q1=3, q2=4, q3=5, q4=6, a=angle_vals[1]),
    SE(q1=7, q2=8, a=angle_vals[2]),
]
logits = fnspace.encode(batch)
print("Encoded logits shape:", logits.shape)
