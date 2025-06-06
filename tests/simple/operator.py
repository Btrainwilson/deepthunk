from deepthunk import ActionSpace, FnSpace
from deepthunk.thunker import OneHotIntDecoder, VocabDecoder
from tensor_mosaic import Mosaic

import torch
from dataclasses import dataclass

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

actions = ["SE", "DE"]

space = Mosaic(cache=True)
space.ACTIONS = len(actions)

for i in range(4):
    space.add(f"q{i}", 10)

actionThunker = VocabDecoder(actions, 1.0, subspace=space.ACTIONS)

num_qs = 10

q = {}
for i in range(4):
    q_name = f"q{i}"
    q[i] = OneHotIntDecoder(num_qs, 1.0, subspace=space[q_name])

angle_vals = [2**(-i) / 320 for i in range(-5, 5)]
space.ANGLE = len(angle_vals)

angleThunker = VocabDecoder(angle_vals, 1.0, subspace=space.ANGLE)


# Compose function space
fnspace = FnSpace(device="cpu")
fnspace.add_type(
    action=actionThunker,
    q1=q[0], q2=q[1], q3=q[2], q4=q[3],
    a=angleThunker,
)

def decode_fn(action, q1, q2, q3, q4, a):
    # Returns a dataclass of the right type
    dec = []
    for i in range(len(action)):
        if action[i] == "SE":
            dec.append(SE(q1[i], q2[i], a[i]))
        elif action[i] == "DE":
            dec.append(DE(q1[i], q2[i], q3[i], q4[i], a[i]))
        else:
            return None
    return dec

fnspace.add_fn("as_struct", decode_fn)

# Generate random logits
x = torch.randn(4, *space.bin_shape)

# Decode to all variables and function output
decoded = fnspace(x, return_types=True)
print("Decoded variables and output:")
for k, v in decoded.items():
    print(f"{k}: {v}")

# Only get the output structure (as_struct)
print("\nDecoded to dataclass:")
out_structs = fnspace(x, subFn="as_struct")
print(out_structs)

# Optionally: print a pretty table of the fnspace structure
print("\nFnSpace structure:")
fnspace.pretty_print()

