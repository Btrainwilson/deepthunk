from deepthunk import ActionSpace, FnSpace
from deepthunk.thunker import OneHotIntDecoder, VocabDecoder
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

lb = 0
actions = ["SE", "DE"]
actionThunker = VocabDecoder(actions, 1.0, torch.arange(lb, lb+len(actions)), "cpu")
lb = len(actions)

n_qs = 10
q = {}
for i in range(4):
    q[i] = OneHotIntDecoder(n_qs, 1.0, torch.arange(lb, lb + n_qs), "cpu")
    lb += n_qs

angle_vals = [2**(-i) / 320 for i in range(-5, 5)]
angleThunker = VocabDecoder(angle_vals, 1.0, torch.arange(lb, lb+10), "cpu")
lb += 10

# Compose function space
fnspace = FnSpace(device="cpu")
fnspace.add_type(
    action=actionThunker,
    q1=q[0], q2=q[1], q3=q[2], q4=q[3],
    a=angleThunker,
)

def decode_fn(action, q1, q2, q3, q4, a):
    # Returns a dataclass of the right type
    if action == "SE":
        return SE(q1, q2, a)
    elif action == "DE":
        return DE(q1, q2, q3, q4, a)
    else:
        return None

fnspace.add_fn("as_struct", decode_fn)

# Generate random logits
x = torch.randn(2, lb)

# Decode to all variables and function output
decoded = fnspace(x, return_types=True)
print("Decoded variables and output:")
for k, v in decoded.items():
    print(f"{k}: {v}")

# Only get the output structure (as_struct)
print("\nDecoded to dataclass:")
out_structs = fnspace(x, subFn="as_struct")
print(out_structs)

# Round-trip: encode a sequence back to logits
# (You could write a batch encoder using each Thunker's .encode, e.g.:)
from collections import namedtuple

# For demonstration, we'll manually create an encode row for a DE instance
de_example = DE(q1=3, q2=4, q3=6, q4=1, a=angle_vals[2])
row = torch.zeros(lb)
row[fnspace.type_indices["action"]] = actionThunker.encode("DE")
row[fnspace.type_indices["q1"]] = q[0].encode(de_example.q1)
row[fnspace.type_indices["q2"]] = q[1].encode(de_example.q2)
row[fnspace.type_indices["q3"]] = q[2].encode(de_example.q3)
row[fnspace.type_indices["q4"]] = q[3].encode(de_example.q4)
row[fnspace.type_indices["a"]] = angleThunker.encode(de_example.a)

print("\nManual encode of DE instance to logits:")
print(row)

# You can batch this for multiple instances if needed.

# Optionally: print a pretty table of the fnspace structure
print("\nFnSpace structure:")
fnspace.pretty_print()

