import torch
import torch.nn.functional as F

from deepthunk import LogitSpace, TokenChoice, ActionSpace
from deepthunk import TokenSampler


lspace = LogitSpace()

data = ["HI", "YE", "PHI"]
basic_toks = TokenChoice(["START", "STOP"])

lspace.add(lambda x : basic_toks(F.softmax(x, dim=-1)), width=len(basic_toks))
lspace.add(TokenChoice(data))

print(lspace.names())

x = torch.randn(10)

y = lspace(x, lspace.names()[0])

print(y)

y = lspace(x)

print(y)

