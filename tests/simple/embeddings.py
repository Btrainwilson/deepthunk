import torch
from deepthunk.thunker import SpaceThunker

alph = VocabDecoder("abcde")

sCache.actions = alph


x = torch.randn(2, len(alph))
print("x: ", x)
y = alph.decode(x)
print("y: ", y)
z = alph.encode(y)
print("z: ", z)

alph = OneHotInt()

