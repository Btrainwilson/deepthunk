import torch
from deepthunk.thunker import SpaceThunker
from deepthunk.thunker import types as thunkTypes

bit32Decoder = thunkTypes.BitStringIntDecoder(32, torch.arange(32))

alph = VocabDecoder("abcde")

sCache.actions = alph


x = torch.randn(2, len(alph))
print("x: ", x)
y = alph.decode(x)
print("y: ", y)
z = alph.encode(y)
print("z: ", z)

alph = OneHotInt()

