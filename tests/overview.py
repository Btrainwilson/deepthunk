import deepthunk
from deepthunk import LogitSpace


# We begin by initializing a logitspace.
# This keeps track of logits for us.

lspace = LogitSpace()

# Think of it like a memory request to the logit space.
# You query a size and you get back a view into the logit space.
lspace.STATE(10)
lspace.REWARD(10)

# For spaces we know the size of a priori, we can assign it like this.
# Otherwise, we can build out a thunker first and then allocate.

#Now, you can use that allocated space in your thunkers.

x = torch.zeros()


