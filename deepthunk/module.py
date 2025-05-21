
class LogitSpace:
    def __init__(self):
        self.processors = []

    def __setitem__(self, idx_slice, processor_fn):
        if not isinstance(idx_slice, slice):
            raise ValueError("Indices must be a slice, e.g., processor[lb:ub] = fn")
        self.processors.append((idx_slice, processor_fn))

    def __call__(self, logits):
        results = []
        for sl, fn in self.processors:
            sub_logits = logits[:, sl]
            results.append(fn(sub_logits))
        return results
