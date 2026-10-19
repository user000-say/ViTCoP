"""Single-forward layer capture using native module hooks, without attentions.

Pass the actual source-bound ViT block list. Completed block 9 -> blocks[8].
For a D-block CLIP, hidden_states[-2] -> output of completed block D-1.
This is a capture utility, not a model-specific architecture resolver.
"""
from .budget import integer


class LayerCapture:
    def __init__(self, blocks, *, early_completed=9, pen_completed=None):
        self.blocks = list(blocks)
        self.early = integer(early_completed, "early_completed", 1)
        self.pen = len(self.blocks)-1 if pen_completed is None else integer(pen_completed, "pen_completed", 1)
        if not self.blocks or not 1 <= self.early <= len(self.blocks) or not 1 <= self.pen <= len(self.blocks):
            raise ValueError("requested layer is outside encoder")
        self.values, self.handles = {}, []

    def __enter__(self):
        if self.handles:
            raise RuntimeError("capture already attached")
        self.values.clear()
        def hook_for(completed):
            def hook(module, args, output):
                value = output[0] if isinstance(output, (tuple, list)) else output
                if completed in self.values:
                    raise RuntimeError("unexpected repeated forward; clear capture per request")
                # Clone only the requested layers; record its cost in TTFT.
                self.values[completed] = value.detach().clone() if hasattr(value, "detach") else value.copy()
            return hook
        for completed in sorted({self.early, self.pen}):
            self.handles.append(self.blocks[completed-1].register_forward_hook(hook_for(completed)))
        return self

    def get(self):
        if not {self.early, self.pen}.issubset(self.values):
            raise RuntimeError("required layer did not execute")
        return self.values[self.early], self.values[self.pen]

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
