"""Output protocol ab_constrained_v1: one greedily selected A/B token.

Mask a clone, preserving the original full-vocabulary logits for diagnostics.
No prompt changes, post-hoc answer extraction, or label-dependent logic.
"""
class ABOnlyLogitsProcessor:
    def __init__(self, codes):
        if set(codes) != {'A','B'} or len(set(codes.values())) != 2:
            raise ValueError('Require distinct verified A and B token IDs')
        self.ids = [codes['A'], codes['B']]

    def __call__(self, input_ids, scores):
        masked = scores.new_full(scores.shape, float('-inf'))
        masked[:, self.ids] = scores[:, self.ids]
        return masked
