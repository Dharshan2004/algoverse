"""A/B decoding constraints.

`ab_constrained_v1` masks every step to A/B and stops after one token.
`ab_first_token_v1` masks only the first generated token, then leaves later
tokens unconstrained so a trailing rationale can finish.
"""
NATIVE_V1 = "native_v1"
AB_CONSTRAINED_V1 = "ab_constrained_v1"
AB_FIRST_TOKEN_V1 = "ab_first_token_v1"
OUTPUT_PROTOCOLS = (NATIVE_V1, AB_CONSTRAINED_V1, AB_FIRST_TOKEN_V1)

NATIVE_GENERATION_BUDGET = 4
ANSWER_ONLY_GENERATION_BUDGET = 1
# Ceiling for text after the forced A/B token. Decoding still stops at EOS.
FIRST_TOKEN_REASONING_BUDGET = 256


def output_protocol_settings(output_protocol, max_new_tokens=NATIVE_GENERATION_BUDGET):
    """Resolve budget and how strongly A/B is enforced. Native keeps the caller budget."""
    if output_protocol not in OUTPUT_PROTOCOLS:
        raise ValueError("Unknown output protocol")
    if output_protocol == AB_CONSTRAINED_V1:
        return {"max_new_tokens": ANSWER_ONLY_GENERATION_BUDGET, "constrain": "all",
                "answer_normalization": "native"}
    if output_protocol == AB_FIRST_TOKEN_V1:
        return {"max_new_tokens": FIRST_TOKEN_REASONING_BUDGET, "constrain": "first",
                "answer_normalization": "first_ab_token_v1"}
    return {"max_new_tokens": max_new_tokens, "constrain": "none",
            "answer_normalization": "native"}


class ABOnlyLogitsProcessor:
    """Output protocol ab_constrained_v1: one greedily selected A/B token.

    Mask a clone, preserving the original full-vocabulary logits for diagnostics.
    No prompt changes, post-hoc answer extraction, or label-dependent logic.
    """
    def __init__(self, codes):
        if set(codes) != {'A', 'B'} or len(set(codes.values())) != 2:
            raise ValueError('Require distinct verified A and B token IDs')
        self.ids = [codes['A'], codes['B']]

    def __call__(self, input_ids, scores):
        masked = scores.new_full(scores.shape, float('-inf'))
        masked[:, self.ids] = scores[:, self.ids]
        return masked


class FirstTokenABLogitsProcessor:
    """Force only the first new token to A or B. Later steps keep the raw scores.

    `prompt_length` is the tokenized prompt length. The first decoding step still
    has that length; each later step is longer. Mask a clone so the saved
    full-vocabulary scores stay intact.
    """
    def __init__(self, codes, prompt_length):
        if set(codes) != {'A', 'B'} or len(set(codes.values())) != 2:
            raise ValueError('Require distinct verified A and B token IDs')
        if not isinstance(prompt_length, int) or isinstance(prompt_length, bool) or prompt_length < 1:
            raise ValueError('prompt_length must be a positive integer')
        self.ids = [codes['A'], codes['B']]
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] != self.prompt_length:
            return scores
        masked = scores.new_full(scores.shape, float('-inf'))
        masked[:, self.ids] = scores[:, self.ids]
        return masked
