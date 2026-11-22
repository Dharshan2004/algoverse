"""Finite JSON grammar keeps both verdict choices available and forbids prose."""
import unittest
import torch
from refusal_cues.structured_output import JsonVerdictConstraint, parse_structured

class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())
    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode()

class JsonConstraintTests(unittest.TestCase):
    def test_both_verdicts_roundtrip_and_only_json_paths_are_possible(self):
        tokenizer = CharTokenizer()
        grammar = JsonVerdictConstraint(tokenizer, 2, 0)
        for code in ('A', 'B'):
            target = tokenizer.encode('{"verdict":"' + code + '"}')
            generated = []
            for token in target:
                allowed = grammar(0, torch.tensor([999, 999] + generated))
                self.assertNotIn(0, allowed)
                if len(generated) == grammar.branch_index:
                    self.assertEqual(set(allowed), {ord('A'), ord('B')})
                else:
                    self.assertEqual(allowed, [token])
                self.assertIn(token, allowed)
                generated.append(token)
            self.assertEqual(grammar(0, torch.tensor([999, 999] + generated)), [0])
            self.assertNotEqual(parse_structured(tokenizer.decode(generated), 'A_safe')['verdict'], 'INVALID')
        self.assertEqual(grammar.required_budget, len(target) + 1)

    def test_invalid_prefix_and_missing_eos_fail_closed(self):
        tokenizer = CharTokenizer()
        grammar = JsonVerdictConstraint(tokenizer, 1, 0)
        with self.assertRaises(ValueError):
            grammar(0, torch.tensor([999, ord('`')]))
        with self.assertRaises(ValueError):
            JsonVerdictConstraint(tokenizer, 1, None)

    def test_asymmetric_tokenization_is_rejected(self):
        class BadTokenizer(CharTokenizer):
            def encode(self, text, add_special_tokens=False):
                return super().encode(text) + ([32] if 'B' in text else [])
        with self.assertRaises(ValueError):
            JsonVerdictConstraint(BadTokenizer(), 1, 0)
