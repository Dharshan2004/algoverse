import unittest
import torch
from refusal_cues.ab_constraint import (
    ABOnlyLogitsProcessor, FIRST_TOKEN_REASONING_BUDGET, FirstTokenABLogitsProcessor,
    NATIVE_GENERATION_BUDGET, output_protocol_settings,
)
from refusal_cues.ab_report import quality

class ABTests(unittest.TestCase):
    def test_mask_excludes_highest_noncode_and_preserves_raw_scores(self):
        scores=torch.tensor([[9.,1.,3.,8.],[-1.,2.,1.,7.]])
        saved=scores.clone();masked=ABOnlyLogitsProcessor({'A':1,'B':2})(None,scores)
        self.assertTrue(torch.equal(saved,scores))
        self.assertEqual(masked.argmax(-1).tolist(),[2,1])
        self.assertTrue(torch.isneginf(masked[:,[0,3]]).all())
        self.assertTrue(torch.equal(masked[:,[1,2]],scores[:,[1,2]]))
    def test_invalid_token_configuration_rejected(self):
        with self.assertRaises(ValueError):ABOnlyLogitsProcessor({'A':1,'B':1})
    def test_answer_only_mask_applies_on_every_step(self):
        scores=torch.tensor([[9.,1.,4.,8.]])
        proc=ABOnlyLogitsProcessor({'A':1,'B':2})
        for length in (4,5):
            masked=proc(torch.zeros(1,length,dtype=torch.long),scores)
            self.assertEqual(masked.argmax(-1).tolist(),[2])
            self.assertTrue(torch.isneginf(masked[:,[0,3]]).all())
    def test_first_token_mask_allows_later_noncode_tokens(self):
        scores=torch.tensor([[9.,1.,4.,8.],[-1.,2.,1.,7.]])
        saved=scores.clone()
        proc=FirstTokenABLogitsProcessor({'A':1,'B':2},prompt_length=5)
        masked=proc(torch.zeros(2,5,dtype=torch.long),scores)
        self.assertTrue(torch.equal(saved,scores))
        self.assertEqual(masked.argmax(-1).tolist(),[2,1])
        self.assertTrue(torch.isneginf(masked[:,[0,3]]).all())
        self.assertTrue(torch.equal(masked[:,[1,2]],scores[:,[1,2]]))
        later=proc(torch.zeros(2,6,dtype=torch.long),scores)
        self.assertIs(later,scores)
        self.assertEqual(later.argmax(-1).tolist(),[0,3])
    def test_first_token_processor_rejects_bad_configuration(self):
        with self.assertRaises(ValueError):FirstTokenABLogitsProcessor({'A':1,'B':1},4)
        with self.assertRaises(ValueError):FirstTokenABLogitsProcessor({'A':1,'B':2},0)
    def test_protocol_budgets_keep_native_and_answer_only_arms(self):
        native=output_protocol_settings('native_v1')
        self.assertEqual(native,{'max_new_tokens':NATIVE_GENERATION_BUDGET,'constrain':'none','answer_normalization':'native'})
        answer_only=output_protocol_settings('ab_constrained_v1',99)
        self.assertEqual(answer_only['max_new_tokens'],1)
        self.assertEqual(answer_only['constrain'],'all')
        first=output_protocol_settings('ab_first_token_v1',4)
        self.assertEqual(first['constrain'],'first')
        self.assertEqual(first['max_new_tokens'],FIRST_TOKEN_REASONING_BUDGET)
        self.assertGreater(FIRST_TOKEN_REASONING_BUDGET,NATIVE_GENERATION_BUDGET)
        with self.assertRaises(ValueError):output_protocol_settings('unknown')
    def test_format_success_is_not_classification_success(self):
        rows=[]
        for u in [True,False]:
            for v in ['B','R','N','R2']:
                for m in ['A_safe','A_unsafe']:
                    rows.append(dict(source_id=str(u),mapped_unsafe=u,variant=v,mapping=m,verdict='SAFE',ab_mass=.001))
        result=quality(rows)
        self.assertEqual(result['format_valid_pct'],100)
        self.assertEqual(result['accuracy_all_pct'],50)
        self.assertEqual(result['per_condition']['mapping_averaged']['B']['false_safe_pct'],100)
if __name__=='__main__':unittest.main()
