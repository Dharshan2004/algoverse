import unittest
import torch
from refusal_cues.ab_constraint import ABOnlyLogitsProcessor
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
