import unittest
from refusal_cues.diagnostics.gemma_invalids import redact_patterns
from refusal_cues.protocol import normalize_native_output, parse

class GemmaPilotFailureTests(unittest.TestCase):
    def test_actual_incomplete_output_patterns_have_no_recoverable_answer(self):
        for raw in ['-<|channel>thought\n','- The user is',"The user's"]:
            for mapping in ['A_safe','A_unsafe']:
                with self.subTest(raw=raw,mapping=mapping):
                    normalized,_=normalize_native_output(raw,'google/gemma-4-12B-it')
                    self.assertEqual(parse(normalized,mapping),'INVALID')

    def test_appending_an_answer_to_prose_is_not_a_parser_repair(self):
        for raw in ['- The user is asking. A',"The user's response is safe. B",'-<|channel>thought\n<channel|>A']:
            normalized,_=normalize_native_output(raw,'google/gemma-4-12B-it')
            self.assertEqual(parse(normalized),'INVALID')

    def test_unknown_patterns_accumulate_without_disclosing_text(self):
        result=redact_patterns({'private-one':2,'private-two':3,'known':1},{'known'})
        self.assertEqual(result,{'[unpublished output]':5,'known':1})
        self.assertEqual(sum(result.values()),6)

    def test_existing_empty_wrapper_is_still_supported(self):
        normalized,_=normalize_native_output('thought\n<channel|>A','google/gemma-4-12B-it')
        self.assertEqual(parse(normalized,'A_safe'),'SAFE')

if __name__=='__main__': unittest.main()
