import unittest

from experiments.bela_query_anchor.question_protocol import content_mask_from_decodes
from experiments.bela_query_anchor.request_context import begin_question, current, end_question

import torch


class QuestionProtocolTests(unittest.TestCase):
    def test_template_words_drop_and_noun_stays(self):
        mask = content_mask_from_decodes([" Is", " there", " a", " dog", " in", " the", " image", "?"])
        self.assertEqual(mask, [False, False, False, True, False, False, False, False])

    def test_overlap_is_refused(self):
        token = begin_question(
            torch.tensor([1, 2]), None,
            question_text="dog", definition="test", source="actual",
        )
        try:
            with self.assertRaises(RuntimeError):
                begin_question(
                    torch.tensor([3]), None,
                    question_text="cat", definition="test", source="actual",
                )
            self.assertEqual(current().question_text, "dog")
        finally:
            end_question(token)
        self.assertIsNone(current())


if __name__ == "__main__":
    unittest.main()
