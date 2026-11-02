import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from bela_next_fix.scoring import choice_letter,paired_choice_audit
class ScoreTests(unittest.TestCase):
    def test_short_choice(self):
        for ans in ('B','B.','(B)',' B ) ','b'):
            self.assertEqual(choice_letter(ans),'B')
        for ans in ('Because this is true', 'B is the correct answer', 'This is B','A or B','B.C'):
            self.assertIsNone(choice_letter(ans))
    def test_comparisons(self):
        rows=paired_choice_audit(['B.','C'],['B','A'])
        self.assertEqual([x['parsed_correct'] for x in rows],[True,False])
if __name__=='__main__':unittest.main()
