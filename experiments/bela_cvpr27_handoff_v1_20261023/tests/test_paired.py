import unittest
from scripts.paired import compare,pope,PairError

class TestPaired(unittest.TestCase):
    def test_pair_and_deterministic_bootstrap(self):
        a={str(i):{'sample_id':str(i),'image_id':str(i//2),'correct':v} for i,v in enumerate([True,False,True,False])}
        b={str(i):{'sample_id':str(i),'image_id':str(i//2),'correct':v} for i,v in enumerate([True,True,False,True])}
        x=compare(a,b,seed=1,bootstrap=30)
        self.assertEqual(x['paired']['candidate_only'],2)
        self.assertEqual(x['paired']['baseline_only'],1)
        self.assertEqual(x['delta_pp'],25)
        self.assertEqual(x,compare(a,b,seed=1,bootstrap=30))
    def test_alignment_error(self):
        a={'1':{'sample_id':'1','image_id':'one','correct':True}}
        b={'2':{'sample_id':'2','image_id':'one','correct':True}}
        with self.assertRaises(PairError):compare(a,b)
        b={'1':{'sample_id':'1','image_id':'two','correct':True}}
        with self.assertRaises(PairError):compare(a,b)
    def test_pope_binary(self):
        rows={str(i):{'sample_id':str(i),'image_id':str(i),'correct':False,
                      'subset':v[2],'gold_binary':v[0],'pred_binary':v[1]}
             for i,v in enumerate([('yes','yes','random'),('no','yes','random'),
                                   ('yes','no','popular'),('no','no','adversarial')])}
        r=pope(rows)['micro']
        self.assertEqual([r['TP'],r['FP'],r['FN'],r['TN']],[1,1,1,1])
        self.assertEqual(r['F1'],0.5)
        with self.assertRaises(PairError):pope({'q':{'gold_binary':'N/A','pred_binary':'yes','subset':'random'}})
    def test_invalid_bootstrap(self):
        a={'1':{'sample_id':'1','image_id':'one','correct':True}}
        with self.assertRaises(PairError):compare(a,a,bootstrap=0)
