import unittest
from elva.core import Schedule,TokenLedger,validate_survival_swap,combine_rank_scores,r4_score,assert_real_trace
from elva.cli import plans

class TestContracts(unittest.TestCase):
    def test_presets(self):
        allp = plans('plans/schedules.json')
        self.assertGreater(len(allp), 10)
        for p in allp:
            self.assertEqual(p.budget,sum(p.trace()))
            self.assertEqual(len(p.trace()),32)
            self.assertEqual(p.trace()[0],p.entry)
            self.assertEqual(p.trace()[-1],p.events[-1][1])
            p.verify_budget(p.budget)
            copy=Schedule.from_dict(p.to_dict())
            self.assertEqual(copy,p)

    def test_budget_anchors(self):
        s=Schedule('r3',172,((4,90),(12,32)))
        self.assertEqual(s.budget,2048)
        self.assertEqual(s.trace(),(172,)*4+(90,)*8+(32,)*20)
        self.assertAlmostEqual(s.compression_pct, 88.8888888889)
        w=Schedule('ref',172,((3,116),(13,32)))
        self.assertEqual(w.budget,2284)
        self.assertAlmostEqual(Schedule('k60 hypothetical',60,((4,48),(12,20))).compression_pct,94.4444444)

    def test_bad_plan(self):
        bad=[((0,32),),((32,32),),((4,90),(4,32)),((5,90),(3,32)),
             ((4,172),),((4,0),),((4,180),)]
        for e in bad:
            with self.subTest(e=e),self.assertRaises(ValueError):
                Schedule('bad',172,e)
        with self.assertRaises(ValueError): Schedule.from_segments('bad',(60,32),(4,20))
        with self.assertRaises(ValueError): Schedule('bad',0,())
        with self.assertRaises(ValueError): Schedule('bad',172,(),0)

    def test_trace(self):
        s=Schedule('ok',80,((2,48),(14,16)))
        self.assertEqual(s.budget,1024)
        assert_real_trace(list(s.trace()),s)
        for bad in ([80]*32,list(s.trace())[:-1],[-1]*32):
            with self.assertRaises(ValueError): assert_real_trace(bad,s)
        with self.assertRaises(ValueError): s.verify_budget(1025)

    def test_ledger(self):
        z=TokenLedger([12,8,9,0,1,2])
        self.assertEqual(z.indices_of([9,12]),(2,0))
        self.assertEqual(z.next_keep([0,2,4],3),(12,9,1))
        self.assertEqual(z.commit([0,2,4],3),(12,9,1))
        self.assertEqual(z.commit([0,2],2),(12,1))
        for bad in ([1,0],[0,0],[2,1],[3,4]):
            with self.assertRaises(ValueError): z.next_keep(bad,2)
        with self.assertRaises(ValueError): TokenLedger([1,1])
        with self.assertRaises(ValueError): z.indices_of([999])

    def test_swap(self):
        s=validate_survival_swap(range(10), [0,2],[3,4],[5,6],duration=4)
        a,b=s.branch_ids()
        self.assertEqual(a,(0,2,3,4))
        self.assertEqual(b,(0,2,5,6))
        self.assertEqual(s.delta_token_layers,8)
        self.assertEqual(len(a),len(b))
        for common,ga,gb,d in [([0],[1],[1],4),([0],[1],[2,3],4),
                                ([0],[0],[2],4),([0],[1],[11],4),([0],[1],[2],0)]:
            with self.subTest(common=common,ga=ga,gb=gb,d=d),self.assertRaises(ValueError):
                validate_survival_swap(range(10),common,ga,gb,d)

    def test_rank_fusion(self):
        a=combine_rank_scores([1,2,3],[3,2,1],beta=0)
        b=combine_rank_scores([1,2,3],[3,2,1],beta=1)
        c=combine_rank_scores([1,2,3],[3,2,1],beta=.5)
        self.assertEqual(a,(0.,.5,1.))
        self.assertEqual(b,(1.,.5,0.))
        self.assertEqual(c,(.5,.5,.5))
        for bad in [-.2,1.2,float('nan')]:
            with self.assertRaises(ValueError): combine_rank_scores([1],[2],beta=bad)
        with self.assertRaises(ValueError): combine_rank_scores([1,2],[2],beta=.5)
        with self.assertRaises(ValueError): combine_rank_scores([float('nan')],[2],beta=.5)

    def test_r4(self):
        self.assertAlmostEqual(r4_score(57.18,84.82,1765.17,62.20),95.8950108575)
        self.assertAlmostEqual(r4_score(55.41,81.84,1710.65,59.97),92.706153895)
        with self.assertRaises(ValueError): r4_score(-1,1,1,1)

if __name__ == '__main__': unittest.main()
