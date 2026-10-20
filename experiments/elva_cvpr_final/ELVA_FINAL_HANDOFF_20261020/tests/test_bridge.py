import unittest
from elva.bridge import LivePruneCoordinator, SameBudgetRouter
from elva.core import Schedule


class TestBridge(unittest.TestCase):
    def test_e94_full_32layer_live_with_fake_host_tensor(self):
        import torch
        sch=Schedule('E94',80,((2,48),(14,16)))
        x=torch.arange(80, dtype=torch.float32).reshape(80,1)
        def select(b,k,ids,state):
            self.assertEqual(tuple(ids),tuple(int(a) for a in state[:,0].tolist()))
            return tuple(range(k))
        def gather(state, local):
            new=state.index_select(0,torch.tensor(local,dtype=torch.long))
            return new, len(new)
        run=LivePruneCoordinator(sch, range(80),select,gather)
        for b in range(1,33):
            run.begin_block(b, x.shape[0])
            x=run.end_block(b,x)
        record=run.finish_prefill()
        self.assertEqual(record['w'],1024)
        self.assertEqual([v['after'] for v in record['events']],[48,16])
        self.assertEqual(len(record['final_original_patch_ids']),16)
        self.assertEqual(record['counts'],list(sch.trace()))

    def test_bridge_fail_fast_and_atomic_gather(self):
        sch=Schedule('small',8,((1,4),),n_layers=3,full_visual_tokens=10)
        def select(*args): return [0,1,2,3]
        def bad_gather(st,keeps): return st,8
        x=list(range(8)); run=LivePruneCoordinator(sch,x,select,bad_gather)
        with self.assertRaises(ValueError): run.begin_block(1,7)
        run.begin_block(1,8)
        with self.assertRaises(RuntimeError): run.end_block(1,x)
        self.assertEqual(run.ledger.ids,tuple(range(8)))
        self.assertEqual(run.completed,0)
        with self.assertRaises(ValueError): LivePruneCoordinator(sch,[1,2],select,bad_gather)
        with self.assertRaises(ValueError): run.begin_block(1,8)

    def test_router_same_prefix(self):
        s1=Schedule('medium',80,((2,48),(14,16)))
        s2=Schedule('wide',80,((2,64),(10,16)))
        router=SameBudgetRouter({'a':s1,'b':s2},decision_post_block=2)
        self.assertEqual(router.choose('b').budget,1024)
        with self.assertRaises(ValueError): router.choose('z')
        with self.assertRaises(ValueError): SameBudgetRouter({'a':s1,'b':s2},3)
        s3=Schedule('different entry',128,((2,48),(11,16)))
        with self.assertRaises(ValueError): SameBudgetRouter({'a':s1,'b':s3},2)

if __name__=='__main__':unittest.main()
