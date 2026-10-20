import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from current_recycle import recycle_current
from recycle import recycle, cosine_cache
from extend_matrix import expanded
from paired_benchmark import summarize_blocks


class CurrentTests(unittest.TestCase):
    def test_rectangle_matches_full_current_cache(self):
        rng = np.random.default_rng(7)
        for n in (2, 9, 32, 172):
            x, f = rng.normal(size=(n,17)).astype('float32'), rng.normal(size=(n,7))
            xy, g = rng.normal(size=(n,2)), np.arange(n)%2
            for count in sorted({1, max(1,n//3), n}):
                k = rng.choice(n,count,replace=False)
                for alpha in (0., .25, .5, 1.):
                    for radius in (None, 2.):
                        y, audit = recycle_current(x,k,f,xy,g,alpha=alpha,radius=radius,cosine_floor=-.3)
                        ref = recycle(x,k,cosine_cache(f),xy,g,alpha=alpha,radius=radius,cosine_floor=-.3)
                        np.testing.assert_allclose(y,ref.hidden,atol=2e-6)
                        self.assertEqual(y.dtype,x.dtype)
                        if not audit['off']:
                            np.testing.assert_array_equal(audit['receiver'],ref.receiver)

    def test_off_does_not_need_metric_or_groups(self):
        x=np.arange(12,dtype='float32').reshape(4,3)
        y,a=recycle_current(x,[2,0],alpha=0)
        self.assertEqual(y.tobytes(),x[[2,0]].tobytes())
        self.assertTrue(a['off'])

    def test_group_and_prefix_controls(self):
        x=np.array([[1.],[3.],[5.],[7.]])
        kwargs=dict(metric=np.ones((4,2)),coords=np.zeros((4,2)),groups=np.zeros(4,dtype=int),alpha=1.)
        free,a=recycle_current(x,[0,2],**kwargs)
        prefix,b=recycle_current(x,[0,2],positions=np.arange(4),prefix_only=True,**kwargs)
        np.testing.assert_array_equal(b['receiver'],[1,-1])
        np.testing.assert_allclose(prefix,[[1.],[4.]])
        self.assertFalse(np.array_equal(free,prefix))
        kwargs['groups']=np.arange(4)
        isolated,c=recycle_current(x,[0,2],**kwargs)
        np.testing.assert_array_equal(isolated,x[[0,2]])
        self.assertEqual(c['assigned'],0)

    def test_zero_metric_and_immutable(self):
        x=np.ones((3,2),dtype='float16'); old=x.copy()
        y,a=recycle_current(x,[0],np.zeros((3,2)),np.zeros((3,2)),np.zeros(3,dtype=int))
        self.assertEqual(a['assigned'],0)
        np.testing.assert_array_equal(x,old)
        np.testing.assert_array_equal(y,x[[0]])

    def test_invalid_inputs(self):
        x=np.ones((3,2))
        args=dict(metric=x,coords=np.zeros((3,2)),groups=np.zeros(3,dtype=int))
        for k in ([],[0,0],[-1],[3],[.5]):
            with self.assertRaises(ValueError): recycle_current(x,k,**args)
        for extra in (dict(groups=None),dict(alpha=np.nan),dict(radius=-1),dict(prefix_only=True),dict(metric=np.full((3,2),np.nan))):
            with self.assertRaises(ValueError): recycle_current(x,[0],**{**args,**extra})

    def test_count_and_work_reverse_check(self):
        from costs import token_trace
        rows=expanded()
        self.assertEqual(len(rows),101)
        for k1 in (100,108,116,132):
            for final in (11,13,15):
                trace=token_trace(32,172,[(3,k1),(final,32)])
                expected=3*172+(final-3)*k1+(32-final)*32
                self.assertEqual(int(trace.sum()),expected)
                self.assertEqual((expected-3*172-(32-final)*32)//(final-3),k1)

    def test_paired_bootstrap_identical_faster_slower(self):
        for ratio in (1.,.9,1.1):
            rows=np.array([[100,100*ratio,100*ratio,100]]*12)
            out=summarize_blocks(rows,resamples=64)
            self.assertAlmostEqual(out['relative_saving'],1-ratio)
            np.testing.assert_allclose(out['bootstrap_saving_ci95'],[1-ratio]*2,atol=1e-14)
            self.assertAlmostEqual(out['candidate_mean_ms']/(1-out['relative_saving']),out['baseline_mean_ms'])
        with self.assertRaises(ValueError): summarize_blocks([[0,1,1,1],[1,1,1,1]])

    def test_quality_callback_rescores_clusters_not_sample_f1(self):
        from paired_quality import paired_quality
        labels=np.array([1,0,1,0])
        a=np.array([1,1,0,0]); b=labels.copy()
        def scorer(pred):
            def score(indices):
                y,p=labels[indices],pred[indices]
                tp=((y==1)&(p==1)).sum()
                fp=((y==0)&(p==1)).sum(); fn=((y==1)&(p==0)).sum()
                return {'f1':float(2*tp/(2*tp+fp+fn))}
            return score
        out=paired_quality(['im1','im1','im2','im2'],['pope']*4,scorer(a),scorer(b),repeats=64)
        self.assertAlmostEqual(out['f1']['delta'],.5)
        same=paired_quality(['x','x','y','y'],['p']*4,scorer(a),scorer(a),repeats=32)
        self.assertEqual(same['f1']['ci95'],[0.,0.])
        with self.assertRaises(ValueError):
            paired_quality(['x','x'],['p','q'],scorer(a),scorer(a),repeats=2)

    def test_cuda_parity_when_present(self):
        try: import torch
        except ImportError: self.skipTest('PyTorch absent; server CPU/CUDA parity required')
        from current_torch import recycle_current as tr
        from current_runtime import CurrentState
        rng=np.random.default_rng(8)
        for dev in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            for dtype in (torch.float32,torch.float16,torch.bfloat16):
                x=torch.tensor(rng.normal(size=(9,7)),device=dev,dtype=dtype)
                f=torch.tensor(rng.normal(size=(9,5)),device=dev,dtype=dtype)
                xy=torch.zeros((9,2),device=dev)
                groups=torch.arange(9,device=dev)%2
                pos=torch.arange(9,device=dev)
                keep=torch.tensor([7,0,3],device=dev)
                for alpha in (0.,.5,1.):
                    for prefix in (False,True):
                        y,a=tr(x,keep,f,xy,groups,alpha=alpha,cosine_floor=-.3,positions=pos,prefix_only=prefix,debug=True)
                        ref,b=recycle_current(x.float().cpu().numpy(),keep.cpu().numpy(),f.float().cpu().numpy(),xy.cpu().numpy(),groups.cpu().numpy(),alpha=alpha,cosine_floor=-.3,positions=pos.cpu().numpy(),prefix_only=prefix)
                        expected=torch.tensor(ref,device=dev).to(dtype).float().cpu().numpy()
                        np.testing.assert_allclose(y.float().cpu().numpy(),expected,atol=.02,rtol=.02)
                state=CurrentState(pos,xy,groups,pos,events=('first','final'),debug=True)
                y,ids,a=state.compress('first',x,keep,f)
                z,ids,b=state.compress('final',y,torch.tensor([1],device=dev),f[keep])
                self.assertEqual(ids.cpu().tolist(),[0])
                with self.assertRaises(ValueError): state.compress('final',z,torch.tensor([0],device=dev),f[:1])


if __name__=='__main__': unittest.main(verbosity=2)
