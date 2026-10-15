"""Run with python checks.py. Synthetic checks never establish VLM accuracy."""
import importlib.util
import json
import tempfile
import unittest
import subprocess
import sys
import os
from pathlib import Path
import numpy as np
from controller_numpy import (Model, SCHEMA, region_summary, projection, inputs,
                              group_splits, fit, choose_defer, token_layers,
                              decision_features, _loss_and_grad, load_dataset)
from adapter_contract import collect_pair
from audit_r4 import r4, factorial


class Checks(unittest.TestCase):
    def test_r4(self):
        self.assertAlmostEqual(r4({'GQA':61.9,'POPE':85.8,'MME':1852,'MMBench':64.1}),100)
        with self.assertRaises(ValueError):
            r4({'GQA':61.9,'POPE':85.8,'MME':1852,'MME2':64.1})
        d = factorial({'COVER_ONE':95.21,'COVER_TWO':96.10,'BASE_ONE':95.22,'BASE_TWO':96.10})
        self.assertAlmostEqual(d['interaction_pp'],.01)

    def test_budget(self):
        self.assertEqual(token_layers(32,172,[(4,96),(8,32)]),1840)
        self.assertEqual(token_layers(32,172,[(4,96),(12,32)]),2096)
        with self.assertRaises(ValueError):
            token_layers(32,172,[(12,96),(8,32)])

    def test_region_moments(self):
        x = np.array([[1.,0.],[-1.,0.],[0.,1.]])
        s = region_summary(x,np.array([0,0,1]),np.array([True,True,False]),np.eye(2),groups=3).reshape(3,5)
        self.assertEqual(s[0,0],1)
        np.testing.assert_allclose(s[0,1:3],0)
        self.assertEqual(s[0,3],1)  # Second moment detects cancelling features.
        np.testing.assert_allclose(s[1:],0)

    def test_matched_metadata(self):
        b=np.zeros((2,3)); h=np.ones((2,10)); r=np.full((2,10),2.)
        keep=inputs(b,h,r,'retained',stride=5)[:,3:]
        np.testing.assert_allclose(keep[:,::5],h[:,::5])
        np.testing.assert_allclose(keep[:,1:5],2)
        self.assertEqual(inputs(b,h,r,'history',5).shape,inputs(b,h,r,'geometry',5).shape)

    def test_group_split(self):
        groups=np.array(['same_image','other','same_image'])
        masks=group_splits(groups)
        for mask in masks:
            self.assertEqual(mask[0],mask[2])
        np.testing.assert_array_equal(sum(m.astype(int) for m in masks),1)

    def test_signed_decision(self):
        np.testing.assert_array_equal(choose_defer(np.array([-1.,0.,1.,3.]),np.ones(4),1),[False,False,False,True])
        with self.assertRaises(ValueError):
            choose_defer(np.ones(1),np.zeros(1),1)

    def test_gradients(self):
        rng=np.random.default_rng(3)
        z=rng.normal(size=(4,3)); y=rng.normal(size=4)
        params=[rng.normal(size=(3,2)),rng.normal(size=2),rng.normal(size=(2,1)),rng.normal(size=1)]
        _, grad=_loss_and_grad(z,y,params,1e-5)
        for j,p in enumerate(params):
            for index in np.ndindex(p.shape):
                before=p[index]; eps=1e-6
                p[index]=before+eps; plus=_loss_and_grad(z,y,params,1e-5)[0]
                p[index]=before-eps; minus=_loss_and_grad(z,y,params,1e-5)[0]
                p[index]=before
                self.assertAlmostEqual(grad[j][index],(plus-minus)/(2*eps),places=5)

    def test_fit_checkpoint(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(120,4)); y=np.tanh(x[:,0])-.3*x[:,1]
        train=np.arange(120)<80; val=(np.arange(120)>=80)&(np.arange(120)<100)
        model=fit(x,y,train,val,hidden=12,steps=250)
        mse=np.mean((model.predict(x[100:])-y[100:])**2)
        self.assertLess(mse,.1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'model.npz'; model.save(path,{'kind':'synthetic'})
            loaded,meta=Model.load(path)
            np.testing.assert_allclose(loaded.predict(x),model.predict(x))
            self.assertEqual(meta['kind'],'synthetic')

    def test_feature_shapes(self):
        rng=np.random.default_rng(2)
        f=decision_features(rng.normal(size=64),rng.normal(size=(96,64)),rng.random(96),
                            projection(64,16,1),projection(64,16,2),target_k=32,
                            completed_blocks=8,total_blocks=32,original_n=576)
        self.assertEqual(f.shape,(55,)); self.assertTrue(np.isfinite(f).all())

    def test_pair_prefix_guard(self):
        class Fake:
            def prepare_prefix(self,sample): return {}
            def features(self,prefix): return {}
            def clone_prefix(self,prefix): return dict(prefix)
            def run_branch(self,prefix,sample,action):
                return {'prefix_fingerprint':action,'answer_token_ids':[1], 'answer_nll':1.}
        with self.assertRaises(ValueError): collect_pair(Fake(),{})

    def test_training_cli_and_synthetic_guard(self):
        rng=np.random.default_rng(8); n=240
        manifest={k:'synthetic-only' for k in [
            'checkpoint_hash','stage1_config_hash','selector_hash','schedule_hash',
            'prompt_hash','projection_hash','cache_policy','query_rule','cost_source']}
        manifest['data_split']='development_only'
        base=rng.normal(size=(n,5)); history=rng.normal(size=(n,10))
        delta=.2*np.tanh(history[:,1])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            data=Path(tmp)/'synthetic.npz'; output=Path(tmp)/'fit'
            np.savez_compressed(data,schema=SCHEMA,kind='synthetic',base=base,
                                history=history,retained=rng.normal(size=(n,10)),
                                early_nll=np.ones(n),late_nll=1-delta,extra_ms=np.ones(n),
                                image_ids=np.array(['synthetic-image-'+str(i) for i in range(n)]),
                                sample_ids=np.array(['synthetic-sample-'+str(i) for i in range(n)]),
                                stride=5,manifest=json.dumps(manifest))
            with self.assertRaises(ValueError): load_dataset(data)
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1')
            subprocess.run([sys.executable,str(Path(__file__).with_name('train_controller.py')),
                            str(data),'--out',str(output),'--allow-synthetic',
                            '--steps','30','--hidden','8'],check=True,capture_output=True,env=env)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(sum(report['n'].values()),n)
            self.assertEqual(set(report['modes']),{'zero','geometry','retained','history'})
            self.assertEqual(report['kind'],'synthetic')
            shapes=[]
            for mode in report['modes']:
                model,meta=Model.load(output/(mode+'.npz'))
                shapes.append(model.w1.shape)
                self.assertEqual(meta['kind'],'synthetic')
            self.assertEqual(len(set(shapes)),1)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable: server parity required')
    def test_torch_parity(self):
        import torch
        import controller_torch as backend
        rng=np.random.default_rng(0)
        x=rng.normal(size=(100,12)); y=np.tanh(x[:,0])
        model=fit(x,y,np.arange(100)<60,(np.arange(100)>=60)&(np.arange(100)<80),steps=40)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'m.npz'
            model.save(path,{'schema':SCHEMA,'kind':'synthetic','lambda':.1})
            online=backend.FrozenController(path,allow_synthetic=True)
            np.testing.assert_allclose(online(torch.tensor(x)).numpy(),model.predict(x),atol=2e-5,rtol=1e-4)
        v=rng.normal(size=(20,16)); g=np.arange(20)%4; m=np.arange(20)%3==0; p=projection(16,8,7)
        cpu=region_summary(v,g,m,p,4)
        device=backend.region_summary(torch.tensor(v),torch.tensor(g),torch.tensor(m),torch.tensor(p),4)
        np.testing.assert_allclose(device.numpy(),cpu,atol=2e-5,rtol=1e-4)
        query=rng.normal(size=16); attention=rng.random(20); qp=projection(16,16,1); vp=projection(16,16,2)
        kwargs=dict(target_k=8,completed_blocks=8,total_blocks=32,original_n=576)
        cpu=decision_features(query,v,attention,qp,vp,**kwargs)
        gpu=backend.decision_features(*map(torch.tensor,[query,v,attention,qp,vp]),**kwargs)
        np.testing.assert_allclose(gpu.numpy(),cpu,atol=2e-5,rtol=1e-4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
