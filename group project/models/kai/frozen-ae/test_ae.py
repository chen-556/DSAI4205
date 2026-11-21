"""Offline checks: no pretrained downloads, teacher APIs, or benchmark training."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
KAI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KAI / ".ae-test-deps"))
spec = importlib.util.spec_from_file_location("run_ae", KAI / "run_frozen_ae.py")
ae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae)


class DataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.manifest = ae.load_snapshot()
        cls.train, cls.val, cls.test, cls.split, cls.fingerprint = ae.prepare_data(cls.raw, cls.manifest)

    def test_counts_and_no_test_labels(self):
        self.assertEqual((len(self.train), len(self.val), len(self.test)), (473, 118, 397))
        self.assertEqual(set(self.test), {"row_id", "text"})

    def test_exact_previous_split(self):
        # Independently exercise the existing data cell, not a duplicate splitter.
        import runpy
        import io, re, hashlib, unicodedata
        import ftfy
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import StratifiedGroupKFold
        recipe = runpy.run_path(str(KAI / "label-only/notebook.py"))["notebook"]
        source = next(c["source"] for c in recipe["cells"] if c["id"] == "data")
        with tempfile.TemporaryDirectory() as d:
            ns = dict(io=io, re=re, hashlib=hashlib, unicodedata=unicodedata, ftfy=ftfy,
                      np=np, pd=pd, Path=Path, json=json, StratifiedGroupKFold=StratifiedGroupKFold,
                      SPLIT_SEED=42, RUN_DIR=Path(d), DATA_MANIFEST=self.manifest, display=lambda *a:None)
            import base64, gzip
            ns.update(base64=base64, gzip=gzip, EMBEDDED_DATA={k:base64.b64encode(gzip.compress(v)).decode() for k,v in self.raw.items()})
            exec(compile(source, '<existing-data-cell>', 'exec'), ns)
            self.assertEqual(self.split.to_csv(index=False), ns['split_manifest'].to_csv(index=False))

    def test_manifest_roundtrip_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'split.csv'
            self.split.sample(frac=1, random_state=4).to_csv(path, index=False)
            self.assertEqual(ae.prepare_data(self.raw, self.manifest, path)[-1], self.fingerprint)
            bad = self.split.copy()
            bad.loc[0, 'group_id'] = 'changed'
            bad.to_csv(path, index=False)
            with self.assertRaises(ValueError): ae.prepare_data(self.raw, self.manifest, path)

    def test_no_import_time_model_or_teacher(self):
        self.assertNotIn('transformers', ae.__dict__)
        self.assertNotIn('requests', ae.__dict__)


class FeatureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = ae.load_helpers()

    def test_padding_pool(self):
        import torch
        hidden = torch.arange(24).reshape(2, 4, 3)
        self.assertTrue(torch.equal(self.h.last_valid_pool(hidden, torch.tensor([[1,1,0,0],[0,0,1,1]])), hidden[[0,1],[1,3]]))

    def test_cache_reuse_and_input_change(self):
        import numpy as np
        import pandas as pd
        frame = pd.DataFrame({'row_id':['a','b'], 'text':['first','second']})
        with tempfile.TemporaryDirectory() as d:
            spec = {'dimension':3}
            _, p, hit = self.h.feature_cache(d,spec,frame,[[1],[2]],[False,False],lambda:np.ones((2,3)))
            self.assertFalse(hit)
            _, same, hit = self.h.feature_cache(d,spec,frame,[[1],[2]],[False,False],lambda:self.fail('cache miss'))
            self.assertTrue(hit)
            self.assertEqual(p,same)
            _, changed, _ = self.h.feature_cache(d,spec,frame,[[1],[3]],[False,False],lambda:np.ones((2,3)))
            self.assertNotEqual(p,changed)
            with self.assertRaises(ValueError):
                self.h.feature_cache(d,spec,frame.assign(Sentiment=[0,1]),[[1],[2]],[False,False],lambda:None)

    def test_frozen_extraction_no_grad(self):
        import torch
        from types import SimpleNamespace
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(9, 4)
                self.config = SimpleNamespace(use_cache=True)
            def forward(self, input_ids, **kwargs):
                assert not torch.is_grad_enabled()
                return SimpleNamespace(last_hidden_state=self.emb(input_ids))
        class Tokenizer:
            def pad(self, rows, **kwargs):
                n=max(len(r['input_ids']) for r in rows)
                return {'input_ids':torch.tensor([r['input_ids']+[0]*(n-len(r['input_ids'])) for r in rows]),
                        'attention_mask':torch.tensor([r['attention_mask']+[0]*(n-len(r['input_ids'])) for r in rows])}
        model=Decoder()
        out=self.h.extract_frozen_vectors(model,Tokenizer(),[[1,2],[3]],2,'cpu')
        self.assertEqual(out.shape,(2,4))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.training)

    def test_scaler_fits_only_train(self):
        import numpy as np
        x=np.arange(64,dtype=float).reshape(16,4)
        y=np.tile(np.arange(4),4)
        cfg=dict(hidden=4,max_epochs=2,patience=2,batch_size=8,alpha=.01,learning_rate=.001)
        bundle=self.h.fit_independent_mlp(x,y,x+1000,y,42,cfg)
        np.testing.assert_allclose(bundle['scaler'].mean_,x.mean(axis=0))
        pred,prob=self.h.predict_independent_mlp(bundle,x)
        self.assertEqual(prob.shape,(16,4))
        self.assertEqual(len(pred),16)


if __name__ == '__main__':
    unittest.main()
