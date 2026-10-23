import unittest
from scripts.audit import expand,summary,validate_record,AuditError

class TestAudit(unittest.TestCase):
    def test_W_Q(self):
        for seg in [[[4,172],[8,90],[20,32]],[[4,172],[8,120],[20,20]],
                    [[4,172],[8,130],[20,16]],[[4,172],[8,108],[8,32],[12,20]],
                    [[12,128],[8,37],[12,18]]]:
            self.assertEqual(summary(expand(seg))['W'],2048)
        self.assertEqual(summary(expand([[4,172],[8,90],[20,20]]))['W'],1808)
        self.assertEqual(summary(expand([[4,172],[8,120],[20,32]]))['W'],2288)
        self.assertEqual(summary(expand([[4,172],[8,90],[20,32]]))['Q'],203616)
        self.assertEqual(summary(expand([[4,172],[8,120],[20,20]]))['Q'],241536)
        self.assertEqual(summary(expand([[12,128],[8,37],[12,18]]))['Q'],211448)
    def example(self):
        return {'run_id':'sha_a2','sample_id':'q1','image_id':'i1',
            'tokens_per_layer':[172]*4+[120]*8+[20]*20,
            'cut_events':[
             {'after_layer_index':3,'patch_ids_before':list(range(172)),
              'patch_ids_after':list(range(120)),'selector_module':'real.layer',
              'selector_function':'f1','selector_callsite':'file.py:111'},
             {'after_layer_index':11,'patch_ids_before':list(range(120)),
              'patch_ids_after':list(range(20)),'selector_module':'real.layer',
              'selector_function':'f2','selector_callsite':'file.py:222'}]}
    def test_valid(self): self.assertEqual(validate_record(self.example())['W'],2048)
    def test_patch_reintroduced(self):
        x=self.example();x['cut_events'][1]['patch_ids_after'][-1]=171
        with self.assertRaises(AuditError): validate_record(x)
    def test_missing_score_provenance(self):
        x=self.example();x['cut_events'][0]['selector_function']=''
        with self.assertRaises(AuditError): validate_record(x)
    def test_layer_offbyone(self):
        x=self.example();x['cut_events'][0]['after_layer_index']=4
        with self.assertRaises(AuditError): validate_record(x)
    def test_bad_layers_and_values(self):
        for stages in ([[3,172],[8,120],[20,20]], [[4,172],[8,True],[20,20]]):
            with self.assertRaises(AuditError): expand(stages)
        with self.assertRaises(AuditError): expand([[4,172],[8,120],[10,20]])
        with self.assertRaises(AuditError): expand([[4,172],[8,180],[20,20]])
