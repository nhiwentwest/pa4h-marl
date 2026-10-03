import copy
import json
from pathlib import Path
import unittest
from pilot_protocol import validate_protocol,recipe_digest,validate_resume


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p=json.loads((Path(__file__).parents[2]/'protocol.json').read_text())
    def test_disjoint_protocol(self):validate_protocol(self.p)
    def test_adjacent_windows_are_allowed_but_overlap_is_rejected(self):
        p=copy.deepcopy(self.p);p['development_hours']=[910,940];validate_protocol(p)
        p['development_hours']=[909,940]
        with self.assertRaises(ValueError):validate_protocol(p)
    def test_holdout_and_automatic_long_training_are_rejected(self):
        for key in ['heldout_used','long_training_automatic']:
            p=copy.deepcopy(self.p);p[key]=True
            with self.assertRaises(ValueError):validate_protocol(p)
        p=copy.deepcopy(self.p);p['development_hours']=[1020]
        with self.assertRaises(ValueError):validate_protocol(p)
    def test_recipe_distinguishes_seed_credit_and_workload(self):
        a=recipe_digest(self.p,'immediate',1)
        self.assertNotEqual(a,recipe_digest(self.p,'horizon',1))
        self.assertNotEqual(a,recipe_digest(self.p,'immediate',2))
        p=copy.deepcopy(self.p);p['train_hours']=[1000]
        self.assertNotEqual(a,recipe_digest(p,'immediate',1))
    def test_resume_must_match_source_recipe_and_batch(self):
        m=dict(recipe_sha256='recipe',source_sha256='source',episode=4)
        validate_resume(m,'recipe','source',4)
        for a,b,c in [('other','source',4),('recipe','other',4),('recipe','source',8)]:
            with self.assertRaises(ValueError):validate_resume(m,a,b,c)
        m['episode']=3
        with self.assertRaises(ValueError):validate_resume(m,'recipe','source',3)
    def test_invalid_load_and_batch_budget(self):
        for key,value in [('subsample',0),('window_hours',0),('horizon',0),
                          ('train_hours',[]),('train_hours',[900,900]),
                          ('pilot_episodes',25),('batch_episodes',2)]:
            p=copy.deepcopy(self.p);p[key]=value
            with self.assertRaises(ValueError):validate_protocol(p)

if __name__=='__main__':unittest.main()
