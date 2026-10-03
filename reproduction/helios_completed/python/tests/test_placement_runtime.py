import unittest
from types import SimpleNamespace
import numpy as np
import torch
from models import Actor,RackSTGNNActor,select_action
from placement_runtime import install_plan_cache,forced_action,install_forced_action_shortcut


class RuntimeTests(unittest.TestCase):
    def env(self):
        e=SimpleNamespace(step_idx=0,_feas_budget=1000.,free=[True]*8,calls=0)
        e.free_host_map=lambda:e.free
        def pick(anchor,job):
            e.calls+=1
            return [anchor*4] if e.free[anchor*4] else None
        e._pick_hosts_pa=pick
        j=SimpleNamespace(per_node_trace_w=np.ones(4),num_nodes=1,progress_steps=0)
        return e,j
    def test_hit_and_returned_plan_mutation(self):
        e,j=self.env();stats=install_plan_cache(e,verify_hits=0)
        plan=e._pick_hosts_pa(0,j);plan[0]=7
        self.assertEqual(e._pick_hosts_pa(0,j),[0]);self.assertEqual(e.calls,1)
        self.assertEqual(stats['hits'],1)
    def test_invalidation_after_place_advance_budget_and_progress(self):
        e,j=self.env();install_plan_cache(e,verify_hits=0);e._pick_hosts_pa(0,j)
        e.free=[False]+[True]*7
        self.assertIsNone(e._pick_hosts_pa(0,j))
        e.step_idx=1;e._pick_hosts_pa(0,j)
        e._feas_budget=900.;e._pick_hosts_pa(0,j)
        j.progress_steps=1;e._pick_hosts_pa(0,j)
        j.per_node_trace_w=np.ones(4);e._pick_hosts_pa(0,j)
        self.assertEqual(e.calls,6)
    def test_cached_none_and_verified_hits(self):
        e,j=self.env();e.free=[False]*8
        stats=install_plan_cache(e,verify_hits=1)
        for _ in range(3):self.assertIsNone(e._pick_hosts_pa(0,j))
        self.assertEqual(stats['verified'],1);self.assertEqual(e.calls,2)
    def test_forced_shortcut_preserves_rng_for_mlp_and_stgnn(self):
        actors=[(Actor(3,17),np.zeros(3,dtype=np.float32))]
        actors.append((RackSTGNNActor(16,6,7,context_dim=29,include_defer=True,action_context_dim=17),
                       np.zeros(701,dtype=np.float32)))
        for actor,obs in actors:
            for mode in ('train','eval'):
                mask=np.zeros(17,bool);mask[-1]=True
                torch.manual_seed(42);before=torch.get_rng_state()
                expected=select_action(actor,obs,mask,mode);after=torch.get_rng_state()
                torch.set_rng_state(before);actual=forced_action(actor,mask,mode)
                self.assertEqual(actual,expected)
                torch.testing.assert_close(torch.get_rng_state(),after,rtol=0,atol=0)
    def test_shortcut_leaves_nonforced_calls_on_original_path(self):
        calls=[];t=SimpleNamespace(select_action=lambda *args,**kwargs:calls.append(kwargs) or (1,-.2))
        install_forced_action_shortcut(t)
        self.assertEqual(t.select_action(None,None,np.ones(3,bool),'train'),(1,-.2))
        self.assertEqual(len(calls),1)
    def test_invalid_forced_requests(self):
        actor=Actor(3,3)
        for mask,mode in [(np.ones(3,bool),'train'),(np.zeros(3,bool),'eval'),(np.array([1,0,0]),'bad')]:
            with self.assertRaises(ValueError):forced_action(actor,mask,mode)

if __name__=='__main__':unittest.main()
