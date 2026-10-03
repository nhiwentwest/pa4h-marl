import unittest
from types import SimpleNamespace
import numpy as np
import torch
from placement_reward_credit import placement_reward_utilities, expected_reward_regret

class PlacementRewardTests(unittest.TestCase):
    def env(self):
        e=SimpleNamespace(num_racks=2, num_hosts=8, rack_budget=1000., _sla_priority_total=30.,
          _all_jobs=list(range(10)), step_idx=0, _first_wait={}, _admission_breached=set(),
          _deadline_crossed_ids=set(), _restart_wait_total={}, _restart_wait_since={},
          _restart_breached=set(), sla_lambda_adm=1., sla_lambda_restart=1.)
        e._pick_hosts_pa=lambda anchor, job: [0] if anchor==0 else [4]
        e._rack_base_fine=lambda rack: np.full(4, 200. if rack==0 else 400.)
        e._fine_window=lambda job, elapsed: np.full(4, 900.)
        e.rack_of=lambda host: host//4
        return e
    def test_reward_difference_matches_service_wait_energy(self):
        from gang_env import W_SERVE,W_WAIT,W_ENERGY,NREL_IDLE_W,NREL_PEAK_W
        env=self.env(); job=SimpleNamespace(job_id=1,num_nodes=1,priority=3.,arrival_step=0)
        u=placement_reward_utilities(env,job,np.array([True,True,True]))
        expected=W_SERVE/8+W_WAIT*3/24-W_ENERGY*(900-NREL_IDLE_W)/(8*NREL_PEAK_W)
        np.testing.assert_allclose(u,[expected,expected,0.],rtol=1e-6)
    def test_same_energy_different_pressure_is_tied_unless_violating(self):
        env=self.env(); job=SimpleNamespace(job_id=1,num_nodes=1,priority=3.,arrival_step=0)
        a=placement_reward_utilities(env,job,np.ones(3,bool))
        self.assertAlmostEqual(a[0],a[1])
        env._rack_base_fine=lambda rack: np.full(4,950. if rack==1 else 200.)
        b=placement_reward_utilities(env,job,np.ones(3,bool))
        self.assertLess(b[1],b[0])
    def test_forced_defer_and_masks(self):
        e=self.env(); j=SimpleNamespace(job_id=1,num_nodes=1,priority=3.,arrival_step=0)
        u=placement_reward_utilities(e,j,np.array([False,False,True]))
        self.assertTrue(np.isneginf(u[:2]).all()); self.assertEqual(u[2],0.)
    def test_regret_does_not_force_uniform_distribution_on_ties(self):
        logits=torch.tensor([[8.,-2.,-float('inf')]],requires_grad=True)
        u=torch.tensor([[2.,2.,-float('inf')]])
        loss,n=expected_reward_regret(logits,u,torch.tensor([[True,True,False]]))
        self.assertEqual(n,0); self.assertEqual(loss.item(),0.)
        loss.backward(); torch.testing.assert_close(logits.grad,torch.zeros_like(logits))
    def test_regret_pushes_probability_away_from_defer_and_stays_finite(self):
        logits=torch.zeros(1,3,requires_grad=True)
        u=torch.tensor([[.1,.1,0.]])
        loss,n=expected_reward_regret(logits,u,torch.ones_like(u,dtype=torch.bool))
        self.assertEqual(n,1); loss.backward()
        self.assertGreater(logits.grad[0,2].item(),0)
        self.assertLess(logits.grad[0,0].item(),0)

    def test_sla_deadline_credit_is_normalized_and_only_charged_once(self):
        from gang_env import W_SLA_BASE,W_DEADLINE,SLA_MAX_WAIT_STEPS
        env=self.env(); job=SimpleNamespace(job_id=1,num_nodes=1,priority=3.,arrival_step=0)
        baseline=placement_reward_utilities(env,job,np.ones(3,bool))[0]
        env.step_idx=SLA_MAX_WAIT_STEPS
        crossing=placement_reward_utilities(env,job,np.ones(3,bool))[0]
        self.assertAlmostEqual(float(crossing-baseline),(W_SLA_BASE+1.)*3/30+W_DEADLINE/10,places=6)
        env._admission_breached.add(1);env._deadline_crossed_ids.add(1)
        self.assertAlmostEqual(placement_reward_utilities(env,job,np.ones(3,bool))[0],baseline)
    def test_restart_credit_is_normalized_and_one_shot(self):
        from gang_env import W_SLA_BASE,SLA_MAX_RESTART_WAIT_STEPS
        env=self.env(); job=SimpleNamespace(job_id=1,num_nodes=1,priority=3.,arrival_step=0)
        env.step_idx=SLA_MAX_RESTART_WAIT_STEPS+2
        baseline=placement_reward_utilities(env,job,np.ones(3,bool))[0]
        env._restart_wait_since[1]=2
        crossing=placement_reward_utilities(env,job,np.ones(3,bool))[0]
        self.assertAlmostEqual(float(crossing-baseline),(W_SLA_BASE+1.)*3/30,places=6)
        env._restart_breached.add(1)
        self.assertAlmostEqual(placement_reward_utilities(env,job,np.ones(3,bool))[0],baseline)

if __name__=='__main__':unittest.main()
