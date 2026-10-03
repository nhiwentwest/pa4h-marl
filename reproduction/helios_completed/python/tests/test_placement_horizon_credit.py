import unittest
from types import SimpleNamespace
import numpy as np
from placement_horizon_credit import horizon_reward_utilities
from placement_reward_credit import placement_reward_utilities


class HorizonCreditTests(unittest.TestCase):
    def fixture(self):
        e=SimpleNamespace(num_racks=2,num_hosts=8,rack_budget=3400.,step_idx=0,
            max_steps=10,_fine_per_step=4,_sla_priority_total=20.,_all_jobs=[1],
            _first_wait={},_admission_breached=set(),_deadline_crossed_ids=set(),
            _restart_wait_total={},_restart_wait_since={},_restart_breached=set(),
            sla_lambda_adm=0.,sla_lambda_restart=0.,running={})
        e.rack_of=lambda h:h//4
        e._pick_hosts_pa=lambda a,j:[a*4]
        e._rack_base_fine=lambda r:np.full(4,2600.)
        e._fine_window=lambda j,t:np.full(4,700. if t==0 else 1200.)
        e.running_remaining=lambda j:j.remaining_duration_steps
        j=SimpleNamespace(job_id='candidate',num_nodes=1,priority=1.,arrival_step=0,
                          remaining_duration_steps=5)
        return e,j
    def test_horizon_one_matches_existing_credit(self):
        e,j=self.fixture();m=np.ones(3,bool)
        np.testing.assert_array_equal(horizon_reward_utilities(e,j,m,1),placement_reward_utilities(e,j,m))
    def test_equal_energy_but_future_risk_separates_anchors(self):
        e,j=self.fixture()
        running=SimpleNamespace(placed_hosts=[4],arrival_step_placed=0,remaining_duration_steps=8)
        e.running={'existing':running}
        immediate=placement_reward_utilities(e,j,np.ones(3,bool))
        self.assertEqual(immediate[0],immediate[1])
        future=horizon_reward_utilities(e,j,np.ones(3,bool),4)
        self.assertGreater(future[0],future[1])
        self.assertEqual(future[-1],0.)
    def test_completed_existing_job_does_not_add_future_risk(self):
        e,j=self.fixture()
        e.running={'existing':SimpleNamespace(placed_hosts=[4],arrival_step_placed=0,remaining_duration_steps=1)}
        future=horizon_reward_utilities(e,j,np.ones(3,bool),4)
        self.assertEqual(future[0],future[1])
    def test_one_step_candidate_cannot_create_future_signal(self):
        e,j=self.fixture();j.remaining_duration_steps=1
        np.testing.assert_array_equal(horizon_reward_utilities(e,j,np.ones(3,bool),8),placement_reward_utilities(e,j,np.ones(3,bool)))
    def test_masks_and_episode_end(self):
        e,j=self.fixture();e.step_idx=e.max_steps-1
        m=np.array([True,False,True])
        np.testing.assert_array_equal(horizon_reward_utilities(e,j,m,8),placement_reward_utilities(e,j,m))
        m=np.array([False,False,True]);u=horizon_reward_utilities(e,j,m,4)
        self.assertTrue(np.isneginf(u[:-1]).all());self.assertEqual(u[-1],0.)
    def test_invalid_configuration(self):
        e,j=self.fixture()
        for h,g in [(0,.99),(1,0),(1,1.1)]:
            with self.assertRaises(ValueError):horizon_reward_utilities(e,j,np.ones(3,bool),h,g)

if __name__=='__main__':unittest.main()
