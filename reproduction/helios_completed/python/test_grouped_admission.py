import unittest
from grouped_admission import select_grouped_admission

class GroupedAdmissionTests(unittest.TestCase):
    def test_observed_near_uniform_policy_admits_instead_of_collapsing_to_defer(self):
        probs = [0.9410514235496521 / 16] * 16 + [0.05894855037331581]
        self.assertGreater(probs[-1], max(probs[:-1]))
        self.assertEqual(select_grouped_admission(probs, [True] * 17), 0)
    def test_majority_defer_is_retained(self):
        self.assertEqual(select_grouped_admission([.2, .1, .7], [True]*3), 2)
    def test_illegal_actions_do_not_contribute_admission_mass_or_win_ranking(self):
        self.assertEqual(select_grouped_admission([.8, .05, .15], [False, True, True]), 2)
        self.assertEqual(select_grouped_admission([.8, .15, .05], [False, True, True]), 1)
    def test_conditional_rack_choice_and_ties_are_deterministic(self):
        self.assertEqual(select_grouped_admission([.1, .3, .2, .4], [True]*4), 1)
        self.assertEqual(select_grouped_admission([.25, .25, .5], [True]*3), 0)
    def test_forced_defer_and_forced_admission(self):
        self.assertEqual(select_grouped_admission([0., 0., 1.], [False, False, True]), 2)
        self.assertEqual(select_grouped_admission([.1, .9], [True, False]), 0)
    def test_invalid_or_nonfinite_legal_probabilities_are_rejected(self):
        for probs, mask in [([1.], [True]), ([.1,.9], [True]), ([0.,0.], [True,True]),
                            ([-.1,1.1], [True,True]), ([float('nan'),1.], [True,True]),
                            ([.2,.8], [False,False])]:
            with self.assertRaises(ValueError):select_grouped_admission(probs,mask)

if __name__ == '__main__':unittest.main()
