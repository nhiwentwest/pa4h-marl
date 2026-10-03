import unittest
from completion_victim import completion_preserving_mask

class CompletionVictimTests(unittest.TestCase):
    def test_observed_zero_slack_job_has_a_legal_alternative(self):
        self.assertEqual(completion_preserving_mask([True, True, False, False],
                         [116, 19], 64, 180), [False, True, False, False])
    def test_legal_causal_mask_stays_respected(self):
        self.assertEqual(completion_preserving_mask([True, False, True, False],
                         [116, 19, 20], 64, 180), [False, False, True, False])
    def test_no_alternative_preserves_original_choices(self):
        self.assertEqual(completion_preserving_mask([True, False], [116, 19], 64, 180),
                         [True, False])
        self.assertEqual(completion_preserving_mask([True, True], [116, 116], 64, 180),
                         [True, True])
    def test_already_unfinishable_and_positive_slack_are_not_reclassified(self):
        self.assertEqual(completion_preserving_mask([True, True], [117, 115], 64, 180),
                         [True, True])
    def test_does_not_mutate_caller_mask(self):
        mask = [True, True]
        completion_preserving_mask(mask, [116, 19], 64, 180)
        self.assertEqual(mask, [True, True])
    def test_bad_state_or_mapping_fails(self):
        for mask, remaining, step in [([False], [1], 1), ([True], [-1], 1),
                                     ([True], [], 1), ([True], [1], 180),
                                     ([True], [1, 2], 1)]:
            with self.assertRaises(ValueError):
                completion_preserving_mask(mask, remaining, step, 180)

if __name__ == '__main__':
    unittest.main()
