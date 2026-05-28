"""Smoke tests for evaluation metric helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.eval.metrics import align_signals, si_sdr_db, snr_improvement_db


class MetricsSmokeTests(unittest.TestCase):
    def test_align_signals_trims_to_shortest_length(self) -> None:
        reference = np.array([1.0, 2.0, 3.0, 4.0])
        estimate = np.array([1.0, 2.0])

        aligned_reference, aligned_estimate = align_signals(reference, estimate)

        np.testing.assert_array_equal(aligned_reference, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(aligned_estimate, np.array([1.0, 2.0]))

    def test_align_signals_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            align_signals(np.array([]), np.array([1.0]))

    def test_snr_improvement_is_positive_when_enhanced_is_closer_to_clean(self) -> None:
        clean = np.array([1.0, -1.0, 0.5, -0.5])
        noisy = clean + np.array([0.5, -0.5, 0.25, -0.25])
        enhanced = clean + np.array([0.1, -0.1, 0.05, -0.05])

        improvement = snr_improvement_db(clean, noisy, enhanced)

        self.assertGreater(improvement, 0.0)

    def test_si_sdr_for_identical_signals_is_finite_and_high(self) -> None:
        clean = np.array([1.0, -1.0, 0.5, -0.5])

        score = si_sdr_db(clean, clean)

        self.assertTrue(np.isfinite(score))
        self.assertGreater(score, 90.0)


if __name__ == "__main__":
    unittest.main()
