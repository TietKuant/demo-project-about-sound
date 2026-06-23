"""Smoke tests for rule-based task recommendations."""

from __future__ import annotations

import inspect
import unittest

import src.router.recommendation_router as recommendation_router
from src.analyzer.audio_analysis import AudioAnalysis
from src.router.recommendation_router import (
    TARGET_SOUND_REMOVAL_NOT_SUPPORTED,
    UNSUPPORTED_OR_UNCERTAIN_INPUT,
    recommend_task,
)
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS


class RecommendationRouterSmokeTests(unittest.TestCase):
    def test_speech_heavy_analysis_recommends_clean_voice(self) -> None:
        recommendation = recommend_task(
            AudioAnalysis(
                speech_score=0.78,
                music_score=0.22,
                noise_score=0.60,
                event_labels=[],
            )
        )

        self.assertEqual(recommendation.recommended_task, CLEAN_VOICE)
        self.assertEqual(recommendation.confidence, 0.78)
        self.assertEqual(recommendation.reason, "speech_score_is_dominant")
        self.assertEqual(recommendation.warnings, [])

    def test_music_heavy_analysis_recommends_extract_vocals(self) -> None:
        recommendation = recommend_task(
            AudioAnalysis(
                speech_score=0.40,
                music_score=0.82,
                noise_score=0.20,
                event_labels=[],
            )
        )

        self.assertEqual(recommendation.recommended_task, EXTRACT_VOCALS)
        self.assertEqual(recommendation.confidence, 0.82)
        self.assertEqual(recommendation.reason, "music_score_is_dominant")
        self.assertEqual(recommendation.warnings, [])

    def test_uncertain_analysis_returns_no_task_and_warning(self) -> None:
        recommendation = recommend_task(
            AudioAnalysis(
                speech_score=0.32,
                music_score=0.41,
                noise_score=0.47,
                event_labels=[],
            )
        )

        self.assertIsNone(recommendation.recommended_task)
        self.assertLessEqual(recommendation.confidence, 0.5)
        self.assertEqual(recommendation.reason, "no_supported_task_confidently_matched")
        self.assertIn(UNSUPPORTED_OR_UNCERTAIN_INPUT, recommendation.warnings)

    def test_target_sound_event_labels_add_unsupported_warning(self) -> None:
        for label in ("dog", "bark", "keyboard", "alarm", "siren"):
            with self.subTest(label=label):
                recommendation = recommend_task(
                    AudioAnalysis(
                        speech_score=0.75,
                        music_score=0.20,
                        noise_score=0.35,
                        event_labels=[label],
                    )
                )

                self.assertEqual(recommendation.recommended_task, CLEAN_VOICE)
                self.assertIn(TARGET_SOUND_REMOVAL_NOT_SUPPORTED, recommendation.warnings)

    def test_recommendation_router_does_not_import_or_run_engines(self) -> None:
        source = inspect.getsource(recommendation_router)

        self.assertNotIn("src.engine", source)
        self.assertNotIn("run_pipeline", source)
        self.assertFalse(hasattr(recommendation_router, "DemucsCliEngine"))


if __name__ == "__main__":
    unittest.main()
