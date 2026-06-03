"""Smoke tests for the lightweight task registry."""

from __future__ import annotations

import unittest

from src.router.task_registry import (
    CLEAN_VOICE,
    EXTRACT_VOCALS,
    REMOVE_VOCALS,
    get_task_spec,
    is_supported_task,
    list_supported_tasks,
)


class TaskRegistrySmokeTests(unittest.TestCase):
    def test_all_supported_tasks_are_listed(self) -> None:
        tasks = list_supported_tasks()
        names = [task.name for task in tasks]

        self.assertEqual(names, [CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS])
        self.assertEqual({task.name for task in tasks}, {CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS})

    def test_get_task_spec_returns_clean_voice_metadata(self) -> None:
        task = get_task_spec(CLEAN_VOICE)

        self.assertEqual(task.name, CLEAN_VOICE)
        self.assertEqual(task.engine, "deepfilternet")
        self.assertEqual(task.output_labels, ["restored"])
        self.assertEqual(task.input_kind, "audio_or_video")

    def test_unsupported_task_raises_value_error(self) -> None:
        self.assertFalse(is_supported_task("transcribe_audio"))

        with self.assertRaisesRegex(ValueError, "Unsupported task: transcribe_audio"):
            get_task_spec("transcribe_audio")

    def test_vocal_tasks_map_to_demucs_with_different_primary_output_order(self) -> None:
        extract = get_task_spec(EXTRACT_VOCALS)
        remove = get_task_spec(REMOVE_VOCALS)

        self.assertTrue(is_supported_task(EXTRACT_VOCALS))
        self.assertTrue(is_supported_task(REMOVE_VOCALS))
        self.assertEqual(extract.engine, "demucs")
        self.assertEqual(remove.engine, "demucs")
        self.assertEqual(extract.output_labels, ["vocals", "no_vocals"])
        self.assertEqual(remove.output_labels, ["no_vocals", "vocals"])
        self.assertNotEqual(extract.output_labels[0], remove.output_labels[0])

    def test_task_specs_are_returned_as_independent_objects(self) -> None:
        task = get_task_spec(EXTRACT_VOCALS)
        task.output_labels.append("mutated")

        fresh_task = get_task_spec(EXTRACT_VOCALS)
        self.assertEqual(fresh_task.output_labels, ["vocals", "no_vocals"])


if __name__ == "__main__":
    unittest.main()
