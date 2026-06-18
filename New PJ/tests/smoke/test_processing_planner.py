"""Smoke tests for the deterministic processing planner."""

from __future__ import annotations

from src.planner.processing_planner import (
    ACTION_ANALYZE_ONLY,
    ACTION_MANUAL_REQUIRED,
    ACTION_NO_PROCESS,
    ACTION_RUN_TASK,
    AUTO,
    BLOCK_ENGINE_UNAVAILABLE,
    BLOCK_INVALID_DURATION,
    BLOCK_INVALID_MEDIA,
    BLOCK_ROUTER_NOT_TRUSTED,
    BLOCK_TARGET_NOISE_SUPPRESSION_MANUAL_ONLY,
    EXTRACT_VOCALS_GOAL,
    IMPROVE_SPEECH_CLARITY,
    REDUCE_TARGET_NOISE,
    WARNING_INPUT_TOO_SHORT,
    WARNING_ROUTER_GOAL_MISMATCH,
    WARNING_SILENCE_OR_NEAR_SILENCE,
    WARNING_TARGET_SUPPRESSOR_EXPERIMENTAL,
    ProcessingCapabilities,
    ProcessingFacts,
    RouterEvidence,
    plan_processing,
)
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS


def test_improve_speech_clarity_without_router_runs_clean_voice() -> None:
    plan = plan_processing(
        IMPROVE_SPEECH_CLARITY,
        ProcessingFacts(input_type="audio"),
    )

    assert plan.action == ACTION_RUN_TASK
    assert plan.recommended_task == CLEAN_VOICE
    assert plan.algorithm == "DeepFilterNet"
    assert plan.blocked_reasons == []


def test_extract_vocals_router_speech_mismatch_warns_without_override() -> None:
    plan = plan_processing(
        EXTRACT_VOCALS_GOAL,
        ProcessingFacts(input_type="audio"),
        RouterEvidence(
            router_status="enabled",
            predicted_label="speech_clean",
            confidence=0.98,
            accepted=True,
        ),
    )

    assert plan.action == ACTION_RUN_TASK
    assert plan.recommended_task == EXTRACT_VOCALS
    assert plan.algorithm == "Demucs"
    assert WARNING_ROUTER_GOAL_MISMATCH in plan.warnings


def test_extract_vocals_target_noise_mismatch_warns_without_override() -> None:
    plan = plan_processing(
        EXTRACT_VOCALS_GOAL,
        ProcessingFacts(input_type="audio"),
        RouterEvidence(
            router_status="enabled",
            predicted_label="speech_target_noise",
            confidence=0.98,
            accepted=True,
        ),
    )

    assert plan.action == ACTION_RUN_TASK
    assert plan.recommended_task == EXTRACT_VOCALS
    assert WARNING_ROUTER_GOAL_MISMATCH in plan.warnings


def test_improve_speech_clarity_environment_mismatch_warns() -> None:
    plan = plan_processing(
        IMPROVE_SPEECH_CLARITY,
        ProcessingFacts(input_type="video"),
        RouterEvidence(
            router_status="enabled",
            predicted_label="environment_only",
            accepted=True,
        ),
    )

    assert plan.action == ACTION_RUN_TASK
    assert plan.recommended_task == CLEAN_VOICE
    assert WARNING_ROUTER_GOAL_MISMATCH in plan.warnings


def test_reduce_target_noise_remains_manual_only_for_matching_router_label() -> None:
    plan = plan_processing(
        REDUCE_TARGET_NOISE,
        ProcessingFacts(input_type="audio"),
        RouterEvidence(
            router_status="enabled",
            predicted_label="speech_target_noise",
            accepted=True,
            route_target="manual_required",
        ),
    )

    assert plan.action == ACTION_MANUAL_REQUIRED
    assert plan.recommended_task is None
    assert BLOCK_TARGET_NOISE_SUPPRESSION_MANUAL_ONLY in plan.blocked_reasons
    assert WARNING_TARGET_SUPPRESSOR_EXPERIMENTAL in plan.warnings
    assert CLEAN_VOICE in plan.alternatives


def test_auto_with_disabled_router_requires_manual_selection() -> None:
    plan = plan_processing(
        AUTO,
        ProcessingFacts(input_type="audio"),
        RouterEvidence(router_status="disabled"),
    )

    assert plan.action == ACTION_MANUAL_REQUIRED
    assert plan.blocked_reasons == [BLOCK_ROUTER_NOT_TRUSTED]


def test_auto_target_noise_uses_safe_clean_voice_fallback() -> None:
    plan = plan_processing(
        AUTO,
        ProcessingFacts(input_type="audio"),
        RouterEvidence(
            router_status="enabled",
            predicted_label="speech_target_noise",
            accepted=True,
        ),
    )

    assert plan.action == ACTION_RUN_TASK
    assert plan.recommended_task == CLEAN_VOICE
    assert plan.algorithm == "DeepFilterNet"
    assert WARNING_TARGET_SUPPRESSOR_EXPERIMENTAL in plan.warnings
    assert plan.blocked_reasons == []


def test_auto_clean_speech_does_not_process() -> None:
    plan = plan_processing(
        AUTO,
        ProcessingFacts(input_type="audio"),
        RouterEvidence(
            router_status="enabled",
            predicted_label="speech_clean",
            accepted=True,
        ),
    )

    assert plan.action == ACTION_NO_PROCESS
    assert plan.recommended_task is None
    assert plan.blocked_reasons == []


def test_auto_noisy_speech_runs_clean_voice() -> None:
    plan = plan_processing(
        AUTO,
        ProcessingFacts(input_type="audio"),
        RouterEvidence(
            router_status="enabled",
            predicted_label="speech_noisy_general",
            accepted=True,
        ),
    )

    assert plan.action == ACTION_RUN_TASK
    assert plan.recommended_task == CLEAN_VOICE
    assert plan.engine_family == "speech_enhancement"
    assert plan.algorithm == "DeepFilterNet"
    assert "restored" in plan.expected_outputs


def test_auto_noisy_speech_without_clean_voice_requires_manual_selection() -> None:
    plan = plan_processing(
        AUTO,
        ProcessingFacts(input_type="audio"),
        RouterEvidence(
            router_status="enabled",
            predicted_label="speech_noisy_general",
            accepted=True,
        ),
        ProcessingCapabilities(clean_voice_available=False),
    )

    assert plan.action == ACTION_MANUAL_REQUIRED
    assert plan.recommended_task is None
    assert plan.blocked_reasons == [BLOCK_ENGINE_UNAVAILABLE]


def test_auto_target_noise_without_clean_voice_requires_manual_selection() -> None:
    plan = plan_processing(
        AUTO,
        ProcessingFacts(input_type="audio"),
        RouterEvidence(
            router_status="enabled",
            predicted_label="speech_target_noise",
            accepted=True,
        ),
        ProcessingCapabilities(clean_voice_available=False),
    )

    assert plan.action == ACTION_MANUAL_REQUIRED
    assert plan.recommended_task is None
    assert plan.blocked_reasons == [BLOCK_ENGINE_UNAVAILABLE]
    assert WARNING_TARGET_SUPPRESSOR_EXPERIMENTAL in plan.warnings


def test_invalid_media_is_a_hard_block() -> None:
    plan = plan_processing(
        IMPROVE_SPEECH_CLARITY,
        ProcessingFacts(input_type="unknown", is_valid_media=False),
    )

    assert plan.action == ACTION_MANUAL_REQUIRED
    assert plan.blocked_reasons == [BLOCK_INVALID_MEDIA]


def test_invalid_duration_is_a_hard_block() -> None:
    plan = plan_processing(
        IMPROVE_SPEECH_CLARITY,
        ProcessingFacts(input_type="audio", duration_sec=0),
    )

    assert plan.action == ACTION_MANUAL_REQUIRED
    assert plan.blocked_reasons == [BLOCK_INVALID_DURATION]


def test_short_input_is_analyzed_without_processing() -> None:
    plan = plan_processing(
        IMPROVE_SPEECH_CLARITY,
        ProcessingFacts(input_type="audio", duration_sec=0.2),
    )

    assert plan.action == ACTION_ANALYZE_ONLY
    assert plan.recommended_task is None
    assert plan.warnings == [WARNING_INPUT_TOO_SHORT]
    assert "analysis_evidence" in plan.expected_outputs


def test_near_silence_warns_and_analyzes_without_processing() -> None:
    plan = plan_processing(
        IMPROVE_SPEECH_CLARITY,
        ProcessingFacts(input_type="audio", rms_energy=1e-7),
    )

    assert plan.action == ACTION_ANALYZE_ONLY
    assert plan.recommended_task is None
    assert plan.warnings == [WARNING_SILENCE_OR_NEAR_SILENCE]


def test_unavailable_clean_voice_engine_requires_manual_selection() -> None:
    plan = plan_processing(
        IMPROVE_SPEECH_CLARITY,
        ProcessingFacts(input_type="audio"),
        capabilities=ProcessingCapabilities(clean_voice_available=False),
    )

    assert plan.action == ACTION_MANUAL_REQUIRED
    assert plan.recommended_task is None
    assert plan.blocked_reasons == [BLOCK_ENGINE_UNAVAILABLE]


def test_router_not_accepted_does_not_block_goal_driven_processing() -> None:
    plan = plan_processing(
        EXTRACT_VOCALS_GOAL,
        ProcessingFacts(input_type="audio"),
        RouterEvidence(
            router_status="enabled",
            predicted_label="speech_clean",
            accepted=False,
        ),
    )

    assert plan.action == ACTION_RUN_TASK
    assert plan.recommended_task == EXTRACT_VOCALS
    assert WARNING_ROUTER_GOAL_MISMATCH not in plan.warnings
