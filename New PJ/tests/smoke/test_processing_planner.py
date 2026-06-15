"""Smoke tests for the processing planner."""

from __future__ import annotations

from src.planner.processing_planner import (
    ANALYZE_ONLY,
    BLOCK_CLEAN_VOICE_UNAVAILABLE,
    BLOCK_DEMUCS_UNAVAILABLE,
    BLOCK_TARGET_NOISE_AUDIO_ONLY,
    BLOCK_TARGET_NOISE_CHECKPOINT_MISSING,
    EXTRACT_THEN_CLEAN_VOCALS,
    EXTRACT_VOCALS_GOAL,
    IMPROVE_SPEECH_CLARITY,
    REDUCE_TARGET_NOISE,
    REMOVE_VOCALS_GOAL,
    PlannerCapabilities,
    PlannerEvidence,
    ProcessingPlan,
    WARNING_MUSIC_LIKE_SPEECH_CLEANUP,
    WARNING_VOCAL_SEPARATION_FOR_MUSIC,
    plan_processing,
)
from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS, TARGET_NOISE_SUPPRESSION


def test_music_video_extract_vocals_plans_extract_vocals() -> None:
    plan = plan_processing(
        goal=EXTRACT_VOCALS_GOAL,
        input_type="video",
        evidence=PlannerEvidence(router_label="music", router_confidence=0.80),
        capabilities=PlannerCapabilities(demucs_available=True),
    )

    assert plan.recommended_steps == [EXTRACT_VOCALS]
    assert plan.blocked_reasons == []
    assert "vocals" in plan.expected_outputs


def test_music_video_remove_vocals_plans_remove_vocals() -> None:
    plan = plan_processing(
        goal=REMOVE_VOCALS_GOAL,
        input_type="video",
        evidence=PlannerEvidence(router_label="music", router_confidence=0.80),
        capabilities=PlannerCapabilities(demucs_available=True),
    )

    assert plan.recommended_steps == [REMOVE_VOCALS]
    assert plan.blocked_reasons == []
    assert "no_vocals" in plan.expected_outputs


def test_music_video_improve_speech_clarity_warns_about_distortion() -> None:
    plan = plan_processing(
        goal=IMPROVE_SPEECH_CLARITY,
        input_type="video",
        evidence=PlannerEvidence(router_label="music", router_confidence=0.80),
        capabilities=PlannerCapabilities(clean_voice_available=True),
    )

    assert plan.recommended_steps == [CLEAN_VOICE]
    assert WARNING_MUSIC_LIKE_SPEECH_CLEANUP in plan.warnings
    assert plan.blocked_reasons == []


def test_video_reduce_target_noise_blocks_and_offers_clean_voice() -> None:
    plan = plan_processing(
        goal=REDUCE_TARGET_NOISE,
        input_type="video",
        capabilities=PlannerCapabilities(clean_voice_available=True, target_noise_checkpoint_available=True),
    )

    assert plan.recommended_steps == []
    assert BLOCK_TARGET_NOISE_AUDIO_ONLY in plan.blocked_reasons
    assert [CLEAN_VOICE] in plan.alternatives


def test_audio_reduce_target_noise_with_checkpoint_plans_target_noise_suppression() -> None:
    plan = plan_processing(
        goal=REDUCE_TARGET_NOISE,
        input_type="audio",
        capabilities=PlannerCapabilities(target_noise_checkpoint_available=True),
    )

    assert plan.recommended_steps == [TARGET_NOISE_SUPPRESSION]
    assert plan.blocked_reasons == []
    assert plan.expected_outputs == ["enhanced_audio"]


def test_extract_then_clean_vocals_plans_two_steps_when_available() -> None:
    plan = plan_processing(
        goal=EXTRACT_THEN_CLEAN_VOCALS,
        input_type="audio",
        capabilities=PlannerCapabilities(demucs_available=True, clean_voice_available=True),
    )

    assert plan.recommended_steps == [EXTRACT_VOCALS, CLEAN_VOICE]
    assert plan.blocked_reasons == []
    assert plan.expected_outputs == ["vocals", "cleaned_vocals"]


def test_missing_capabilities_produce_blocked_reasons() -> None:
    plan = plan_processing(
        goal=EXTRACT_THEN_CLEAN_VOCALS,
        input_type="audio",
        capabilities=PlannerCapabilities(demucs_available=False, clean_voice_available=False),
    )

    assert plan.recommended_steps == []
    assert set(plan.blocked_reasons) == {BLOCK_DEMUCS_UNAVAILABLE, BLOCK_CLEAN_VOICE_UNAVAILABLE}


def test_router_evidence_never_blocks_alone() -> None:
    plan = plan_processing(
        goal=EXTRACT_VOCALS_GOAL,
        input_type="audio",
        evidence=PlannerEvidence(router_label="speech_noise", router_confidence=0.99),
        capabilities=PlannerCapabilities(demucs_available=True),
    )

    assert plan.recommended_steps == [EXTRACT_VOCALS]
    assert plan.blocked_reasons == []
    assert WARNING_VOCAL_SEPARATION_FOR_MUSIC in plan.warnings


def test_analyze_only_returns_analysis_output_without_processing_steps() -> None:
    plan = plan_processing(
        goal=ANALYZE_ONLY,
        input_type="audio",
    )

    assert plan.recommended_steps == []
    assert plan.blocked_reasons == []
    assert plan.expected_outputs == ["analysis_evidence"]


def test_unsupported_goal_is_blocked() -> None:
    plan = plan_processing(
        goal="unknown_goal",
        input_type="audio",
    )

    assert plan.recommended_steps == []
    assert plan.blocked_reasons == ["unsupported_goal:unknown_goal"]


def test_unsupported_input_type_is_blocked() -> None:
    plan = plan_processing(
        goal=IMPROVE_SPEECH_CLARITY,
        input_type="document",
    )

    assert plan.recommended_steps == []
    assert plan.blocked_reasons == ["unsupported_input_type:document"]


def test_default_capabilities_do_not_assume_engines_are_available() -> None:
    plan = plan_processing(
        goal=IMPROVE_SPEECH_CLARITY,
        input_type="audio",
    )

    assert plan.recommended_steps == []
    assert plan.blocked_reasons == [BLOCK_CLEAN_VOICE_UNAVAILABLE]


def test_audio_reduce_target_noise_missing_checkpoint_blocks_and_offers_clean_voice() -> None:
    plan = plan_processing(
        goal=REDUCE_TARGET_NOISE,
        input_type="audio",
        capabilities=PlannerCapabilities(clean_voice_available=True, target_noise_checkpoint_available=False),
    )

    assert plan.recommended_steps == []
    assert plan.blocked_reasons == [BLOCK_TARGET_NOISE_CHECKPOINT_MISSING]
    assert plan.alternatives == [[CLEAN_VOICE]]


def test_video_reduce_target_noise_without_clean_voice_has_no_alternative() -> None:
    plan = plan_processing(
        goal=REDUCE_TARGET_NOISE,
        input_type="video",
        capabilities=PlannerCapabilities(clean_voice_available=False, target_noise_checkpoint_available=True),
    )

    assert plan.recommended_steps == []
    assert plan.blocked_reasons == [BLOCK_TARGET_NOISE_AUDIO_ONLY]
    assert plan.alternatives == []


def test_extract_then_clean_with_demucs_only_blocks_clean_voice_and_offers_extract() -> None:
    plan = plan_processing(
        goal=EXTRACT_THEN_CLEAN_VOCALS,
        input_type="audio",
        capabilities=PlannerCapabilities(demucs_available=True, clean_voice_available=False),
    )

    assert plan.recommended_steps == []
    assert plan.blocked_reasons == [BLOCK_CLEAN_VOICE_UNAVAILABLE]
    assert plan.alternatives == [[EXTRACT_VOCALS]]


def test_extract_then_clean_with_clean_voice_only_blocks_demucs_and_offers_clean_voice() -> None:
    plan = plan_processing(
        goal=EXTRACT_THEN_CLEAN_VOCALS,
        input_type="audio",
        capabilities=PlannerCapabilities(demucs_available=False, clean_voice_available=True),
    )

    assert plan.recommended_steps == []
    assert plan.blocked_reasons == [BLOCK_DEMUCS_UNAVAILABLE]
    assert plan.alternatives == [[CLEAN_VOICE]]


def test_same_input_returns_equal_processing_plan() -> None:
    kwargs = {
        "goal": IMPROVE_SPEECH_CLARITY,
        "input_type": "video",
        "evidence": PlannerEvidence(router_label="music", router_confidence=0.88),
        "capabilities": PlannerCapabilities(clean_voice_available=True),
    }

    assert plan_processing(**kwargs) == plan_processing(**kwargs)


def test_plan_blocking_invariants_for_representative_plans() -> None:
    representative_plans: list[ProcessingPlan] = [
        plan_processing(
            goal=IMPROVE_SPEECH_CLARITY,
            input_type="video",
            evidence=PlannerEvidence(router_label="music"),
            capabilities=PlannerCapabilities(clean_voice_available=True),
        ),
        plan_processing(
            goal=REDUCE_TARGET_NOISE,
            input_type="video",
            capabilities=PlannerCapabilities(clean_voice_available=True, target_noise_checkpoint_available=True),
        ),
        plan_processing(
            goal=EXTRACT_VOCALS_GOAL,
            input_type="audio",
            evidence=PlannerEvidence(router_label="speech_noise", router_confidence=0.99),
            capabilities=PlannerCapabilities(demucs_available=True),
        ),
        plan_processing(
            goal=EXTRACT_THEN_CLEAN_VOCALS,
            input_type="audio",
            capabilities=PlannerCapabilities(demucs_available=False, clean_voice_available=True),
        ),
    ]

    for plan in representative_plans:
        if plan.blocked_reasons:
            assert plan.recommended_steps == []
        if plan.recommended_steps:
            assert plan.blocked_reasons == []

    warning_only_plan = representative_plans[2]
    assert warning_only_plan.warnings == [WARNING_VOCAL_SEPARATION_FOR_MUSIC]
    assert warning_only_plan.blocked_reasons == []
    assert warning_only_plan.recommended_steps == [EXTRACT_VOCALS]
