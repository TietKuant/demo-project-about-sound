"""Small rule-based processing planner for multi-step audio/video workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS, TARGET_NOISE_SUPPRESSION


IMPROVE_SPEECH_CLARITY = "improve_speech_clarity"
EXTRACT_VOCALS_GOAL = "extract_vocals"
REMOVE_VOCALS_GOAL = "remove_vocals"
REDUCE_TARGET_NOISE = "reduce_target_noise"
EXTRACT_THEN_CLEAN_VOCALS = "extract_then_clean_vocals"
ANALYZE_ONLY = "analyze_only"

SUPPORTED_GOALS = {
    IMPROVE_SPEECH_CLARITY,
    EXTRACT_VOCALS_GOAL,
    REMOVE_VOCALS_GOAL,
    REDUCE_TARGET_NOISE,
    EXTRACT_THEN_CLEAN_VOCALS,
    ANALYZE_ONLY,
}

MUSIC_LIKE_LABEL = "music"
SPEECH_NOISE_LABEL = "speech_noise"

WARNING_MUSIC_LIKE_SPEECH_CLEANUP = "music_like_input_may_be_distorted_by_speech_cleanup"
WARNING_VOCAL_SEPARATION_FOR_MUSIC = "vocal_separation_is_intended_for_music_mixtures"
BLOCK_CLEAN_VOICE_UNAVAILABLE = "clean_voice_unavailable"
BLOCK_DEMUCS_UNAVAILABLE = "demucs_unavailable"
BLOCK_TARGET_NOISE_AUDIO_ONLY = "target_noise_suppression_audio_only"
BLOCK_TARGET_NOISE_CHECKPOINT_MISSING = "target_noise_checkpoint_missing"


@dataclass(slots=True)
class PlannerEvidence:
    """Advisory analysis evidence for planning; never blocks by itself."""

    router_label: str | None = None
    router_confidence: float | None = None
    music_like: bool = False
    speech_noise_like: bool = False


@dataclass(slots=True)
class PlannerCapabilities:
    """Available runtime capabilities for the planner.

    Defaults are intentionally conservative. Callers should pass detected runtime
    capabilities from the app or CLI instead of relying on implicit availability.
    """

    clean_voice_available: bool = False
    demucs_available: bool = False
    target_noise_checkpoint_available: bool = False


@dataclass(slots=True)
class ProcessingPlan:
    """Planner output for the selected user goal."""

    recommended_steps: list[str] = field(default_factory=list)
    alternatives: list[list[str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)


def plan_processing(
    *,
    goal: str,
    input_type: str,
    evidence: PlannerEvidence | None = None,
    capabilities: PlannerCapabilities | None = None,
) -> ProcessingPlan:
    """Return a conservative plan for one user-facing processing goal."""
    evidence = evidence or PlannerEvidence()
    capabilities = capabilities or PlannerCapabilities()
    plan = ProcessingPlan()

    if goal not in SUPPORTED_GOALS:
        plan.blocked_reasons.append(f"unsupported_goal:{goal}")
        return plan
    if input_type not in {"audio", "video"}:
        plan.blocked_reasons.append(f"unsupported_input_type:{input_type}")
        return plan
    if goal == ANALYZE_ONLY:
        plan.expected_outputs.append("analysis_evidence")
        return plan

    _add_evidence_warnings(goal, evidence, plan)

    if goal == IMPROVE_SPEECH_CLARITY:
        _plan_clean_voice(plan, capabilities)
    elif goal == EXTRACT_VOCALS_GOAL:
        _plan_demucs_step(plan, capabilities, EXTRACT_VOCALS)
    elif goal == REMOVE_VOCALS_GOAL:
        _plan_demucs_step(plan, capabilities, REMOVE_VOCALS)
    elif goal == REDUCE_TARGET_NOISE:
        _plan_target_noise(plan, input_type, capabilities)
    elif goal == EXTRACT_THEN_CLEAN_VOCALS:
        _plan_extract_then_clean(plan, capabilities)

    return plan


def _add_evidence_warnings(goal: str, evidence: PlannerEvidence, plan: ProcessingPlan) -> None:
    music_like = evidence.music_like or evidence.router_label == MUSIC_LIKE_LABEL
    speech_noise_like = evidence.speech_noise_like or evidence.router_label == SPEECH_NOISE_LABEL

    if goal == IMPROVE_SPEECH_CLARITY and music_like:
        plan.warnings.append(WARNING_MUSIC_LIKE_SPEECH_CLEANUP)
    if goal in {EXTRACT_VOCALS_GOAL, REMOVE_VOCALS_GOAL, EXTRACT_THEN_CLEAN_VOCALS} and speech_noise_like:
        plan.warnings.append(WARNING_VOCAL_SEPARATION_FOR_MUSIC)


def _plan_clean_voice(plan: ProcessingPlan, capabilities: PlannerCapabilities) -> None:
    if not capabilities.clean_voice_available:
        plan.blocked_reasons.append(BLOCK_CLEAN_VOICE_UNAVAILABLE)
        return
    plan.recommended_steps.append(CLEAN_VOICE)
    plan.expected_outputs.append("restored_audio_or_video")


def _plan_demucs_step(plan: ProcessingPlan, capabilities: PlannerCapabilities, task: str) -> None:
    if not capabilities.demucs_available:
        plan.blocked_reasons.append(BLOCK_DEMUCS_UNAVAILABLE)
        return
    plan.recommended_steps.append(task)
    if task == EXTRACT_VOCALS:
        plan.expected_outputs.extend(["vocals", "no_vocals"])
    else:
        plan.expected_outputs.extend(["no_vocals", "vocals"])


def _plan_target_noise(plan: ProcessingPlan, input_type: str, capabilities: PlannerCapabilities) -> None:
    if input_type == "video":
        plan.blocked_reasons.append(BLOCK_TARGET_NOISE_AUDIO_ONLY)
        _add_clean_voice_alternative(plan, capabilities)
        return
    if not capabilities.target_noise_checkpoint_available:
        plan.blocked_reasons.append(BLOCK_TARGET_NOISE_CHECKPOINT_MISSING)
        _add_clean_voice_alternative(plan, capabilities)
        return
    plan.recommended_steps.append(TARGET_NOISE_SUPPRESSION)
    plan.expected_outputs.append("enhanced_audio")


def _plan_extract_then_clean(plan: ProcessingPlan, capabilities: PlannerCapabilities) -> None:
    if not capabilities.demucs_available:
        plan.blocked_reasons.append(BLOCK_DEMUCS_UNAVAILABLE)
    if not capabilities.clean_voice_available:
        plan.blocked_reasons.append(BLOCK_CLEAN_VOICE_UNAVAILABLE)
    if plan.blocked_reasons:
        if capabilities.demucs_available:
            plan.alternatives.append([EXTRACT_VOCALS])
        if capabilities.clean_voice_available:
            plan.alternatives.append([CLEAN_VOICE])
        return
    plan.recommended_steps.extend([EXTRACT_VOCALS, CLEAN_VOICE])
    plan.expected_outputs.extend(["vocals", "cleaned_vocals"])


def _add_clean_voice_alternative(plan: ProcessingPlan, capabilities: PlannerCapabilities) -> None:
    if capabilities.clean_voice_available:
        plan.alternatives.append([CLEAN_VOICE])
