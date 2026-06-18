"""Deterministic decision-table planner for goal-driven and auto processing."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.router.task_registry import CLEAN_VOICE, EXTRACT_VOCALS, REMOVE_VOCALS


ANALYZE_ONLY = "analyze_only"
AUTO = "auto"
IMPROVE_SPEECH_CLARITY = "improve_speech_clarity"
EXTRACT_VOCALS_GOAL = "extract_vocals"
REMOVE_VOCALS_GOAL = "remove_vocals"
REDUCE_TARGET_NOISE = "reduce_target_noise"

MODE_GOAL_DRIVEN = "goal_driven"
MODE_AUTO_RECOMMENDATION = "auto_recommendation"
MODE_ANALYZE_ONLY = "analyze_only"

ACTION_RUN_TASK = "run_task"
ACTION_MANUAL_REQUIRED = "manual_required"
ACTION_ANALYZE_ONLY = "analyze_only"
ACTION_NO_PROCESS = "no_process"

WARNING_ROUTER_GOAL_MISMATCH = "router_goal_mismatch"
WARNING_SILENCE_OR_NEAR_SILENCE = "silence_or_near_silence"
WARNING_INPUT_TOO_SHORT = "input_too_short"
WARNING_MANUAL_MUSIC_TASK_SELECTION_REQUIRED = "manual_music_task_selection_required"
WARNING_TARGET_SUPPRESSOR_EXPERIMENTAL = "target_suppressor_experimental"

BLOCK_INVALID_MEDIA = "invalid_media"
BLOCK_INVALID_DURATION = "invalid_duration"
BLOCK_ENGINE_UNAVAILABLE = "engine_unavailable"
BLOCK_ROUTER_NOT_TRUSTED = "router_not_trusted"
BLOCK_TARGET_NOISE_SUPPRESSION_MANUAL_ONLY = "target_noise_suppression_manual_only"
BLOCK_UNKNOWN_ROUTER_LABEL = "unknown_router_label"
BLOCK_UNSUPPORTED_GOAL = "unsupported_goal"

SILENCE_RMS_THRESHOLD = 1e-5


@dataclass(slots=True)
class ProcessingFacts:
    """Factual media evidence used for hard planning guards."""

    input_type: str
    duration_sec: float | None = None
    rms_energy: float | None = None
    is_valid_media: bool = True


@dataclass(slots=True)
class RouterEvidence:
    """Optional advisory or recommendation evidence from the classifier router."""

    router_status: str = "disabled"
    predicted_label: str | None = None
    confidence: float | None = None
    accepted: bool = False
    route_target: str | None = None
    recommended_task: str | None = None
    decision_reason: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProcessingCapabilities:
    """Runtime engine availability."""

    clean_voice_available: bool = True
    extract_vocals_available: bool = True
    remove_vocals_available: bool = True
    target_noise_suppression_available: bool = False


# Short contract names matching the planner inputs described by callers.
Facts = ProcessingFacts
Capabilities = ProcessingCapabilities


@dataclass(slots=True)
class ProcessingPlan:
    """One deterministic processing decision."""

    mode: str
    goal: str
    action: str
    recommended_task: str | None = None
    engine_family: str | None = None
    algorithm: str | None = None
    expected_outputs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class _TaskPolicy:
    task: str
    capability: str
    engine_family: str
    algorithm: str
    expected_outputs: tuple[str, ...]


_GOAL_TASK_POLICIES = {
    IMPROVE_SPEECH_CLARITY: _TaskPolicy(
        task=CLEAN_VOICE,
        capability="clean_voice_available",
        engine_family="speech_enhancement",
        algorithm="DeepFilterNet",
        expected_outputs=("restored",),
    ),
    EXTRACT_VOCALS_GOAL: _TaskPolicy(
        task=EXTRACT_VOCALS,
        capability="extract_vocals_available",
        engine_family="source_separation",
        algorithm="Demucs",
        expected_outputs=("vocals", "no_vocals"),
    ),
    REMOVE_VOCALS_GOAL: _TaskPolicy(
        task=REMOVE_VOCALS,
        capability="remove_vocals_available",
        engine_family="source_separation",
        algorithm="Demucs",
        expected_outputs=("no_vocals", "vocals"),
    ),
}

_GOAL_CONFLICT_LABELS = {
    IMPROVE_SPEECH_CLARITY: {"environment_only", "music_with_vocals"},
    EXTRACT_VOCALS_GOAL: {
        "environment_only",
        "speech_clean",
        "speech_target_noise",
        "speech_noise",
        "speech_noisy_general",
    },
    REMOVE_VOCALS_GOAL: {
        "environment_only",
        "speech_clean",
        "speech_target_noise",
        "speech_noise",
        "speech_noisy_general",
    },
}


def plan_processing(
    goal: str,
    facts: ProcessingFacts,
    router_evidence: RouterEvidence | None = None,
    capabilities: ProcessingCapabilities | None = None,
) -> ProcessingPlan:
    """Plan processing from user intent, hard facts, router evidence, and capabilities."""
    router = router_evidence or RouterEvidence()
    available = capabilities or ProcessingCapabilities()
    mode = _mode_for_goal(goal)

    hard_guard = _hard_fact_guard(goal, facts, mode)
    if hard_guard is not None:
        return hard_guard

    if facts.duration_sec is not None and facts.duration_sec < 0.3:
        return ProcessingPlan(
            mode=MODE_ANALYZE_ONLY,
            goal=goal,
            action=ACTION_ANALYZE_ONLY,
            warnings=[WARNING_INPUT_TOO_SHORT],
            expected_outputs=["analysis_evidence"],
            explanation="The input is too short for a reliable processing decision.",
        )

    if facts.rms_energy is not None and facts.rms_energy < SILENCE_RMS_THRESHOLD:
        return ProcessingPlan(
            mode=MODE_ANALYZE_ONLY,
            goal=goal,
            action=ACTION_ANALYZE_ONLY,
            warnings=[WARNING_SILENCE_OR_NEAR_SILENCE],
            expected_outputs=["analysis_evidence"],
            explanation="Near-silent input is retained for analysis instead of automatic processing.",
        )

    if goal == ANALYZE_ONLY:
        return ProcessingPlan(
            mode=MODE_ANALYZE_ONLY,
            goal=goal,
            action=ACTION_ANALYZE_ONLY,
            expected_outputs=["analysis_evidence"],
            explanation="The selected goal requests analysis without processing.",
        )
    if goal == AUTO:
        return _plan_auto(goal, router, available)
    if goal == REDUCE_TARGET_NOISE:
        warnings = _router_warnings(router)
        _append_unique(warnings, WARNING_TARGET_SUPPRESSOR_EXPERIMENTAL)
        return ProcessingPlan(
            mode=MODE_GOAL_DRIVEN,
            goal=goal,
            action=ACTION_MANUAL_REQUIRED,
            warnings=warnings,
            blocked_reasons=[BLOCK_TARGET_NOISE_SUPPRESSION_MANUAL_ONLY],
            alternatives=[CLEAN_VOICE] if available.clean_voice_available else [],
            explanation=(
                "The target suppressor is experimental and manual-only; "
                "clean_voice is the safer automatic alternative when appropriate."
            ),
        )
    if goal in _GOAL_TASK_POLICIES:
        return _plan_goal_task(goal, router, available)
    return ProcessingPlan(
        mode=MODE_GOAL_DRIVEN,
        goal=goal,
        action=ACTION_MANUAL_REQUIRED,
        blocked_reasons=[BLOCK_UNSUPPORTED_GOAL],
        explanation="The requested processing goal is unsupported.",
    )


def _mode_for_goal(goal: str) -> str:
    if goal == AUTO:
        return MODE_AUTO_RECOMMENDATION
    if goal == ANALYZE_ONLY:
        return MODE_ANALYZE_ONLY
    return MODE_GOAL_DRIVEN


def _hard_fact_guard(goal: str, facts: ProcessingFacts, mode: str) -> ProcessingPlan | None:
    if not facts.is_valid_media:
        return ProcessingPlan(
            mode=mode,
            goal=goal,
            action=ACTION_MANUAL_REQUIRED,
            blocked_reasons=[BLOCK_INVALID_MEDIA],
            explanation="The input failed media validation.",
        )
    if facts.duration_sec is not None and facts.duration_sec <= 0:
        return ProcessingPlan(
            mode=mode,
            goal=goal,
            action=ACTION_MANUAL_REQUIRED,
            blocked_reasons=[BLOCK_INVALID_DURATION],
            explanation="The input duration must be greater than zero.",
        )
    return None


def _plan_goal_task(
    goal: str,
    router: RouterEvidence,
    capabilities: ProcessingCapabilities,
) -> ProcessingPlan:
    policy = _GOAL_TASK_POLICIES[goal]
    warnings = _router_warnings(router)
    if (
        router.router_status == "enabled"
        and router.accepted
        and router.predicted_label in _GOAL_CONFLICT_LABELS[goal]
    ):
        _append_unique(warnings, WARNING_ROUTER_GOAL_MISMATCH)

    if not getattr(capabilities, policy.capability):
        return ProcessingPlan(
            mode=MODE_GOAL_DRIVEN,
            goal=goal,
            action=ACTION_MANUAL_REQUIRED,
            warnings=warnings,
            blocked_reasons=[BLOCK_ENGINE_UNAVAILABLE],
            explanation=f"The required {policy.task} engine is unavailable.",
        )
    return ProcessingPlan(
        mode=MODE_GOAL_DRIVEN,
        goal=goal,
        action=ACTION_RUN_TASK,
        recommended_task=policy.task,
        engine_family=policy.engine_family,
        algorithm=policy.algorithm,
        expected_outputs=list(policy.expected_outputs),
        warnings=warnings,
        explanation=f"The user-selected goal maps to the available {policy.algorithm} processing path.",
    )


def _plan_auto(
    goal: str,
    router: RouterEvidence,
    capabilities: ProcessingCapabilities,
) -> ProcessingPlan:
    warnings = _router_warnings(router)
    if router.router_status != "enabled" or not router.accepted:
        return ProcessingPlan(
            mode=MODE_AUTO_RECOMMENDATION,
            goal=goal,
            action=ACTION_MANUAL_REQUIRED,
            warnings=warnings,
            blocked_reasons=[BLOCK_ROUTER_NOT_TRUSTED],
            explanation="Auto mode requires accepted evidence from an enabled router.",
        )

    label = router.predicted_label
    if label == "speech_clean":
        return ProcessingPlan(
            mode=MODE_AUTO_RECOMMENDATION,
            goal=goal,
            action=ACTION_NO_PROCESS,
            warnings=warnings,
            explanation="The accepted router evidence indicates clean speech.",
        )
    if label == "environment_only":
        return ProcessingPlan(
            mode=MODE_AUTO_RECOMMENDATION,
            goal=goal,
            action=ACTION_NO_PROCESS,
            warnings=warnings,
            explanation="Environment-only input is outside automatic processing scope.",
        )
    if label == "speech_target_noise":
        _append_unique(warnings, WARNING_TARGET_SUPPRESSOR_EXPERIMENTAL)
        return _plan_auto_clean_voice(
            goal=goal,
            capabilities=capabilities,
            warnings=warnings,
            explanation=(
                "Target-specific suppression is experimental, so accepted target-noise speech "
                "falls back to clean_voice rather than the target suppressor."
            ),
        )
    if label in {"speech_noise", "speech_noisy_general"}:
        return _plan_auto_clean_voice(
            goal=goal,
            capabilities=capabilities,
            warnings=warnings,
            explanation="The accepted router evidence indicates noisy speech suitable for clean_voice.",
        )
    if label == "music_with_vocals":
        _append_unique(warnings, WARNING_MANUAL_MUSIC_TASK_SELECTION_REQUIRED)
        return ProcessingPlan(
            mode=MODE_AUTO_RECOMMENDATION,
            goal=goal,
            action=ACTION_MANUAL_REQUIRED,
            warnings=warnings,
            alternatives=[
                task
                for task, is_available in (
                    (EXTRACT_VOCALS, capabilities.extract_vocals_available),
                    (REMOVE_VOCALS, capabilities.remove_vocals_available),
                )
                if is_available
            ],
            explanation="Music processing requires manual selection between vocal extraction and removal.",
        )
    return ProcessingPlan(
        mode=MODE_AUTO_RECOMMENDATION,
        goal=goal,
        action=ACTION_MANUAL_REQUIRED,
        warnings=warnings,
        blocked_reasons=[BLOCK_UNKNOWN_ROUTER_LABEL],
        explanation="The accepted router label has no automatic processing policy.",
    )


def _plan_auto_clean_voice(
    *,
    goal: str,
    capabilities: ProcessingCapabilities,
    warnings: list[str],
    explanation: str,
) -> ProcessingPlan:
    if not capabilities.clean_voice_available:
        return ProcessingPlan(
            mode=MODE_AUTO_RECOMMENDATION,
            goal=goal,
            action=ACTION_MANUAL_REQUIRED,
            warnings=warnings,
            blocked_reasons=[BLOCK_ENGINE_UNAVAILABLE],
            explanation="The recommended clean_voice engine is unavailable.",
        )
    return ProcessingPlan(
        mode=MODE_AUTO_RECOMMENDATION,
        goal=goal,
        action=ACTION_RUN_TASK,
        recommended_task=CLEAN_VOICE,
        engine_family="speech_enhancement",
        algorithm="DeepFilterNet",
        expected_outputs=["restored"],
        warnings=warnings,
        explanation=explanation,
    )


def _router_warnings(router: RouterEvidence) -> list[str]:
    warnings: list[str] = []
    for warning in router.warnings:
        _append_unique(warnings, warning)
    return warnings


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
