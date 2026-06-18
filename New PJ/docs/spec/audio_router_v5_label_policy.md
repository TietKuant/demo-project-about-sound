# Audio Router V5 Label Policy

## Purpose

Audio Router V5 is a calibrated abstaining router. It identifies the likely acoustic content and either suggests a supported processing route or abstains. Labels must describe operationally distinct inputs, not dataset identity.

## Operational Labels

### `speech_clean`

- Meaning: speech is present and no meaningful background-noise problem is evident.
- Route target: `no_process`.
- Engine target: `none`.
- Boundary: use `speech_noisy_general` when background noise materially affects speech clarity.
- Abstain when speech presence or cleanliness is uncertain, the clip is too short, or music competes with speech.

### `speech_noisy_general`

- Meaning: speech plus non-target background noise.
- Examples: fan, cafe, traffic, rain, air conditioner, keyboard, and machinery.
- Route target: `clean_voice`.
- Engine target: `deepfilternet`.
- This label must not include cases focused on project-supported target noises.
- It must not become a general trash bucket for ambiguous inputs.
- Boundary with `speech_clean`: noise must be meaningful enough to justify speech enhancement.
- Boundary with `speech_target_noise`: dog bark, car horn, and siren cases belong to the target-noise label when speech is present and the target cue is supported.
- Synthetic SNR values are nominal amplitude-scaling settings unless RMS normalization and measured post-mix SNR verification are performed.

### `speech_target_noise`

- Meaning: speech plus a project-supported target noise, currently dog bark, car horn, or siren.
- Route target: `target_noise_suppression`.
- Engine target: `target_noise_suppressor`.
- Status: experimental.
- Target noise without speech must not automatically route to target-noise suppression.
- Unsupported or uncertain target noises should use `unknown_or_manual_required`.

### `music_with_vocals`

- Meaning: music mixture with likely singing or vocal content.
- Route target: `manual_required`.
- Engine target: `demucs`.
- The user must decide whether to extract vocals or remove vocals.
- Speech over music, television, or radio is ambiguous and should normally abstain.

### `environment_only`

- Meaning: environmental sound without a supported speech or music processing target.
- Route target: `out_of_scope`.
- Engine target: `none`.
- It must not route into speech enhancement or target-noise suppression.

### `unknown_or_manual_required`

- Meaning: ambiguous, out-of-distribution, very short, silent, corrupt, speech-over-music, unclear-intent, or conflicting input.
- Route target: `manual_required`.
- Engine target: `none`.
- This is the safe default when evidence is insufficient or contradictory.

## Route Mapping

| Label | `route_target` | `engine_target` | Should auto-accept? |
| --- | --- | --- | --- |
| `speech_clean` | `no_process` | `none` | Only with calibrated high confidence and speech-presence support |
| `speech_noisy_general` | `clean_voice` | `deepfilternet` | Only with calibrated high confidence and agreeing safety gates |
| `speech_target_noise` | `target_noise_suppression` | `target_noise_suppressor` | No by default; experimental confirmation required |
| `music_with_vocals` | `manual_required` | `demucs` | No; user goal is required |
| `environment_only` | `out_of_scope` | `none` | Yes only as a no-processing decision |
| `unknown_or_manual_required` | `manual_required` | `none` | No |

## Abstain Policy

The router must abstain or require manual confirmation when:

- confidence is below the calibrated acceptance threshold;
- independent safety gates disagree with the classifier;
- speech, music, or target-noise evidence conflicts;
- the input is silent, too short, corrupt, or out of distribution;
- the route depends on user intent, such as extracting versus removing vocals;
- a target-noise prediction lacks reliable speech-presence evidence.

Abstention is a valid safety outcome, not a classification failure to hide.

## Label Boundaries And Known Ambiguous Cases

- Clean speech with mild room tone may remain `speech_clean`.
- Speech with meaningful fan, traffic, cafe, rain, keyboard, or machinery noise is `speech_noisy_general`.
- Speech with supported dog bark, car horn, or siren evidence may be `speech_target_noise`.
- Dog bark, car horn, or siren without speech is not `speech_target_noise`.
- Singing with instrumental accompaniment is `music_with_vocals`.
- Spoken voice over music, television, or radio should normally be `unknown_or_manual_required`.
- Instrumental music is outside this MVP taxonomy unless a later label is explicitly added.
- Mixed environmental scenes with uncertain speech presence should abstain.

## What This Policy Does Not Claim

- The router does not remove noise.
- The router only suggests, rejects, or abstains from a processing route.
- DeepFilterNet, Demucs, and the custom target-noise suppressor perform waveform processing.
- The taxonomy does not imply universal sound recognition or universal noise removal.
