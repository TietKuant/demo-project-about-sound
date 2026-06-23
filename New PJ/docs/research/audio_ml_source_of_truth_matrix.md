# Audio ML Source-of-Truth Matrix

## 1. Purpose

This document turns audio ML literature into concrete project decisions. It is not a general literature review and not a report chapter.

The system should not add features only because they are technically possible. Every algorithmic choice should be tied to:

- signal assumption
- data requirement
- evaluation metric
- current/future project scope

The project goal is:

audio/video input -> audio analysis/router -> choose suitable processor -> run processor -> output.

## 2. Core rule

Router and processors are different problems.

Router:

- pattern recognition / classification
- target is a content label
- output is label, confidence, probabilities, and routing evidence

Processor:

- signal estimation / regression
- target is a waveform or stem
- output is processed audio

A content label is not automatically the same as "this processor will improve the clip." Routing decisions must combine classifier evidence, task assumptions, confidence, and safety rules.

## 3. Research-to-project matrix

| Claim / concept | Academic basis | Algorithm family | Signal assumption | Required data | Valid metrics | Current project status | MVP / Future / Avoid | Main risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Audio can be analyzed by time-frequency representation | Waveform, framing/windowing, FFT/STFT, spectrogram, mel-spectrogram, MFCC | DSP feature extraction; spectrogram representation | Audio has local time-frequency structure | Audio files with consistent decoding and sample-rate handling | Feature sanity checks; downstream classifier metrics | Current feature extraction exists; spectrogram/mel support is future/needs repo audit | MVP baseline now; mel/spectrogram future | Treating visualization as proof of quality |
| Low-level features can support coarse audio classification | RMS, ZCR, spectral centroid, bandwidth, rolloff, flatness, band-energy ratios, MFCC | Handcrafted feature classifier | Coarse classes differ in energy, temporal, and spectral statistics | Labeled audio clips | Accuracy, macro-F1, per-class F1, confusion matrix | Current lightweight analyzer/router can be defended as a baseline | MVP | Overclaiming fine-grained understanding |
| Global mean features can miss short events | Transient events such as dog bark, car horn, siren may appear in max, percentile, temporal dynamics, not only clip mean | Frame-level statistics; event-aware features | Important evidence may be brief and localized | Clips with event labels and frame-level or clip-level labels | Per-class F1, event recall, confusion matrix | Current features include some frame-summary statistics such as `rms_std`, `zcr_std`, and `silence_ratio`; stronger transient-aware statistics such as max, p95, burst ratio, or event-level summaries are future | MVP improvement candidate | Missing short target events |
| Speech/music/noise routing is classification | Handcrafted features + classical classifier; lightweight feature classifier; possible embeddings | Classifier / router | Content class is inferable from acoustic evidence | Labeled clips: speech/noise, music, environment | Accuracy, macro-F1, per-class F1, confusion matrix, per-source breakdown | Current MLP router is a baseline | MVP, with strict limitations | Accuracy hides source-label confounding |
| Pretrained audio/speech embeddings can improve router with limited data | PANNs, VGGish/AudioSet, HuBERT, WavLM | Frozen embedding + light classifier | Pretrained representations encode general acoustic patterns | Labeled project clips plus pretrained model | Macro-F1, per-class F1, source-disjoint holdout | Not implemented; future upgrade | Future | Dependency size and domain mismatch |
| Transformer audio classifiers are frontier but high-risk | AST, PaSST, BEATs | Transformer classifier | Large models learn richer time-frequency patterns | Larger labeled data or stable pretrained checkpoints | Macro-F1, calibration, robustness tests | Not implemented | Future / avoid for MVP | Too heavy and hard to justify in current app |
| Speech enhancement is not generic audio separation | Speech enhancement assumes observed = speech + noise | Speech enhancement / denoising | Main desired signal is speech | Noisy speech and optionally clean references | SI-SDR, STOI, PESQ, SNR improvement when clean reference exists; listening for real samples | `clean_voice` uses DeepFilterNet | MVP | Applying speech enhancement to music may distort it |
| Music source separation is not speech denoising | Source separation assumes mixture = vocals + accompaniment/stems | Music source separation | Music mixture contains structured sources | Music mixtures and reference stems for evaluation | SDR, SI-SDR, stem-level metrics when references exist | `extract_vocals` and `remove_vocals` use Demucs | MVP | Misusing Demucs as speech denoiser |
| Target-noise suppression is closed-set supervised denoising | Synthetic clean speech + target noise pairs at controlled SNR | Supervised waveform denoising | Input is speech plus supported target-noise classes | Clean speech, labeled target noise, mixed noisy speech, SNR metadata | L1/MSE, SI-SDR/SNR improvement when paired references exist | `target_noise_suppression` exists in the current project as an experimental/manual processor; exact evaluation artifacts should be checked before report claims | Future / experimental manual | Claiming universal noise removal |
| Data imbalance can bias classifier learning | Class imbalance, stratified split, class weighting, oversampling, undersampling | Dataset design and training controls | Training distribution affects learned decision boundary | Balanced labels and source-aware splits | Macro-F1, per-class F1, per-source metrics | Balanced sampling and split controls exist in dataset tooling | MVP data hygiene | Trusting accuracy despite imbalance |
| SMOTE is not native audio augmentation | SMOTE creates feature-space minority samples | Feature-space oversampling | Interpolated feature vectors may not correspond to valid waveforms | Tabular feature matrix | Classifier metrics on validation data | Not implemented | Avoid for raw audio; discussion only | Claiming synthetic audio exists when only feature-space interpolation exists |
| Audio-specific augmentation should preserve signal meaning | Synthetic mixing, SNR variation, SpecAugment for spectrogram models, Mixup with caution | Audio augmentation / representation augmentation | Augmentation should not change the intended label incorrectly | Clean sources, noise sources, labels, augmentation parameters | Classifier/processor metrics on held-out data | Target-noise dataset uses synthetic mixing and SNR control | MVP for synthetic mixing; SpecAugment/Mixup future | Creating unrealistic training examples |
| Router evaluation is label evaluation | Accuracy, macro-F1, per-class F1, confusion matrix, per-source breakdown | Classification evaluation | Ground-truth content labels exist | Labeled router manifest | Accuracy, macro-F1, per-class F1, confusion matrix, source-disjoint breakdown | Router evaluation exists; deeper V4 reporting is future/needs repo audit | MVP/Future | Reporting accuracy alone |
| Processor evaluation needs reference signal when using intrusive metrics | PESQ/STOI/SI-SDR/SDR require clean reference or reference stems | Intrusive signal-quality evaluation | Reference target is available | Clean speech or reference stems | PESQ, STOI, SI-SDR, SDR, L1/MSE depending on task | VoiceBank/benchmark data and synthetic target-noise data can support paired evaluation when clean/reference paths are present; arbitrary uploads do not | MVP rule | Claiming objective improvement on no-reference uploads |
| Real uploaded audio needs non-intrusive or qualitative evaluation | DNSMOS/MOS-style evaluation | Non-intrusive quality estimation; listening protocol | Reference signal is unavailable | Real audio samples, human notes, optional pretrained non-intrusive model | Runtime, RTF, output_exists, listening notes, DNSMOS if implemented | Vietnamese real sample protocol exists; DNSMOS not implemented | Future | Presenting subjective listening as objective metric |
| Voice cloning / singing conversion is representation learning, not current scope | Content/speaker/prosody/F0 disentanglement, speaker embeddings, self-supervised speech representation | Voice conversion, singing voice conversion, acoustic rendering | Model separates and recombines latent audio factors | Paired/unpaired speech/singing data, speaker identity, pitch/prosody data | MOS, speaker similarity, intelligibility, F0/prosody metrics | Not part of current product | Avoid for MVP | Scope creep away from audio processing tasks |

## 4. Current MVP decisions

- Keep router as an evidence layer, not an automatic truth source.
- Keep processors separated by signal-model assumption.
- Do not auto-route `target_noise_suppression` from generic `speech_noise`.
- Treat DeepFilterNet as speech enhancement, not music separation.
- Treat Demucs as music source separation, not speech denoising.
- Use synthetic clean+noise mixing as valid target-noise data creation.
- Use balanced dataset sampling and source-disjoint splits when possible.
- Report limitations when reference signals are unavailable.

## 5. Theory-driven improvement candidates

P0:

- Separate content labels from routing decisions in app explanation.
- Avoid raw "recommended task" without reason.
- Show algorithm assumption and limitation.

P1:

- Router metrics: macro-F1, per-class F1, confusion matrix, per-source/source-disjoint breakdown.
- Transient-aware analyzer features for short target-noise events.
- Stronger abstain policy when confidence is low.

P2:

- Frozen pretrained embeddings + light classifier.
- Better SNR/noise-floor estimation.
- Target-noise event classifier.

Avoid for now:

- Training AST/Transformer from scratch.
- Voice cloning/singing conversion product scope.
- Universal noise remover claim.
- Objective metrics on arbitrary user uploads without reference.

## 6. Voice cloning and singing conversion as a theory case study

Voice cloning, voice conversion, and singing voice conversion work by separating or controlling representations such as content, speaker/timbre, prosody, pitch/F0, rhythm/duration, and acoustic rendering.

This is relevant as a conceptual example of representation learning. It should not become part of the current product scope.

The project lesson is: audio ML must define what representation each model is learning. In this project, the router learns content labels, speech enhancement estimates cleaner speech, source separation estimates stems, and target-noise suppression estimates a cleaner waveform for supported target-noise speech examples.

## 7. Forbidden claims

- The system understands all audio.
- The router always chooses the best processor.
- DeepFilterNet separates vocals from music.
- Demucs denoises speech.
- `target_noise_suppression` handles any background noise.
- Compression improves audio quality.
- Arbitrary uploads can be objectively evaluated with PESQ/STOI/SI-SDR without reference.

## 8. Reference anchor list

- Davis & Mermelstein, MFCC
- Scheirer & Slaney, speech/music discrimination
- DeepFilterNet / DeepFilterNet2 / DeepFilterNet3
- RNNoise
- Demucs / Hybrid Demucs / HT-Demucs
- Spleeter
- Open-Unmix
- VoiceBank+DEMAND
- DNS Challenge
- UrbanSound8K
- ESC-50
- MUSAN
- MUSDB18
- PANNs
- VGGish / AudioSet
- AST
- HuBERT
- WavLM
- ECAPA-TDNN
- SMOTE
- SpecAugment
- Mixup
- SI-SDR
- PESQ
- STOI
- DNSMOS

These are reference anchors, not final bibliography entries. Before using them in the report, verify the exact paper title, venue, year, DOI/arXiv link, and what claim each source actually supports.
