# Engine Bake-off Log

Goal: choose real engines for Speech Restoration Studio.

Current fixed engine:
- DeepFilterNet = Fast Mode baseline, already integrated.

Candidates:
- LavaSR
- VoiceRestore
- VoiceFixer

Decision rule:
- Only engines that install, run one local WAV, produce playable output, and have acceptable runtime can enter the app.
- Failed installs are documented and rejected or deferred.
- No UI work during bake-off.

## LavaSR spike

Status:
- Install:
- Inference:
- Output:
- Runtime:
- Notes:
- Decision:

## LavaSR spike result

Status:
- Clone: OK
- README inspected: OK
- Install: FAIL
- Import: FAIL
- Output: Not produced
- Runtime: Not measured

Failure details:
- `python -m pip install -e .` failed while building `llvmlite`.
- `from LavaSR.model import LavaEnhance2` failed because `torch` was not installed.
- LavaSR dependency chain includes torch/librosa/soundfile/vocos; local Mac install is not clean yet.

Decision:
- Defer LavaSR.
- Do not spend more time building llvmlite/LLVM in this phase.
- Revisit later through Python 3.10 or ONNX/C++ implementation only if needed.
