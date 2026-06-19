import { useMemo, useState } from "react";

const GOALS = [
  ["auto", "Auto detect / not sure"],
  ["improve_speech_clarity", "Improve speech clarity"],
  ["extract_vocals", "Extract vocals"],
  ["remove_vocals", "Remove vocals"],
  ["reduce_target_noise", "Target noise reduction (experimental)"],
  ["analyze_only", "Analyze only"],
];

const TASKS = [
  ["clean_voice", "Clean voice"],
  ["extract_vocals", "Extract vocals"],
  ["remove_vocals", "Remove vocals"],
  ["target_noise_suppression", "Target noise suppression (manual-only)"],
];

const BADGES = {
  run_task: "RUN TASK",
  manual_required: "MANUAL REQUIRED",
  analyze_only: "ANALYZE ONLY",
  no_process: "NO PROCESS",
};

const DECISION_TITLES = {
  manual_required: "Needs manual review",
  analyze_only: "Analysis only",
  no_process: "No processing needed",
};

async function readJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed.");
  }
  return payload;
}

function SafetyNotes({ controller }) {
  const items = [
    ...(controller?.warnings || []),
    ...(controller?.blocked_reasons || []),
    ...(controller?.alternatives || []).map((item) => `Alternative: ${item}`),
  ];
  return items.length ? (
    <div className="safety-list">
      {items.map((item) => (
        <span key={item}>{item}</span>
      ))}
    </div>
  ) : (
    <span className="muted">No safety flags.</span>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [goal, setGoal] = useState("auto");
  const [manualTask, setManualTask] = useState("clean_voice");
  const [analysis, setAnalysis] = useState(null);
  const [runResult, setRunResult] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const controller = analysis?.controller;
  const isExperimental = useMemo(
    () =>
      goal === "reduce_target_noise" ||
      controller?.warnings?.some((item) => item.includes("target_suppressor")),
    [goal, controller],
  );

  async function analyze() {
    if (!file) {
      setError("Choose an audio or video file first.");
      return;
    }
    setBusy("analyze");
    setError("");
    setRunResult(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("goal", goal);
      const payload = await readJson(
        await fetch("/api/analyze", { method: "POST", body: form }),
      );
      setAnalysis(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy("");
    }
  }

  async function runTask(task, runMode) {
    if (!analysis?.file_id) {
      setError("Analyze the file before running a task.");
      return;
    }
    setBusy(runMode);
    setError("");
    try {
      const payload = await readJson(
        await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_id: analysis.file_id, task }),
        }),
      );
      setRunResult(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy("");
    }
  }

  const decisionTitle = useMemo(() => {
    if (!controller) return "WAITING FOR ANALYSIS";
    if (controller.decision === "run_task") {
      return {
        clean_voice: "Ready to clean speech",
        extract_vocals: "Ready to extract vocals",
        remove_vocals: "Ready to remove vocals",
      }[controller.recommended_task] || "Ready to process";
    }
    return DECISION_TITLES[controller.decision] || "Controller decision";
  }, [controller]);

  const modeLabel = goal === "auto"
    ? "Automatic recommendation mode"
    : "Directed goal verification";

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">AUDIO CONTROL / SAFE ROUTING</p>
          <h1>Target-Aware Audio Processing Controller</h1>
          <p className="subtitle">
            Upload audio or video, choose a goal, and let the controller recommend a safe path.
          </p>
        </div>
        <div className="system-status">
          <span className="pulse" />
          Local processing
        </div>
      </header>

      <div className="workspace">
        <aside className="control-card">
          <div className="section-heading">
            <span>01</span>
            <div>
              <h2>Control input</h2>
              <p>Choose the source and intended outcome.</p>
            </div>
          </div>

          <label className="upload-zone">
            <input
              type="file"
              accept="audio/*,video/*"
              onChange={(event) => {
                setFile(event.target.files?.[0] || null);
                setAnalysis(null);
                setRunResult(null);
                setError("");
              }}
            />
            <span className="upload-icon">↥</span>
            <strong>{file ? file.name : "Drop or choose a media file"}</strong>
            <small>{file ? `${(file.size / 1048576).toFixed(2)} MB` : "Audio and video supported"}</small>
          </label>

          <label className="field">
            <span>Processing goal</span>
            <select
              value={goal}
              onChange={(event) => {
                setGoal(event.target.value);
                setAnalysis(null);
                setRunResult(null);
                setError("");
              }}
            >
              {GOALS.map(([value, label]) => (
                <option value={value} key={value}>{label}</option>
              ))}
            </select>
          </label>

          <div className="button-row">
            <button className="secondary" onClick={analyze} disabled={Boolean(busy)}>
              {busy === "analyze" ? "Analyzing…" : "Analyze"}
            </button>
            <button
              className="primary"
              onClick={() => runTask(controller.recommended_task, "recommended")}
              disabled={Boolean(busy) || !controller?.recommended_task}
            >
              {busy === "recommended" ? "Running…" : "Run recommended task"}
            </button>
          </div>

          <details className="advanced-controls">
            <summary>Advanced</summary>
            <div className="advanced-content">
              <label className="field">
                <span>Manual task override</span>
                <select
                  value={manualTask}
                  onChange={(event) => setManualTask(event.target.value)}
                >
                  {TASKS.map(([value, label]) => (
                    <option value={value} key={value}>{label}</option>
                  ))}
                </select>
              </label>
              <button
                className="manual-button"
                onClick={() => runTask(manualTask, "manual")}
                disabled={Boolean(busy) || !analysis}
              >
                {busy === "manual" ? "Running…" : "Run manual override"}
              </button>
            </div>
          </details>

          {error && <div className="error-box">{error}</div>}
        </aside>

        <section className="main-panel">
          <div className={`decision-card ${isExperimental ? "experimental" : ""}`}>
            <div className="decision-topline">
              <div>
                <p className="eyebrow">CONTROLLER DECISION</p>
                <span className="mode-label">{modeLabel}</span>
                <h2>{decisionTitle}</h2>
              </div>
              {controller && (
                <span className={`badge badge-${controller.decision}`}>
                  {BADGES[controller.decision] || controller.decision}
                </span>
              )}
            </div>

            {controller ? (
              <>
                <div className="decision-grid">
                  <div>
                    <span>Recommended next step</span>
                    <strong>{controller.recommended_task || "None"}</strong>
                  </div>
                  <div>
                    <span>Engine / algorithm</span>
                    <strong>
                      {[controller.engine_family, controller.algorithm].filter(Boolean).join(" / ") || "None"}
                    </strong>
                  </div>
                </div>
                <div className="why">
                  <span>Why</span>
                  <p>{controller.why}</p>
                </div>
                <div>
                  <span className="label">Safety notes</span>
                  <SafetyNotes controller={controller} />
                </div>
              </>
            ) : (
              <div className="empty-state">
                <div className="wave">▂▄▆█▆▄▂</div>
                <p>Analyze a file to see the safe processing recommendation.</p>
              </div>
            )}
          </div>

          {analysis && (
            <details className="evidence-card">
              <summary>
                <span>Detailed evidence</span>
                <small>Router and signal features</small>
              </summary>
              <div className="evidence-grid">
                <div>
                  <h3>Router</h3>
                  <dl>
                    <dt>Status</dt><dd>{analysis.router.status}</dd>
                    <dt>Label</dt><dd>{analysis.router.predicted_label || "n/a"}</dd>
                    <dt>Confidence</dt><dd>{analysis.router.confidence ?? "n/a"}</dd>
                    <dt>Accepted</dt><dd>{analysis.router.accepted ? "yes" : "no"}</dd>
                  </dl>
                </div>
                <div>
                  <h3>Signal features</h3>
                  <dl>
                    {Object.entries(analysis.features).map(([key, value]) => (
                      <div className="feature-row" key={key}>
                        <dt>{key.replaceAll("_", " ")}</dt>
                        <dd>{value || "n/a"}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </div>
            </details>
          )}

          {runResult && (
            <section className={`output-card ${runResult.status === "blocked" ? "blocked" : ""}`}>
              <div>
                <p className="eyebrow">PROCESSING OUTPUT</p>
                <h2>{runResult.status === "blocked" ? "Run blocked safely" : "Output ready"}</h2>
                <p>{runResult.error || runResult.controller?.why}</p>
              </div>
              {runResult.download_url && (
                <div className="media-output">
                  <audio controls src={runResult.download_url} />
                  <a href={runResult.download_url} download>Download processed file</a>
                </div>
              )}
            </section>
          )}
        </section>
      </div>
    </main>
  );
}
