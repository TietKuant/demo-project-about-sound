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

function PipelineStrip({ file, analysis, controller, runResult }) {
  const hasOutput = Boolean(runResult?.outputs?.length || runResult?.download_url);
  const stages = [
    ["Upload", Boolean(file)],
    ["Analyze", Boolean(analysis)],
    ["Decide", Boolean(controller)],
    ["Run", Boolean(runResult)],
    ["Output", hasOutput],
  ];
  return (
    <nav className="pipeline-strip" aria-label="Processing pipeline">
      {stages.map(([label, active], index) => (
        <div className={`pipeline-stage ${active ? "active" : ""}`} key={label}>
          <span>{index + 1}</span>
          <strong>{label}</strong>
        </div>
      ))}
    </nav>
  );
}

function MediaPlayer({ src, filename }) {
  const isVideo = /\.(mp4|mov|mkv|avi|webm)$/i.test(filename || "");
  return isVideo ? <video controls src={src} /> : <audio controls src={src} />;
}

function SelectedPath({ controller }) {
  const gateBlocked = controller.decision === "manual_required";
  const runsEngine = controller.decision === "run_task";
  const noProcessing = ["no_process", "analyze_only"].includes(controller.decision);
  return (
    <div className="selected-path">
      <div className="path-node active"><small>Source</small><strong>Input</strong></div>
      <span>→</span>
      <div className="path-node active"><small>Signal</small><strong>Evidence</strong></div>
      <span>→</span>
      <div className={`path-node active ${gateBlocked ? "blocked" : ""}`}>
        <small>Safety</small>
        <strong>{gateBlocked ? "Policy block" : "Policy gate"}</strong>
      </div>
      <span>→</span>
      <div className={`path-node ${runsEngine || noProcessing ? "active" : ""} ${noProcessing ? "neutral" : ""}`}>
        <small>Action</small>
        <strong>
          {runsEngine
            ? controller.algorithm || controller.recommended_task
            : noProcessing
              ? "No processing"
              : "Manual review"}
        </strong>
      </div>
    </div>
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

  async function runControllerPlan() {
    if (!analysis?.file_id) {
      setError("Analyze the file before running the controller plan.");
      return;
    }
    setBusy("recommended");
    setError("");
    try {
      const payload = await readJson(
        await fetch("/api/run-plan", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_id: analysis.file_id }),
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
    ? "Automatic recommendation mode — the controller only runs a task when evidence is trusted."
    : "Directed goal verification — the controller checks whether the selected goal maps to a safe engine.";
  const inputUrl = analysis ? `/api/files/${analysis.file_id}?kind=input` : "";

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
              onClick={runControllerPlan}
              disabled={Boolean(busy) || controller?.decision !== "run_task"}
            >
              {busy === "recommended" ? "Running…" : "Run controller plan"}
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
          <PipelineStrip
            file={file}
            analysis={analysis}
            controller={controller}
            runResult={runResult}
          />

          {analysis && (
            <section className="input-preview-card">
              <div className="card-heading">
                <div>
                  <p className="eyebrow">INPUT PREVIEW</p>
                  <h2>{analysis.filename}</h2>
                </div>
                <span className="file-chip">{file?.type || "media"}</span>
              </div>
              <MediaPlayer src={inputUrl} filename={analysis.filename} />
              <div className="fact-row">
                <div><span>Duration</span><strong>{analysis.features.duration_sec || "n/a"} s</strong></div>
                <div><span>RMS</span><strong>{analysis.features.rms_energy || "n/a"}</strong></div>
                <div><span>Centroid</span><strong>{analysis.features.spectral_centroid_hz || "n/a"} Hz</strong></div>
                <div>
                  <span>Router</span>
                  <strong>
                    {analysis.router.predicted_label || "No trusted label"}
                    {analysis.router.confidence != null ? ` · ${Number(analysis.router.confidence).toFixed(2)}` : ""}
                  </strong>
                </div>
              </div>
            </section>
          )}

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
                <div className="plan-label">Selected path</div>
                <SelectedPath controller={controller} />
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
                <div className="proof-card">
                  <strong>What this demo proves</strong>
                  <ul>
                    <li>Safe abstention when confidence is low</li>
                    <li>Goal-directed routing for clear user intent</li>
                    <li>Expert execution through DeepFilterNet and Demucs</li>
                    <li>Experimental target suppression is blocked by policy</li>
                  </ul>
                </div>
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
              {runResult.status === "blocked" ? (
                <div className="policy-block">
                  <p className="eyebrow">POLICY GATE</p>
                  <h2>Run blocked safely</h2>
                  <p>{runResult.controller?.why || "The selected task is not permitted for automatic execution."}</p>
                  <SafetyNotes controller={runResult.controller} />
                </div>
              ) : runResult.status === "no_process" ? (
                <div className="policy-block no-process">
                  <p className="eyebrow">CONTROLLER PLAN</p>
                  <h2>No processing needed</h2>
                  <p>{runResult.controller?.why}</p>
                </div>
              ) : (
                <>
                  <div className="output-heading">
                    <p className="eyebrow">PROCESSING OUTPUT</p>
                    <h2>Before and after</h2>
                    <p>{runResult.error || runResult.controller?.why}</p>
                  </div>
                  <div className="comparison-grid">
                    <div className="media-panel">
                      <span>Before</span>
                      <MediaPlayer src={inputUrl} filename={analysis?.filename} />
                    </div>
                    {(runResult.outputs || (
                      runResult.download_url
                        ? [{
                            label: "processed_output",
                            path: runResult.primary_output_path,
                            download_url: runResult.download_url,
                          }]
                        : []
                    )).map((output) => (
                      <div className="media-panel after" key={output.label}>
                        <span>
                          {{
                            vocals: "Vocals",
                            no_vocals: "Instrumental",
                            enhanced_speech: "Enhanced speech",
                          }[output.label] || output.label.replaceAll("_", " ")}
                        </span>
                        <MediaPlayer src={output.download_url} filename={output.path} />
                        <a className="download-button" href={output.download_url} download>
                          Download {output.label.replaceAll("_", " ")}
                        </a>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </section>
          )}
        </section>
      </div>
    </main>
  );
}
