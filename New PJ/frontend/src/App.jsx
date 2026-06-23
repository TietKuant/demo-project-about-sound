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

const STEPS = [
  ["start", "Start"],
  ["analyze", "Analyze"],
  ["results", "Run & results"],
];

const BADGES = {
  run_task: "READY TO RUN",
  manual_required: "REVIEW REQUIRED",
  analyze_only: "ANALYSIS ONLY",
  no_process: "NO PROCESS",
};

async function readJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed.");
  }
  return payload;
}

function displayValue(value) {
  return value == null || value === "" ? "n/a" : String(value);
}

function displayScore(value) {
  return value == null || value === "" || Number.isNaN(Number(value))
    ? "n/a"
    : Number(value).toFixed(3);
}

function formatLabel(value) {
  return displayValue(value).replaceAll("_", " ");
}

function MediaPlayer({ src, filename }) {
  const isVideo = /\.(mp4|mov|mkv|avi|webm)$/i.test(filename || "");
  return isVideo ? <video controls src={src} /> : <audio controls src={src} />;
}

function isPlayableMedia(output) {
  const mediaType = output?.media_type?.toLowerCase() || "";
  if (mediaType.startsWith("audio") || mediaType.startsWith("video")) {
    return true;
  }
  return /\.(wav|mp3|mp4|m4a|flac|ogg)$/i.test(
    output?.path || output?.filename || "",
  );
}

function AppProgress({ step }) {
  const activeIndex = STEPS.findIndex(([value]) => value === step);
  return (
    <nav className="app-progress" aria-label="Application progress">
      {STEPS.map(([value, label], index) => (
        <div
          className={`progress-item ${index <= activeIndex ? "active" : ""}`}
          key={value}
        >
          <span>{index + 1}</span>
          <strong>{label}</strong>
        </div>
      ))}
    </nav>
  );
}

function SafetyNotes({ controller }) {
  const items = [
    ...(controller?.warnings || []),
    ...(controller?.blocked_reasons || []),
    ...(controller?.alternatives || []).map((item) => `Alternative: ${item}`),
  ];
  return items.length ? (
    <div className="tag-list">
      {items.map((item) => (
        <span key={item}>{formatLabel(item)}</span>
      ))}
    </div>
  ) : (
    <span className="muted">No safety flags.</span>
  );
}

function TechnicalDetails({ analysis, runResult }) {
  if (!analysis) return null;
  const payloads = [
    ["Full controller payload", analysis.controller],
    ["Router payload", analysis.router],
    ["Experimental detection fusion payload", analysis.experimental_detection_fusion],
    ["Detection fusion summary", analysis.detection_fusion_summary],
    ["Raw features", analysis.features],
    ["Raw API/run response", runResult],
  ].filter(([, payload]) => payload != null);

  return (
    <details className="technical-details">
      <summary>
        <span>
          <strong>Technical details</strong>
          <small>Controller, router, fusion, features, and raw responses</small>
        </span>
        <span className="details-action">Show payloads</span>
      </summary>
      <div className="technical-grid">
        {payloads.map(([title, payload]) => (
          <section className="payload-card" key={title}>
            <h3>{title}</h3>
            <pre>{JSON.stringify(payload, null, 2)}</pre>
          </section>
        ))}
      </div>
    </details>
  );
}

function OutputItem({ output }) {
  const labels = {
    restored: "Restored audio",
    enhanced_speech: "Enhanced speech",
    vocals: "Vocals",
    no_vocals: "No vocals / instrumental",
    target_noise_report_json: "Target-noise report (JSON)",
    target_noise_report_csv: "Target-noise report (CSV)",
    environment_event_report: "Environment event report",
  };
  const title = labels[output.label] || formatLabel(output.label);
  return (
    <article className="output-item">
      <div className="output-label">
        <span>{isPlayableMedia(output) ? "Audio output" : "Report output"}</span>
        <h3>{title}</h3>
      </div>
      {isPlayableMedia(output) ? (
        <MediaPlayer src={output.download_url} filename={output.path || output.filename} />
      ) : (
        <p className="file-description">
          {output.media_type || "File"} ·{" "}
          {output.path?.split("/").pop() || output.filename || "download"}
        </p>
      )}
      <a className="download-button" href={output.download_url} download>
        Download {title}
      </a>
    </article>
  );
}

export default function App() {
  const [step, setStep] = useState("start");
  const [file, setFile] = useState(null);
  const [goal, setGoal] = useState("auto");
  const [manualTask, setManualTask] = useState("clean_voice");
  const [analysis, setAnalysis] = useState(null);
  const [runResult, setRunResult] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const controller = analysis?.controller;
  const inputUrl = analysis ? `/api/files/${analysis.file_id}?kind=input` : "";
  const isEnvironmentEventAnalysis =
    controller?.workflow_kind === "environment_event_analysis";
  const canRunControllerPlan =
    controller?.decision === "run_task" || isEnvironmentEventAnalysis;
  const recommendedTask = controller?.recommended_task || "";

  const fusionSummary = useMemo(() => {
    if (analysis?.detection_fusion_summary) {
      return analysis.detection_fusion_summary;
    }
    const fusion = analysis?.experimental_detection_fusion;
    if (!fusion) return null;
    return {
      controller: { decision_source: controller?.decision_source || "" },
      fusion_label: fusion.fusion_label || "",
      recommended_workflow: fusion.recommended_workflow || "",
      scores: fusion.detection_scores || {},
      review_recommended: fusion.review_recommended === true,
      review_reasons: fusion.review_reasons || [],
    };
  }, [analysis, controller]);

  const decisionTitle = useMemo(() => {
    if (!controller) return "Waiting for analysis";
    if (controller.decision === "run_task") return "Workflow ready";
    if (controller.decision === "manual_required") return "Review required";
    if (controller.decision === "analyze_only") return "Analysis complete";
    if (controller.decision === "no_process") return "No processing needed";
    return "Controller decision";
  }, [controller]);

  const displayedOutputs = useMemo(() => {
    if (!runResult) return [];
    if (runResult.outputs?.length) return runResult.outputs;
    if (!runResult.download_url) return [];
    const isSpeechWorkflow = ["speech_cleanup", "speech_target_noise_cleanup"].includes(
      runResult.controller?.workflow_kind || controller?.workflow_kind,
    );
    return [{
      label: isSpeechWorkflow ? "enhanced_speech" : "restored",
      path: runResult.primary_output_path,
      download_url: runResult.download_url,
    }];
  }, [runResult, controller]);

  const targetNoiseFallback = (
    runResult?.controller?.workflow_kind || controller?.workflow_kind
  ) === "speech_target_noise_cleanup";

  function resetAnalysis() {
    setAnalysis(null);
    setRunResult(null);
    setError("");
  }

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
      if (payload.controller?.recommended_task) {
        setManualTask(payload.controller.recommended_task);
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy("");
    }
  }

  async function runTask(task) {
    if (!analysis?.file_id) {
      setError("Analyze the file before running a task.");
      return;
    }
    setBusy("manual");
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

  return (
    <main className={`app-shell step-${step}`}>
      {step === "start" ? (
        <section className="landing-page">
          <div className="landing-status">
            <span className="pulse" />
            Local processing
          </div>
          <div className="landing-copy">
            <p className="eyebrow">AUDIO CONTROL / SAFE ROUTING</p>
            <h1>Target-Aware Audio Processing Controller</h1>
            <p>
              Upload audio or video, let the controller analyze it, then run the
              recommended workflow.
            </p>
            <button className="primary landing-button" onClick={() => setStep("analyze")}>
              Start analysis
              <span aria-hidden="true">→</span>
            </button>
          </div>
          <div className="landing-visual" aria-hidden="true">
            <div className="sound-line">▂▄▆█▆▄▂</div>
            <span>Analyze</span>
            <i />
            <span>Decide</span>
            <i />
            <span>Process</span>
          </div>
        </section>
      ) : (
        <div className="application-page">
          <header className="app-header">
            <button className="brand-button" onClick={() => setStep("start")}>
              <span>TA</span>
              <strong>Target-Aware Controller</strong>
            </button>
            <div className="system-status">
              <span className="pulse" />
              Local processing
            </div>
          </header>

          <AppProgress step={step} />

          {step === "analyze" && (
            <section className="work-page">
              <div className="page-intro">
                <p className="eyebrow">STEP 2 / ANALYZE</p>
                <h1>Understand the source before processing.</h1>
                <p>Choose a file and goal. The controller will recommend a safe workflow.</p>
              </div>

              <section className="input-card">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">SOURCE</p>
                    <h2>Audio or video input</h2>
                  </div>
                  {file && <span className="file-chip">{(file.size / 1048576).toFixed(2)} MB</span>}
                </div>
                <div className="input-grid">
                  <label className="upload-zone">
                    <input
                      type="file"
                      accept="audio/*,video/*"
                      onChange={(event) => {
                        setFile(event.target.files?.[0] || null);
                        resetAnalysis();
                      }}
                    />
                    <span className="upload-icon">↥</span>
                    <strong>{file ? file.name : "Choose a media file"}</strong>
                    <small>Audio and video supported</small>
                  </label>
                  <div className="input-actions">
                    <label className="field">
                      <span>Processing goal</span>
                      <select
                        value={goal}
                        onChange={(event) => {
                          setGoal(event.target.value);
                          resetAnalysis();
                        }}
                      >
                        {GOALS.map(([value, label]) => (
                          <option value={value} key={value}>{label}</option>
                        ))}
                      </select>
                    </label>
                    <button
                      className="primary analyze-button"
                      onClick={analyze}
                      disabled={Boolean(busy)}
                    >
                      {busy === "analyze" ? "Analyzing…" : "Analyze"}
                    </button>
                  </div>
                </div>
                {error && <div className="error-box">{error}</div>}
              </section>

              {analysis && (
                <>
                  <section className="analysis-summary">
                    <div className="summary-heading">
                      <div>
                        <p className="eyebrow">ANALYSIS COMPLETE</p>
                        <h2>{analysis.filename}</h2>
                      </div>
                      <span className={`badge badge-${controller?.decision}`}>
                        {BADGES[controller?.decision] || displayValue(controller?.decision)}
                      </span>
                    </div>

                    <div className="decision-cards">
                      <article className="metric-card primary-decision">
                        <span>Primary controller decision</span>
                        <strong>{decisionTitle}</strong>
                        <p>{controller?.why || "No controller explanation was returned."}</p>
                      </article>
                      <article className="metric-card">
                        <span>Recommended task</span>
                        <strong>{formatLabel(recommendedTask || "None")}</strong>
                      </article>
                      <article className="metric-card">
                        <span>Recommended workflow</span>
                        <strong>{formatLabel(
                          fusionSummary?.recommended_workflow ||
                          controller?.workflow_kind ||
                          "None",
                        )}</strong>
                      </article>
                      <article className="metric-card">
                        <span>Decision source</span>
                        <strong>{formatLabel(
                          fusionSummary?.controller?.decision_source ||
                          controller?.decision_source,
                        )}</strong>
                      </article>
                    </div>

                    <section className="fusion-card">
                      <div className="fusion-heading">
                        <div>
                          <p className="eyebrow">DETECTION FUSION SUGGESTION</p>
                          <h3>{formatLabel(fusionSummary?.fusion_label || "Unavailable")}</h3>
                          <p>
                            Suggested workflow:{" "}
                            <strong>{formatLabel(
                              fusionSummary?.recommended_workflow || "Unavailable",
                            )}</strong>
                          </p>
                        </div>
                        <span className={`review-chip ${fusionSummary?.review_recommended ? "warn" : ""}`}>
                          Review {fusionSummary?.review_recommended ? "recommended" : "not required"}
                        </span>
                      </div>
                      <div className="score-grid">
                        {[
                          ["speech_present", "Speech present"],
                          ["music_present", "Music present"],
                          ["target_event_present", "Target event present"],
                        ].map(([key, label]) => (
                          <div key={key}>
                            <span>{label}</span>
                            <strong>{displayScore(fusionSummary?.scores?.[key])}</strong>
                          </div>
                        ))}
                      </div>
                      <div className="review-reasons">
                        <span>Review reasons</span>
                        {fusionSummary?.review_reasons?.length ? (
                          <div className="tag-list">
                            {fusionSummary.review_reasons.map((reason) => (
                              <span key={reason}>{formatLabel(reason)}</span>
                            ))}
                          </div>
                        ) : (
                          <p className="muted">No review reasons reported.</p>
                        )}
                      </div>
                    </section>

                    <div className="analysis-footer">
                      <div>
                        <span className="field-label">Safety and alternatives</span>
                        <SafetyNotes controller={controller} />
                      </div>
                      <button
                        className="primary continue-button"
                        onClick={() => setStep("results")}
                      >
                        Continue to run
                        <span aria-hidden="true">→</span>
                      </button>
                    </div>
                  </section>

                  <TechnicalDetails analysis={analysis} runResult={runResult} />
                </>
              )}
            </section>
          )}

          {step === "results" && (
            <section className="results-page">
              <div className="page-intro results-intro">
                <button className="text-button" onClick={() => setStep("analyze")}>
                  ← Back to analysis
                </button>
                <p className="eyebrow">STEP 3 / RUN & RESULTS</p>
                <h1>Run the selected workflow.</h1>
                <p>Review the controller choice, process the file, and download each output.</p>
              </div>

              <section className="run-card">
                <div className="run-selection">
                  <span>Selected / recommended task</span>
                  <h2>{formatLabel(
                    isEnvironmentEventAnalysis
                      ? "environment event report"
                      : recommendedTask || manualTask,
                  )}</h2>
                  <p>
                    Workflow: <strong>{formatLabel(controller?.workflow_kind)}</strong>
                    {" · "}Source: <strong>{formatLabel(controller?.decision_source)}</strong>
                  </p>
                </div>
                <button
                  className="primary run-button"
                  onClick={runControllerPlan}
                  disabled={Boolean(busy) || !canRunControllerPlan}
                >
                  {busy === "recommended"
                    ? "Running…"
                    : isEnvironmentEventAnalysis
                      ? "Generate environment report"
                      : "Run recommended workflow"}
                </button>
                {!canRunControllerPlan && (
                  <div className="run-blocked">
                    <strong>Automatic run is unavailable.</strong>
                    <p>{controller?.why || "The controller requires review before processing."}</p>
                    <SafetyNotes controller={controller} />
                  </div>
                )}
                <details className="manual-controls">
                  <summary>Manual task override</summary>
                  <div>
                    <label className="field">
                      <span>Task</span>
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
                      className="secondary"
                      onClick={() => runTask(manualTask)}
                      disabled={Boolean(busy)}
                    >
                      {busy === "manual" ? "Running…" : "Run manual task"}
                    </button>
                  </div>
                </details>
                {error && <div className="error-box">{error}</div>}
              </section>

              {runResult && (
                <section className={`results-card ${runResult.status === "blocked" ? "blocked" : ""}`}>
                  {runResult.status === "blocked" ? (
                    <div className="result-message">
                      <p className="eyebrow">POLICY GATE</p>
                      <h2>Run blocked safely</h2>
                      <p>{runResult.controller?.why || "This task cannot be run automatically."}</p>
                      <SafetyNotes controller={runResult.controller} />
                    </div>
                  ) : runResult.status === "no_process" ? (
                    <div className="result-message">
                      <p className="eyebrow">CONTROLLER RESULT</p>
                      <h2>No processing needed</h2>
                      <p>{runResult.controller?.why}</p>
                    </div>
                  ) : (
                    <>
                      <div className="results-heading">
                        <div>
                          <p className="eyebrow">OUTPUTS READY</p>
                          <h2>Processing results</h2>
                          <p>{runResult.error || "Download or preview the generated outputs below."}</p>
                        </div>
                        <span className="success-chip">{displayValue(runResult.status)}</span>
                      </div>

                      {targetNoiseFallback && (
                        <div className="target-noise-notice">
                          <strong>Safe target-noise fallback</strong>
                          <p>
                            Enhanced speech is produced by safe speech enhancement.
                            The target-noise JSON and CSV reports are attached as
                            evidence/report outputs.
                          </p>
                        </div>
                      )}

                      <div className="output-context">
                        <article>
                          <span>Original input</span>
                          <h3>{analysis?.filename}</h3>
                          <MediaPlayer src={inputUrl} filename={analysis?.filename} />
                        </article>
                      </div>
                      <div className="output-grid">
                        {displayedOutputs.map((output) => (
                          <OutputItem output={output} key={output.label} />
                        ))}
                      </div>
                    </>
                  )}
                </section>
              )}

              <TechnicalDetails analysis={analysis} runResult={runResult} />
            </section>
          )}
        </div>
      )}
    </main>
  );
}
