import "./styles.css";

import { HistoryChart } from "./chart";
import { LaboratoryView } from "./lab";
import type {
  ExperimentSnapshot,
  HyperparameterSnapshot,
  MnistTaskSnapshot,
  SequenceTaskSnapshot,
} from "./protocol";
import { DishRenderer, type FieldLayer } from "./renderer";
import { ExperimentSocket } from "./socket";

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required element: ${selector}`);
  return element;
}

const host = required<HTMLElement>("#dish-host");
const connection = required<HTMLElement>("#connection-status");
const playPause = required<HTMLButtonElement>("#play-pause");
const edgeThreshold = required<HTMLInputElement>("#edge-threshold");
const edgeThresholdValue = required<HTMLOutputElement>("#edge-threshold-value");
const inspector = required<HTMLElement>("#inspector");
const speedSelect = required<HTMLSelectElement>("#speed-select");
const hyperparameterControls = required<HTMLElement>("#hyperparameter-controls");
const hyperparameterApply = required<HTMLButtonElement>("#hyperparameter-apply");
const hyperparameterStatus = required<HTMLOutputElement>("#hyperparameter-status");
const experimentSelect = required<HTMLSelectElement>("#experiment-select");
const generationPrompt = required<HTMLTextAreaElement>("#generation-prompt");
const fastTraining = required<HTMLButtonElement>("#fast-training");
const trainingStatus = required<HTMLOutputElement>("#training-status");
const sequenceTrainingControls = required<HTMLElement>("#sequence-training-controls");
const savedOrganismSelect = required<HTMLSelectElement>("#saved-organism-select");
const savedOrganismLoad = required<HTMLButtonElement>("#saved-organism-load");
const savedOrganismStatus = required<HTMLOutputElement>("#saved-organism-status");

let currentSnapshot: ExperimentSnapshot | null = null;
let playing = true;
let configurationSignature = "";
let currentExperiment: ExperimentSnapshot["experiment"] | null = null;
let pendingTrainingMode: boolean | null = null;
let pendingPlayback: boolean | null = null;
let pendingExperiment: ExperimentSnapshot["experiment"] | null = null;
let pendingSpeed: number | null = null;
let lastControlRevision = -1;
let cadenceKind: ExperimentSnapshot["task"]["kind"] | null = null;
let savedOrganismSignature = "";
let pendingSavedOrganism: string | null = null;
const pendingHyperparameters = new Map<string, number>();

const socket = new ExperimentSocket(
  (snapshot) => receiveSnapshot(snapshot),
  (status) => {
    if (status === "connecting") lastControlRevision = -1;
    if (status !== "connected") {
      pendingPlayback = null;
      playPause.disabled = false;
      pendingExperiment = null;
      experimentSelect.disabled = false;
      pendingSpeed = null;
      pendingSavedOrganism = null;
      savedOrganismLoad.disabled = savedOrganismSelect.options.length === 0;
    }
    connection.className = `connection ${status}`;
    connection.querySelector("span:last-child")!.textContent = status;
  },
  (message) => {
    connection.className = "connection disconnected";
    connection.querySelector("span:last-child")!.textContent = message;
    if (pendingHyperparameters.size) {
      hyperparameterStatus.value = message;
      hyperparameterApply.disabled = false;
    }
    if (pendingSavedOrganism !== null) savedOrganismStatus.value = message;
    pendingTrainingMode = null;
    fastTraining.disabled = false;
    pendingPlayback = null;
    playPause.disabled = false;
    pendingExperiment = null;
    experimentSelect.disabled = false;
    pendingSpeed = null;
    pendingSavedOrganism = null;
    savedOrganismSelect.disabled = savedOrganismSelect.options.length === 0;
    savedOrganismLoad.disabled = savedOrganismSelect.options.length === 0;
  },
);

const renderer = new DishRenderer(host, (x, y, painting) => {
  if (!painting && currentSnapshot) {
    const cellX = Math.floor(x);
    const cellY = Math.floor(y);
    const index = cellY * currentSnapshot.field.width + cellX;
    renderer.selectCell(index);
    showCell(index, currentSnapshot);
  }
});

const history = new HistoryChart(
  required<SVGPolylineElement>("#reward-line"),
  required<SVGPolylineElement>("#accuracy-line"),
  required<SVGSVGElement>("#history-chart"),
);

await renderer.initialize();
const laboratoryView = new LaboratoryView();
laboratoryView.start();
socket.connect();

playPause.addEventListener("click", () => {
  if (pendingPlayback !== null) return;
  pendingPlayback = !playing;
  playing = pendingPlayback;
  playPause.textContent = playing ? "Pause" : "Play";
  playPause.disabled = true;
  socket.send({ type: playing ? "play" : "pause" });
});
required<HTMLButtonElement>("#step").addEventListener("click", () => socket.send({ type: "step" }));
required<HTMLButtonElement>("#reset").addEventListener("click", () => socket.send({ type: "reset" }));
required<HTMLButtonElement>("#evaluate").addEventListener("click", () =>
  socket.send({ type: "evaluate", batches: 8 }),
);
required<HTMLButtonElement>("#lifecycle-cycle").addEventListener("click", () =>
  socket.send({ type: "lifecycle" }),
);
fastTraining.addEventListener("click", () => {
  const enabled = currentSnapshot?.runtime.mode !== "headless";
  pendingTrainingMode = enabled;
  fastTraining.disabled = true;
  trainingStatus.value = "finishing current update…";
  socket.send({ type: "training", enabled });
});
required<HTMLButtonElement>("#prompt-apply").addEventListener("click", () => {
  pendingPlayback = false;
  playing = false;
  playPause.textContent = "Play";
  playPause.disabled = true;
  socket.send({ type: "prompt", text: generationPrompt.value });
});
required<HTMLButtonElement>("#generate-token").addEventListener("click", () => {
  pendingPlayback = false;
  playing = false;
  playPause.textContent = "Play";
  playPause.disabled = true;
  socket.send({ type: "generate" });
});
experimentSelect.addEventListener("change", () => {
  pendingExperiment = experimentSelect.value as ExperimentSnapshot["experiment"];
  experimentSelect.disabled = true;
  socket.send({
    type: "experiment",
    name: pendingExperiment,
  });
});
savedOrganismLoad.addEventListener("click", () => {
  const organism = savedOrganismSelect.value;
  if (!organism || pendingSavedOrganism !== null) return;
  pendingSavedOrganism = organism;
  savedOrganismSelect.disabled = true;
  savedOrganismLoad.disabled = true;
  savedOrganismStatus.value = "loading checkpoint…";
  socket.send({ type: "load", organism });
});
hyperparameterControls.addEventListener("input", (event) => {
  const input = event.target;
  if (!(input instanceof HTMLInputElement) || input.type !== "range") return;
  const key = input.dataset.key;
  if (!key) return;
  const choices = input.dataset.choices ? JSON.parse(input.dataset.choices) as number[] : null;
  const value = choices ? (choices[Number(input.value)] ?? choices[0]!) : Number(input.value);
  pendingHyperparameters.set(key, value);
  const output = input.parentElement?.querySelector("output");
  if (output instanceof HTMLOutputElement) {
    output.value = formatHyperparameter(value, Number(input.step), input.dataset.integer === "true");
  }
  hyperparameterApply.disabled = false;
  hyperparameterStatus.value = `${pendingHyperparameters.size} pending change${pendingHyperparameters.size === 1 ? "" : "s"}`;
});
hyperparameterApply.addEventListener("click", () => {
  if (!pendingHyperparameters.size) return;
  hyperparameterApply.disabled = true;
  hyperparameterStatus.value = "restarting organism…";
  socket.send({ type: "configure", values: Object.fromEntries(pendingHyperparameters) });
});
speedSelect.addEventListener("change", (event) => {
  pendingSpeed = Number((event.currentTarget as HTMLSelectElement).value);
  socket.send({ type: "speed", steps: pendingSpeed });
});
required<HTMLSelectElement>("#layer-select").addEventListener("change", (event) => {
  renderer.setLayer((event.currentTarget as HTMLSelectElement).value as FieldLayer);
});

edgeThreshold.addEventListener("input", () => {
  edgeThresholdValue.value = Number(edgeThreshold.value).toFixed(2);
  renderer.setEdgeThreshold(Number(edgeThreshold.value));
});
window.addEventListener("beforeunload", () => socket.close());

function receiveSnapshot(snapshot: ExperimentSnapshot): void {
  if (snapshot.runtime.controlRevision < lastControlRevision) return;
  lastControlRevision = snapshot.runtime.controlRevision;
  if (snapshot.experiment !== currentExperiment) {
    currentExperiment = snapshot.experiment;
    history.reset();
    configurationSignature = "";
    pendingHyperparameters.clear();
  }
  currentSnapshot = snapshot;
  if (pendingExperiment === snapshot.experiment) {
    pendingExperiment = null;
    experimentSelect.disabled = false;
  }
  experimentSelect.value = pendingExperiment ?? snapshot.experiment;
  updateSavedOrganisms(snapshot.runtime);
  connection.className = "connection connected";
  connection.querySelector("span:last-child")!.textContent = "connected";
  host.setAttribute(
    "aria-label",
    `Interactive ${snapshot.field.width} by ${snapshot.field.height} cellular field`,
  );
  if (pendingPlayback === snapshot.runtime.running) pendingPlayback = null;
  playing = pendingPlayback ?? snapshot.runtime.running;
  playPause.textContent = playing ? "Pause" : "Play";
  playPause.disabled = pendingPlayback !== null;
  const headlessMode = snapshot.runtime.mode === "headless";
  if (pendingTrainingMode === headlessMode) pendingTrainingMode = null;
  fastTraining.disabled = pendingTrainingMode !== null;
  fastTraining.textContent = headlessMode ? "Stop headless" : "Train headless";
  if (speedSelect.disabled !== headlessMode) speedSelect.disabled = headlessMode;
  if (pendingSpeed === snapshot.runtime.stepsPerFrame) pendingSpeed = null;
  if (document.activeElement !== speedSelect) {
    speedSelect.value = String(pendingSpeed ?? snapshot.runtime.stepsPerFrame);
  }
  trainingStatus.value = pendingTrainingMode !== null
    ? "finishing current update…"
    : headlessMode
      ? `${playing ? "running" : "paused"} · traces suspended`
      : computeStatus(snapshot.runtime);
  if (!headlessMode) renderer.render(snapshot);
  history.update(snapshot);
  text("#metric-tick", snapshot.tick.toLocaleString());
  text("#metric-objective-label", "loss");
  text("#metric-reward", (snapshot.metrics.loss ?? 0).toFixed(3));
  text("#metric-accuracy", `${Math.round(snapshot.task.accuracy * 100)}%`);
  text("#metric-edges", snapshot.metrics.edgeCount.toLocaleString());
  text("#metric-device", snapshot.metrics.device);
  text("#metric-compute-time", `${snapshot.runtime.lastComputeSeconds.toFixed(2)} s`);
  text(
    "#metric-throughput",
    snapshot.runtime.trainingUpdatesPerSecond > 0
      ? `${snapshot.runtime.trainingUpdatesPerSecond.toFixed(3)} upd/s · ${snapshot.runtime.trainingExamplesPerSecond.toFixed(2)} seq/s`
      : "—",
  );
  text("#metric-cells", snapshot.metrics.livingCells.toLocaleString());
  text("#metric-weight", snapshot.metrics.meanWeight.toFixed(3));
  text(
    "#metric-synapse-update",
    snapshot.metrics.synapseUpdateRatio === null
      ? "—"
      : `${(snapshot.metrics.synapseUpdateRatio * 100).toFixed(3)}%`,
  );
  if (snapshot.task.kind === "mnist") updateMnistTask(snapshot.task);
  else updateSequenceTask(snapshot.task, snapshot.runtime);
  updateCadenceLabels(snapshot.task.kind);
  text("#metric-learning-phase", snapshot.task.learningPhase);
  text(
    "#metric-lifecycle",
    snapshot.task.lifecycleActive
      ? `active · ${snapshot.task.lifecycleReason}`
      : `locked · ${snapshot.task.lifecycleReason}`,
  );
  text(
    "#metric-homeostasis",
    `${snapshot.metrics.meanEnergy.toFixed(3)} · ${snapshot.metrics.stressedCells.toLocaleString()} stressed`,
  );
  text("#metric-age", `${snapshot.metrics.meanAge.toFixed(1)} trials`);
  text(
    "#metric-turnover",
    `${snapshot.task.cumulativeBirths.toLocaleString()} born · ${snapshot.task.cumulativeDeaths.toLocaleString()} died`,
  );
  text(
    "#metric-death-causes",
    `${snapshot.task.deathCauses.starvation} starved · ${snapshot.task.deathCauses.excitotoxicity} excitotoxic · ${snapshot.task.deathCauses.maintenance} maintenance`,
  );
  text(
    "#metric-homeostasis",
    `${snapshot.metrics.meanEnergy.toFixed(3)} energy · ${snapshot.metrics.stunnedCells.toLocaleString()} stunned · ${snapshot.metrics.meanExcitotoxicDamage.toFixed(3)} damage`,
  );
  const minimumHops = snapshot.metrics.minimumOutputHops;
  const medianHops = snapshot.metrics.medianOutputHops;
  text(
    "#metric-hops",
    minimumHops === null || minimumHops === undefined
      ? "unreachable"
      : `${minimumHops} min · ${medianHops ?? minimumHops} median`,
  );
  text(
    "#metric-reachability",
    `${snapshot.metrics.temporallyReachableOutputs ?? 0}/${snapshot.metrics.reachableOutputs ?? 0}`,
  );
  text("#metric-attention", (snapshot.metrics.meanAttentionEntropy ?? 0).toFixed(3));
  text(
    "#metric-parameters",
    `${(snapshot.metrics.activeParameters ?? 0).toLocaleString()} · ${(snapshot.metrics.parametersPerLivingCell ?? 0).toFixed(1)}/cell`,
  );
  if (snapshot.configuration) renderHyperparameters(snapshot.configuration.parameters);
}

function updateSavedOrganisms(runtime: ExperimentSnapshot["runtime"]): void {
  const signature = runtime.savedOrganisms
    .map((organism) => `${organism.id}:${organism.label}`)
    .join("|");
  if (signature !== savedOrganismSignature) {
    const previous = savedOrganismSelect.value;
    savedOrganismSelect.replaceChildren(
      ...runtime.savedOrganisms.map((organism) => {
        const option = document.createElement("option");
        option.value = organism.id;
        option.textContent = organism.label;
        return option;
      }),
    );
    savedOrganismSignature = signature;
    if (runtime.savedOrganisms.some((organism) => organism.id === previous)) {
      savedOrganismSelect.value = previous;
    }
  }
  if (pendingSavedOrganism === runtime.loadedOrganism) pendingSavedOrganism = null;
  if (runtime.loadedOrganism && document.activeElement !== savedOrganismSelect) {
    savedOrganismSelect.value = runtime.loadedOrganism;
  }
  const empty = runtime.savedOrganisms.length === 0;
  savedOrganismSelect.disabled = empty || pendingSavedOrganism !== null;
  savedOrganismLoad.disabled = empty || pendingSavedOrganism !== null;
  savedOrganismStatus.value = pendingSavedOrganism !== null
    ? "loading checkpoint…"
    : runtime.loadedOrganism
      ? `loaded · ${runtime.loadedOrganism}`
      : empty
        ? "no checkpoints found"
        : `${runtime.savedOrganisms.length} checkpoint${runtime.savedOrganisms.length === 1 ? "" : "s"} available`;
}

function updateMnistTask(task: MnistTaskSnapshot): void {
  required<HTMLElement>("#mnist-preview-panel").hidden = false;
  required<HTMLElement>("#sequence-preview-panel").hidden = true;
  required<HTMLElement>("#generation-panel").hidden = true;
  sequenceTrainingControls.hidden = true;
  text("#task-name", "mnist spatial neural organism");
  const stage = mnistStageLabel(task);
  text("#task-phase", stage);
  text("#phase-badge", task.phase);
  required<HTMLElement>("#phase-badge").dataset.phase = task.phase;
  setTaskLabels("label", "guess", "confidence", "update");
  text("#task-a", String(task.target));
  text("#task-b", String(task.prediction));
  text("#task-target", `${Math.round(task.confidence * 100)}%`);
  text("#task-prediction", task.trainingStep.toLocaleString());
  text(
    "#mnist-test-accuracy",
    task.testAccuracy === null ? "not evaluated" : `${(task.testAccuracy * 100).toFixed(1)}%`,
  );
  text(
    "#mnist-seen",
    `curriculum ${task.curriculumStage}/${task.curriculumStageCount}: ${task.curriculumExamples.toLocaleString()} examples · ${(task.curriculumStageAccuracy * 100).toFixed(1)}%${task.curriculumTargetAccuracy === null ? "" : ` / ${(task.curriculumTargetAccuracy * 100).toFixed(0)}%`} · ${task.seenExamples.toLocaleString()} seen`,
  );
  text(
    "#metric-structure",
    task.learningPhase === "structure"
      ? `plastic · ${task.structureUnlockReason}`
      : `locked · ${task.structureUnlockReason}`,
  );
  text("#chart-title", "Loss objective + accuracy");
  text("#chart-objective-label", "loss objective");
  drawDigit(task.image);
}

function updateSequenceTask(
  task: SequenceTaskSnapshot,
  runtime: ExperimentSnapshot["runtime"],
): void {
  required<HTMLElement>("#mnist-preview-panel").hidden = true;
  required<HTMLElement>("#sequence-preview-panel").hidden = false;
  required<HTMLElement>("#generation-panel").hidden = !task.interactive;
  sequenceTrainingControls.hidden = false;
  text("#task-name", task.title.toLowerCase());
  text(
    "#task-phase",
    runtime.computePhase === "idle"
      ? sequenceStageLabel(task)
      : computePhaseLabel(runtime),
  );
  text("#phase-badge", task.phase);
  required<HTMLElement>("#phase-badge").dataset.phase = task.phase;
  text("#sequence-description", task.description);
  text(
    "#sequence-test-accuracy",
    task.testAccuracy === null ? "not evaluated" : `${(task.testAccuracy * 100).toFixed(1)}%`,
  );
  text("#sequence-perplexity", task.perplexity.toFixed(2));
  if (task.taskKey === "associative_recall") {
    text(
      "#sequence-description",
      `${task.description} Curriculum: ${task.recallPairs}/${task.recallMaxPairs} bindings at ${(task.stageAccuracy * 100).toFixed(0)}% recent accuracy.`,
    );
  }
  if (task.datasetName) {
    const unit = task.datasetTokens > 0
      ? `${task.datasetTokens.toLocaleString()} tokens · ${task.contextLength}-token context${task.tokenizerName ? ` · ${task.tokenizerName}` : ""}`
      : `${task.datasetCharacters.toLocaleString()} characters · ${task.contextLength}-character context`;
    text(
      "#sequence-description",
      `${task.description} ${unit}.`,
    );
  }
  const position = Math.max(0, Math.min(task.position, task.tokens.length - 1));
  setTaskLabels("token", "prediction", "confidence", "update");
  text("#task-a", task.tokens[position] ?? "—");
  text("#task-b", task.predictions[position] ?? "—");
  text("#task-target", `${Math.round(task.confidence * 100)}%`);
  text("#task-prediction", task.trainingStep.toLocaleString());
  const list = required<HTMLOListElement>("#sequence-tokens");
  const windowStart = Math.max(0, position - 12);
  const windowEnd = Math.min(task.tokens.length, windowStart + 25);
  list.replaceChildren(
    ...task.tokens.slice(windowStart, windowEnd).map((token, offset) => {
      const index = windowStart + offset;
      const item = document.createElement("li");
      if (index <= position) item.classList.add("consumed");
      if (index === position) item.classList.add("current");
      item.textContent = visibleToken(token);
      const prediction = document.createElement("em");
      prediction.textContent = `→ ${visibleToken(task.predictions[index] ?? "—")}`;
      item.append(prediction);
      return item;
    }),
  );
  if (task.interactive) {
    if (document.activeElement !== generationPrompt) {
      generationPrompt.value = task.interactivePrompt;
    }
    text("#generation-output", task.interactivePrompt + task.generatedText);
    text("#next-token-prediction", visibleToken(task.nextTokenPrediction));
  }
  text(
    "#metric-structure",
    task.learningPhase === "structure"
      ? `plastic · ${task.structureUnlockReason}`
      : `locked · ${task.structureUnlockReason}`,
  );
  text("#chart-title", "Loss objective + sequence accuracy");
  text("#chart-objective-label", "loss objective");
}

function renderHyperparameters(parameters: HyperparameterSnapshot[]): void {
  const signature = JSON.stringify(parameters.map(({ key, value }) => [key, value]));
  if (signature === configurationSignature) return;
  configurationSignature = signature;
  pendingHyperparameters.clear();
  hyperparameterApply.disabled = true;
  hyperparameterStatus.value = "no pending changes";
  text("#hyperparameter-count", `${parameters.length} parameters`);
  hyperparameterControls.replaceChildren();

  const groups = new Map<string, HyperparameterSnapshot[]>();
  for (const parameter of parameters) {
    const group = groups.get(parameter.group) ?? [];
    group.push(parameter);
    groups.set(parameter.group, group);
  }
  let firstGroup = true;
  for (const [name, group] of groups) {
    const details = document.createElement("details");
    details.className = "hyperparameter-group";
    details.open = firstGroup;
    firstGroup = false;
    const summary = document.createElement("summary");
    summary.textContent = `${name} (${group.length})`;
    details.append(summary);
    const grid = document.createElement("div");
    grid.className = "hyperparameter-grid";
    for (const parameter of group) grid.append(hyperparameterControl(parameter));
    details.append(grid);
    hyperparameterControls.append(details);
  }
}

function hyperparameterControl(parameter: HyperparameterSnapshot): HTMLLabelElement {
  const label = document.createElement("label");
  label.className = "hyperparameter-control";
  const heading = document.createElement("span");
  heading.textContent = parameter.label;
  const output = document.createElement("output");
  output.value = formatHyperparameter(parameter.value, parameter.step, parameter.integer);
  heading.append(output);
  const input = document.createElement("input");
  input.type = "range";
  if (parameter.choices?.length) {
    input.min = "0";
    input.max = String(parameter.choices.length - 1);
    input.step = "1";
    input.value = String(Math.max(0, parameter.choices.indexOf(parameter.value)));
    input.dataset.choices = JSON.stringify(parameter.choices);
  } else {
    input.min = String(parameter.min);
    input.max = String(parameter.max);
    input.step = String(parameter.step);
    input.value = String(parameter.value);
  }
  input.setAttribute("aria-label", parameter.label);
  input.dataset.key = parameter.key;
  input.dataset.integer = String(parameter.integer);
  label.append(heading, input);
  return label;
}

function visibleToken(token: string): string {
  if (token === "\n") return "↵ newline";
  if (token === "\t") return "⇥ tab";
  if (token === " ") return "␠ space";
  if (token.startsWith(" ")) return `␠${token.slice(1)}`;
  return token || "—";
}

function formatHyperparameter(value: number, step: number, integer: boolean): string {
  if (integer) return Math.round(value).toLocaleString();
  if (value === 0) return "0";
  const decimals = Math.max(0, Math.min(6, Math.ceil(-Math.log10(step))));
  return value.toFixed(decimals);
}

function mnistStageLabel(task: MnistTaskSnapshot): string {
  if (task.phase === "input") return "Input — 49 patch neurons";
  if (task.phase === "forward") return `Forward traffic — step ${task.trialStep}/${task.trialSteps}`;
  if (task.phase === "feedback") return `Backward credit ${task.target} → ${task.prediction}`;
  if (task.learningPhase !== "structure") {
    return `Structure locked — ${task.learningPhase} learning phase`;
  }
  return `Lifecycle cycle — ${task.births} born, ${task.deaths} died`;
}

function sequenceStageLabel(task: SequenceTaskSnapshot): string {
  if (task.phase === "token") {
    return `Consume token ${task.position + 1}/${task.tokens.length} — ${task.tokens[task.position] ?? "—"}`;
  }
  if (task.phase === "feedback") return "Gradient credit through the consumed sequence";
  return `Lifecycle cycle — ${task.births} born, ${task.deaths} died`;
}

function computeStatus(runtime: ExperimentSnapshot["runtime"]): string {
  if (runtime.computePhase === "idle") return "measured trace active";
  return computePhaseLabel(runtime).toLowerCase();
}

function computePhaseLabel(runtime: ExperimentSnapshot["runtime"]): string {
  const progress = runtime.computeTotal > 1
    ? ` ${runtime.computeProgress}/${runtime.computeTotal}`
    : "";
  if (runtime.computePhase === "forward") return `Forward — token${progress}`;
  if (runtime.computePhase === "backward") return `Backward credit — token${progress}`;
  if (runtime.computePhase === "optimizer") return "Optimizer — applying gradients";
  if (runtime.computePhase === "credit") return "Local credit — eligibility and homeostasis";
  if (runtime.computePhase === "lifecycle") return "Lifecycle — growth and pruning";
  if (runtime.computePhase === "evaluation") return `Evaluation — batch${progress}`;
  if (runtime.computePhase === "headless") return "Headless optimizer update";
  return "Measured trace active";
}

function updateCadenceLabels(kind: ExperimentSnapshot["task"]["kind"]): void {
  if (kind === cadenceKind) return;
  cadenceKind = kind;
  const sequence = kind === "sequence";
  text("#cadence-title", sequence ? "Trace sampling" : "Simulation speed");
  for (const option of speedSelect.options) {
    const value = Number(option.value);
    option.textContent = sequence
      ? (value === 1 ? "every token" : `every ${value} tokens`)
      : `${value}×`;
  }
}

function setTaskLabels(a: string, b: string, target: string, prediction: string): void {
  text("#task-label-a", a);
  text("#task-label-b", b);
  text("#task-label-target", target);
  text("#task-label-prediction", prediction);
}

function drawDigit(pixels: number[]): void {
  const canvas = required<HTMLCanvasElement>("#mnist-preview");
  const context = canvas.getContext("2d");
  if (!context) return;
  const image = context.createImageData(28, 28);
  for (let index = 0; index < 28 * 28; index += 1) {
    const value = Math.max(0, Math.min(1, pixels[index] ?? 0));
    image.data[index * 4] = Math.round(72 + 92 * value);
    image.data[index * 4 + 1] = Math.round(103 + 140 * value);
    image.data[index * 4 + 2] = Math.round(99 + 132 * value);
    image.data[index * 4 + 3] = Math.round(20 + 235 * value);
  }
  context.clearRect(0, 0, 28, 28);
  context.putImageData(image, 0, 0);
}

function showCell(index: number, snapshot: ExperimentSnapshot): void {
  const row = snapshot.field.indices === null ? index : snapshot.field.indices.indexOf(index);
  const cell = row >= 0 ? snapshot.field.cells[row] : undefined;
  if (!cell) return;
  const value = (channel: string): number => {
    const channelIndex = snapshot.field.channels.indexOf(channel);
    return channelIndex >= 0 ? (cell[channelIndex] ?? 0) : 0;
  };
  const x = index % snapshot.field.width;
  const y = Math.floor(index / snapshot.field.width);
  inspector.innerHTML = `
    <strong>cell ${x}, ${y}</strong>
    <span>activation ${value("activation").toFixed(3)}</span>
    <span>energy ${value("energy").toFixed(3)}</span>
    <span>alive ${value("alive").toFixed(3)}</span>
    <span>stimulation ${value("stimulation").toFixed(4)}</span>
    <span>traffic load ${value("load").toFixed(4)}</span>
    <span>backward credit ${value("credit").toFixed(6)}</span>
    <span>task utility ${value("utility").toFixed(4)}</span>
    <span>genotype magnitude ${value("genotype_norm").toFixed(4)}</span>
    <span>emit gate ${value("emission").toFixed(4)}</span>
    <span>age ${value("age").toFixed(0)} trials</span>
    <span>homeostatic stress ${value("stress").toFixed(3)}</span>
    <span>lineage depth ${value("lineage").toFixed(0)}</span>
    <span>parent site ${value("parent").toFixed(0)}</span>
  `;
}

function text(selector: string, value: string): void {
  required<HTMLElement>(selector).textContent = value;
}
