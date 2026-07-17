import "./styles.css";

import { HistoryChart } from "./chart";
import type { ExperimentSnapshot } from "./protocol";
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
const lesionToggle = required<HTMLButtonElement>("#lesion-toggle");
const lesionRadius = required<HTMLInputElement>("#lesion-radius");
const lesionRadiusValue = required<HTMLOutputElement>("#lesion-radius-value");
const edgeThreshold = required<HTMLInputElement>("#edge-threshold");
const edgeThresholdValue = required<HTMLOutputElement>("#edge-threshold-value");
const inspector = required<HTMLElement>("#inspector");
const experimentSelect = required<HTMLSelectElement>("#experiment-select");
const speedSelect = required<HTMLSelectElement>("#speed-select");

let currentSnapshot: ExperimentSnapshot | null = null;
let playing = true;
let lesionArmed = false;
let lastLesionAt = 0;

const socket = new ExperimentSocket(
  (snapshot) => receiveSnapshot(snapshot),
  (status) => {
    connection.className = `connection ${status}`;
    connection.querySelector("span:last-child")!.textContent = status;
  },
  (message) => {
    connection.className = "connection disconnected";
    connection.querySelector("span:last-child")!.textContent = message;
  },
);

const renderer = new DishRenderer(host, (x, y, painting) => {
  if (lesionArmed) {
    const now = performance.now();
    if (!painting || now - lastLesionAt > 70) {
      lastLesionAt = now;
      socket.send({ type: "lesion", x, y, radius: Number(lesionRadius.value) });
    }
    return;
  }
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
socket.connect();

playPause.addEventListener("click", () => {
  playing = !playing;
  playPause.textContent = playing ? "Pause" : "Play";
  socket.send({ type: playing ? "play" : "pause" });
});
required<HTMLButtonElement>("#step").addEventListener("click", () => socket.send({ type: "step" }));
required<HTMLButtonElement>("#reset").addEventListener("click", () => socket.send({ type: "reset" }));
required<HTMLButtonElement>("#stim-a").addEventListener("click", () =>
  socket.send({ type: "stimulate", region: "sensor_a" }),
);
required<HTMLButtonElement>("#stim-b").addEventListener("click", () =>
  socket.send({ type: "stimulate", region: "sensor_b" }),
);
required<HTMLButtonElement>("#reward").addEventListener("click", () =>
  socket.send({ type: "reward", amount: 1 }),
);
required<HTMLButtonElement>("#evaluate").addEventListener("click", () =>
  socket.send({ type: "evaluate", batches: 8 }),
);
required<HTMLButtonElement>("#rewire").addEventListener("click", () =>
  socket.send({ type: "rewire" }),
);
experimentSelect.addEventListener("change", () => {
  const name = experimentSelect.value as ExperimentSnapshot["experiment"];
  connection.querySelector("span:last-child")!.textContent = name === "mnist" ? "loading MNIST…" : "switching…";
  socket.send({ type: "experiment", name });
});
speedSelect.addEventListener("change", (event) => {
  socket.send({ type: "speed", steps: Number((event.currentTarget as HTMLSelectElement).value) });
});
required<HTMLSelectElement>("#layer-select").addEventListener("change", (event) => {
  renderer.setLayer((event.currentTarget as HTMLSelectElement).value as FieldLayer);
});

edgeThreshold.addEventListener("input", () => {
  edgeThresholdValue.value = Number(edgeThreshold.value).toFixed(2);
  renderer.setEdgeThreshold(Number(edgeThreshold.value));
});
lesionRadius.addEventListener("input", () => {
  lesionRadiusValue.value = `${Number(lesionRadius.value).toFixed(1)} cells`;
});
lesionToggle.addEventListener("click", () => {
  lesionArmed = !lesionArmed;
  lesionToggle.setAttribute("aria-pressed", String(lesionArmed));
  lesionToggle.textContent = lesionArmed ? "Armed" : "Arm";
  host.classList.toggle("lesion-armed", lesionArmed);
});
window.addEventListener("beforeunload", () => socket.close());

function receiveSnapshot(snapshot: ExperimentSnapshot): void {
  const switched = currentSnapshot?.experiment !== snapshot.experiment;
  currentSnapshot = snapshot;
  connection.className = "connection connected";
  connection.querySelector("span:last-child")!.textContent = "connected";
  experimentSelect.value = snapshot.experiment;
  host.setAttribute(
    "aria-label",
    `Interactive ${snapshot.field.width} by ${snapshot.field.height} cellular field`,
  );
  if (switched) speedSelect.value = snapshot.experiment === "mnist" ? "1" : "2";
  renderer.render(snapshot);
  history.update(snapshot);
  text("#metric-tick", snapshot.tick.toLocaleString());
  text("#metric-objective-label", snapshot.experiment === "mnist" ? "loss" : "reward");
  text(
    "#metric-reward",
    snapshot.experiment === "mnist"
      ? (snapshot.metrics.loss ?? 0).toFixed(3)
      : snapshot.metrics.rollingReward.toFixed(3),
  );
  text("#metric-accuracy", `${Math.round(snapshot.task.accuracy * 100)}%`);
  text("#metric-edges", snapshot.metrics.edgeCount.toLocaleString());
  text("#metric-device", snapshot.metrics.device);
  text("#metric-cells", snapshot.metrics.livingCells.toLocaleString());
  text("#metric-weight", snapshot.metrics.meanWeight.toFixed(3));
  if (snapshot.task.kind === "mnist") updateMnistTask(snapshot.task);
  else updateXorTask(snapshot.task);
}

function updateXorTask(task: Extract<ExperimentSnapshot["task"], { kind: "xor" }>): void {
  text("#task-name", "delayed xor");
  text("#task-phase", task.phase[0]!.toUpperCase() + task.phase.slice(1));
  text("#phase-badge", task.phase);
  required<HTMLElement>("#phase-badge").dataset.phase = task.phase;
  setTaskLabels("A", "B", "target", "guess");
  text("#task-a", String(task.bitA));
  text("#task-b", String(task.bitB));
  text("#task-target", String(task.target));
  text("#task-prediction", task.prediction === null ? "—" : String(task.prediction));
  required<HTMLElement>("#xor-controls").hidden = false;
  required<HTMLElement>("#mnist-controls").hidden = true;
  required<HTMLElement>("#mnist-preview-panel").hidden = true;
  text("#chart-title", "Reward + accuracy");
  text("#chart-objective-label", "reward");
}

function updateMnistTask(task: Extract<ExperimentSnapshot["task"], { kind: "mnist" }>): void {
  text("#task-name", "mnist self-assembling cells");
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
    `assembly ${task.assemblyStep}/${task.assemblySteps} · route ${task.routingRound} · ${task.seenExamples.toLocaleString()} seen`,
  );
  required<HTMLElement>("#xor-controls").hidden = true;
  required<HTMLElement>("#mnist-controls").hidden = false;
  required<HTMLElement>("#mnist-preview-panel").hidden = false;
  text("#chart-title", "Loss objective + accuracy");
  text("#chart-objective-label", "loss objective");
  drawDigit(task.image);
}

function mnistStageLabel(task: Extract<ExperimentSnapshot["task"], { kind: "mnist" }>): string {
  if (task.phase === "seed") return "Seed — empty graph";
  if (task.phase === "sensing") return `Sense patch row ${task.tokenRow + 1}/7`;
  if (task.phase === "developing") return `Develop — step ${task.assemblyStep}/${task.assemblySteps}`;
  return `Readout ${task.target} → ${task.prediction}`;
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
  const cell = snapshot.field.cells[index];
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
    <span>sensory ${value("reward_trace").toFixed(3)}</span>
    <span>axon request ${value("axon_growth").toFixed(3)}</span>
    <span>receptor ${value("dendrite_growth").toFixed(3)}</span>
  `;
}

function text(selector: string, value: string): void {
  required<HTMLElement>(selector).textContent = value;
}
