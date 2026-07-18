interface GpuProcess {
  gpuUuid: string;
  pid: number;
  memoryMiB: number;
}

interface GpuSnapshot {
  index: number;
  name: string;
  uuid: string;
  pciBusId: string;
  memoryTotalMiB: number;
  memoryUsedMiB: number;
  utilizationPercent: number;
  powerWatts: number;
  processes: GpuProcess[];
}

interface MetricRecord {
  type: "train" | "held_out" | "diagnostic";
  update: number;
  loss?: number;
  rollingLoss?: number;
  accuracy?: number;
  targetCharactersPerSecond?: number;
  targetTokensPerSecond?: number;
  generation?: number;
  livingCells?: number;
  stunnedCells?: number;
  meanEnergy?: number;
  meanExcitotoxicDamage?: number;
  edgeCount?: number;
  conductingEdgeCount?: number;
  pruneEligibleEdges?: number;
  minimumOutputHops?: number | null;
  medianOutputHops?: number | null;
  reachableOutputs?: number;
  tokenReachableOutputs?: number;
  contextReachableOutputs?: number;
  outputCount?: number;
  cumulativeBirths?: number;
  cumulativeDeaths?: number;
  cumulativeStuns?: number;
  cumulativeRecoveries?: number;
  cumulativeGrownEdges?: number;
  cumulativePrunedEdges?: number;
  generationPrompt?: string;
  generationSample?: string;
  generationUniqueTokenRatio?: number;
}

interface RunSnapshot {
  id: string;
  task: string;
  architecture: string;
  gpuUuid: string | null;
  pid: number | null;
  status: "running" | "checkpointed" | "stopped" | "failed";
  configuration: Record<string, unknown>;
  commit: string | null;
  latestTrain: MetricRecord | null;
  latestHeldOut: MetricRecord | null;
  latestDiagnostics: MetricRecord | null;
  hasCheckpoint: boolean;
}

interface LaboratorySnapshot {
  controlEnabled: boolean;
  capabilities: {
    tasks: string[];
    architectures: string[];
    ampModes: string[];
    lifecycleProfiles: string[];
  };
  gpus: GpuSnapshot[];
  runs: RunSnapshot[];
  benchmarks: BenchmarkSnapshot[];
  timestamp: number;
}

interface BenchmarkCheckpoint {
  update: number;
  trainingAccuracy?: number;
  heldOutAccuracy: number;
  heldOutSlotAccuracy?: number[];
  heldOutPositionAccuracy?: number[];
  heldOutPositionIndices?: number[];
  heldOutPresentedValueRate?: number;
  heldOutDistractorRate?: number;
  heldOutAbsentValueRate?: number;
  loss?: number;
  recallPairs?: number;
  livingCells?: number;
  edgeCount?: number;
  generation?: number;
  cumulativeBirths?: number;
  cumulativeDeaths?: number;
}

interface BenchmarkSnapshot {
  id: string;
  task: string;
  profile: string;
  architecture: string;
  intervention: string | null;
  recallMode: string;
  seed: number | null;
  deterministic: boolean;
  globalRngMatched: boolean;
  device: string | null;
  steps: number | null;
  completedSteps: number;
  status: "running" | "complete" | "failed";
  failureType: string | null;
  failureMessage: string | null;
  seconds: number | null;
  lesionCount: number;
  lesionRadius: number | null;
  parameterCount: number | null;
  trainableParameterCount: number | null;
  cudaAllocatedGiB: number | null;
  peakCudaAllocatedGiB: number | null;
  bindingDiagnostics: {
    distinctOwners: number;
    vocabularySize: number;
    meanAddressEntropy: number;
    meanAddressOverlap: number;
    meanPeakOwnership: number;
  } | null;
  livingCells: number | null;
  edgeCount: number | null;
  minimumOutputHops: number | null;
  temporallyReachableOutputs: number | null;
  contextReachableOutputs: number | null;
  messageSteps: number | null;
  broadcastGain: number | null;
  outputCount: number | null;
  sequenceLength: number | null;
  dependencyTokens: number | null;
  chanceAccuracy: number | null;
  checkpoints: BenchmarkCheckpoint[];
  artifactMtime: number;
}

const SVG_NS = "http://www.w3.org/2000/svg";
const SERIES_CLASSES = ["series-a", "series-b", "series-c", "series-d", "series-e"] as const;

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing laboratory element: ${selector}`);
  return element;
}

export class LaboratoryView {
  private readonly gpuHost = required<HTMLElement>("#laboratory-gpus");
  private readonly runsHost = required<HTMLTableSectionElement>("#laboratory-runs");
  private readonly chart = required<SVGSVGElement>("#laboratory-chart");
  private readonly legend = required<HTMLElement>("#laboratory-chart-legend");
  private readonly diagnosticsHost = required<HTMLTableSectionElement>("#laboratory-diagnostics");
  private readonly status = required<HTMLOutputElement>("#laboratory-status");
  private readonly refreshButton = required<HTMLButtonElement>("#laboratory-refresh");
  private readonly form = required<HTMLFormElement>("#laboratory-launch-form");
  private readonly gpuSelect = required<HTMLSelectElement>("#lab-gpu");
  private readonly taskSelect = required<HTMLSelectElement>("#lab-task");
  private readonly architectureSelect = required<HTMLSelectElement>("#lab-architecture");
  private readonly messageStepsSelect = required<HTMLSelectElement>("#lab-message-steps");
  private readonly lifecycleProfileSelect = required<HTMLSelectElement>("#lab-lifecycle-profile");
  private readonly launchButton = required<HTMLButtonElement>("#lab-launch");
  private readonly launchStatus = required<HTMLOutputElement>("#lab-launch-status");
  private readonly benchmarksHost = required<HTMLTableSectionElement>("#laboratory-benchmarks");
  private readonly benchmarkChart = required<SVGSVGElement>("#benchmark-chart");
  private readonly benchmarkTopologyChart = required<SVGSVGElement>("#benchmark-topology-chart");
  private readonly benchmarkLegend = required<HTMLElement>("#benchmark-chart-legend");
  private readonly benchmarkSummary = required<HTMLOutputElement>("#benchmark-summary");
  private selectedRuns = new Set<string>();
  private histories = new Map<string, MetricRecord[]>();
  private snapshot: LaboratorySnapshot | null = null;
  private timer: number | null = null;
  private refreshing = false;

  public start(): void {
    this.refreshButton.addEventListener("click", () => void this.refresh());
    this.form.addEventListener("submit", (event) => void this.launch(event));
    this.taskSelect.addEventListener("change", () => {
      this.messageStepsSelect.value = this.taskSelect.value === "tiny_stories" ? "12" : "2";
    });
    void this.refresh();
    this.timer = window.setInterval(() => void this.refresh(), 3000);
    window.addEventListener("beforeunload", () => {
      if (this.timer !== null) window.clearInterval(this.timer);
    });
  }

  private async refresh(): Promise<void> {
    if (this.refreshing) return;
    this.refreshing = true;
    this.refreshButton.disabled = true;
    try {
      const response = await fetch("/api/lab", { cache: "no-store" });
      if (!response.ok) throw new Error(`laboratory returned ${response.status}`);
      this.snapshot = await response.json() as LaboratorySnapshot;
      this.renderSnapshot(this.snapshot);
      await this.refreshHistories();
      this.status.value = `measured ${new Date(this.snapshot.timestamp * 1000).toLocaleTimeString()}`;
    } catch (error) {
      this.status.value = error instanceof Error ? error.message : "laboratory unavailable";
    } finally {
      this.refreshButton.disabled = false;
      this.refreshing = false;
    }
  }

  private renderSnapshot(snapshot: LaboratorySnapshot): void {
    this.renderGpus(snapshot.gpus);
    this.renderBenchmarks(snapshot.benchmarks ?? []);
    if (this.selectedRuns.size === 0) {
      const running = snapshot.runs.find((run) => run.status === "running");
      const fallback = running ?? snapshot.runs.at(-1);
      if (fallback) this.selectedRuns.add(fallback.id);
    }
    const known = new Set(snapshot.runs.map((run) => run.id));
    this.selectedRuns = new Set([...this.selectedRuns].filter((id) => known.has(id)));
    this.renderRuns(snapshot.runs, snapshot.gpus);
    this.renderRunDiagnostics(snapshot.runs);
    this.populateSelect(
      this.gpuSelect,
      snapshot.gpus.map((gpu) => ({ value: gpu.uuid, label: gpu.name })),
    );
    this.populateSelect(
      this.taskSelect,
      snapshot.capabilities.tasks.map((name) => ({
        value: name, label: name === "tiny_stories" ? "Token cellular language" : "Tiny Shakespeare",
      })),
    );
    this.populateSelect(
      this.architectureSelect,
      snapshot.capabilities.architectures.map((name) => ({ value: name, label: name.toUpperCase() })),
    );
    this.populateSelect(
      this.lifecycleProfileSelect,
      snapshot.capabilities.lifecycleProfiles.map((name) => ({ value: name, label: name })),
    );
    for (const control of this.form.elements) {
      if (control instanceof HTMLInputElement || control instanceof HTMLSelectElement || control instanceof HTMLButtonElement) {
        control.disabled = !snapshot.controlEnabled;
      }
    }
    this.launchStatus.value = snapshot.controlEnabled
      ? "new runs receive immutable manifests"
      : "launch control disabled on server";
  }

  private renderBenchmarks(benchmarks: BenchmarkSnapshot[]): void {
    const rows = benchmarks.map((benchmark) => {
      const final = benchmark.checkpoints.at(-1);
      const peak = benchmark.checkpoints.length
        ? Math.max(...benchmark.checkpoints.map((checkpoint) => checkpoint.heldOutAccuracy))
        : undefined;
      const tokenControl = [
        "token_routing", "token_memory", "token_context", "token_stream",
      ].includes(benchmark.task);
      const topology = tokenControl
        ? `min ${benchmark.minimumOutputHops ?? "—"} hops · dependency ${benchmark.dependencyTokens ?? 0} tokens · ${benchmark.temporallyReachableOutputs ?? 0}/${benchmark.contextReachableOutputs ?? 0}/${benchmark.outputCount ?? "—"} token/context/graph · ${benchmark.messageSteps ?? "—"}×${benchmark.sequenceLength ?? "—"} ticks`
        : final?.livingCells === undefined
          ? "—"
          : `${final.livingCells} cells · ${final.edgeCount ?? "—"} edges · +${final.cumulativeBirths ?? 0}/−${final.cumulativeDeaths ?? 0}`;
      const row = document.createElement("tr");
      const values = [
        benchmark.id,
        benchmark.intervention ?? "—",
        benchmark.architecture.toUpperCase(),
        benchmark.seed?.toString() ?? "—",
        `${benchmark.completedSteps.toLocaleString()} / ${benchmark.steps?.toLocaleString() ?? "—"}`,
        tokenControl
          ? `${benchmark.messageSteps ?? "—"}×${benchmark.sequenceLength ?? "—"} ticks`
          : final?.recallPairs?.toString() ?? "—",
        this.percent(peak),
        this.percent(final?.heldOutAccuracy),
        final?.heldOutSlotAccuracy?.length
          ? final.heldOutSlotAccuracy.map((accuracy) => this.percent(accuracy)).join(" / ")
          : final?.heldOutPositionAccuracy?.length
            ? final.heldOutPositionAccuracy.map((accuracy, index) => (
                `t${(final.heldOutPositionIndices?.[index] ?? index) + 1} ${this.percent(accuracy)}`
              )).join(" / ")
            : "—",
        benchmark.task !== "associative_recall" || final?.heldOutPresentedValueRate === undefined
          ? "—"
          : `${this.percent(final.heldOutPresentedValueRate)} (${this.percent(final.heldOutDistractorRate)} distractor)`,
        topology,
        benchmark.bindingDiagnostics
          ? `${benchmark.bindingDiagnostics.distinctOwners}/${benchmark.bindingDiagnostics.vocabularySize} · H ${benchmark.bindingDiagnostics.meanAddressEntropy.toFixed(2)} · overlap ${benchmark.bindingDiagnostics.meanAddressOverlap.toFixed(2)}`
          : "—",
        benchmark.peakCudaAllocatedGiB === null ? "—" : `${benchmark.peakCudaAllocatedGiB.toFixed(2)} GiB`,
        benchmark.seconds === null ? "—" : `${benchmark.seconds.toFixed(1)} s`,
        benchmark.status === "failed"
          ? `failed · ${benchmark.failureType ?? "error"}${benchmark.failureMessage ? `: ${benchmark.failureMessage}` : ""}`
          : benchmark.status,
      ];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      return row;
    });
    if (rows.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td colspan="15">No persisted benchmark artifacts found.</td>';
      rows.push(row);
    }
    this.benchmarksHost.replaceChildren(...rows);

    const newest = benchmarks.at(0);
    if (!newest) {
      this.benchmarkSummary.value = "no persisted benchmark artifacts";
      this.drawBenchmarkChart([]);
      this.drawBenchmarkTopologyChart([]);
      return;
    }
    const cohort = benchmarks.filter((benchmark) => (
      benchmark.task === newest.task
      && benchmark.profile === newest.profile
      && benchmark.recallMode === newest.recallMode
      && benchmark.steps === newest.steps
    ));
    const rngStatus = newest.globalRngMatched ? " · branch RNG matched" : "";
    const chance = newest.chanceAccuracy === null
      ? ""
      : ` · chance ${this.percent(newest.chanceAccuracy)}`;
    this.benchmarkSummary.value = `${newest.profile} · ${newest.recallMode.replace("_", " ")} · seed ${newest.seed ?? "—"} · ${newest.deterministic ? "deterministic" : "seeded"}${rngStatus}${chance} · ${newest.steps ?? "—"} updates`;
    const visibleCohort = cohort.slice(0, SERIES_CLASSES.length);
    this.drawBenchmarkChart(visibleCohort);
    this.drawBenchmarkTopologyChart(visibleCohort);
  }

  private drawBenchmarkChart(benchmarks: BenchmarkSnapshot[]): void {
    this.benchmarkChart.replaceChildren();
    this.benchmarkLegend.replaceChildren();
    const measured = benchmarks.filter((benchmark) => benchmark.checkpoints.length > 0);
    if (measured.length === 0) {
      const label = this.svg("text", { x: "360", y: "98", class: "chart-empty" });
      label.textContent = benchmarks.length === 0
        ? "No matched benchmark cohort"
        : "No evaluated checkpoint in matched cohort";
      this.benchmarkChart.append(label);
      return;
    }
    const maxUpdate = Math.max(...measured.flatMap((benchmark) => benchmark.checkpoints.map((checkpoint) => checkpoint.update)));
    const left = 46; const right = 706; const top = 12; const bottom = 164;
    const x = (value: number) => left + value / Math.max(1, maxUpdate) * (right - left);
    const y = (value: number) => bottom - Math.max(0, Math.min(1, value)) * (bottom - top);
    this.benchmarkChart.append(
      this.svg("line", { x1: String(left), y1: String(bottom), x2: String(right), y2: String(bottom), class: "lab-axis" }),
      this.svg("line", { x1: String(left), y1: String(top), x2: String(left), y2: String(bottom), class: "lab-axis" }),
    );
    const chance = measured[0]?.chanceAccuracy;
    if (chance !== null && chance !== undefined) {
      const chanceY = y(chance);
      this.benchmarkChart.append(
        this.svg("line", { x1: String(left), y1: String(chanceY), x2: String(right), y2: String(chanceY), class: "lab-axis" }),
      );
      const chanceLabel = this.svg("text", {
        x: String(right - 4), y: String(chanceY - 4), "text-anchor": "end", class: "lab-axis-label",
      });
      chanceLabel.textContent = `chance ${this.percent(chance)}`;
      this.benchmarkChart.append(chanceLabel);
    }
    for (const [text, px, py, anchor] of [
      ["0", left, 184, "start"], [String(maxUpdate), right, 184, "end"],
      ["100%", left - 6, top + 4, "end"], ["0%", left - 6, bottom + 4, "end"],
    ] as const) {
      const label = this.svg("text", { x: String(px), y: String(py), "text-anchor": anchor, class: "lab-axis-label" });
      label.textContent = text;
      this.benchmarkChart.append(label);
    }
    measured.forEach((benchmark, index) => {
      const seriesClass: string = SERIES_CLASSES[index] ?? "series-a";
      const path = this.svg("polyline", {
        points: benchmark.checkpoints.map((checkpoint) => (
          `${x(checkpoint.update).toFixed(2)},${y(checkpoint.heldOutAccuracy).toFixed(2)}`
        )).join(" "),
        class: `lab-series ${seriesClass}`,
      });
      this.benchmarkChart.append(path);
      let previousPairs = benchmark.checkpoints[0]?.recallPairs;
      for (const checkpoint of benchmark.checkpoints) {
        if (checkpoint.recallPairs !== previousPairs) {
          this.benchmarkChart.append(this.svg("circle", {
            cx: x(checkpoint.update).toFixed(2), cy: y(checkpoint.heldOutAccuracy).toFixed(2),
            r: "3.2", class: `curriculum-transition ${seriesClass}`,
          }));
          previousPairs = checkpoint.recallPairs;
        }
      }
      const legend = document.createElement("span");
      legend.className = seriesClass;
      legend.textContent = benchmark.id;
      this.benchmarkLegend.append(legend);
    });
  }

  private drawBenchmarkTopologyChart(benchmarks: BenchmarkSnapshot[]): void {
    this.benchmarkTopologyChart.replaceChildren();
    const measured = benchmarks.filter((benchmark) => benchmark.checkpoints.some(
      (checkpoint) => checkpoint.livingCells !== undefined && checkpoint.edgeCount !== undefined,
    ));
    if (measured.length === 0) {
      const label = this.svg("text", { x: "360", y: "82", class: "chart-empty" });
      label.textContent = "No topology history in matched cohort";
      this.benchmarkTopologyChart.append(label);
      return;
    }

    const control = measured.find((benchmark) => benchmark.intervention === "control");
    const initial = control?.checkpoints[0] ?? measured
      .flatMap((benchmark) => benchmark.checkpoints)
      .reduce((best, checkpoint) => (
        (checkpoint.livingCells ?? 0) > (best.livingCells ?? 0) ? checkpoint : best
      ));
    const referenceCells = Math.max(1, initial.livingCells ?? 1);
    const referenceEdges = Math.max(1, initial.edgeCount ?? 1);
    const maxUpdate = Math.max(...measured.flatMap(
      (benchmark) => benchmark.checkpoints.map((checkpoint) => checkpoint.update),
    ));
    const ratios = measured.flatMap((benchmark) => benchmark.checkpoints.flatMap((checkpoint) => [
      (checkpoint.livingCells ?? 0) / referenceCells,
      (checkpoint.edgeCount ?? 0) / referenceEdges,
    ]));
    const ceiling = Math.max(1, Math.ceil(Math.max(...ratios) * 5) / 5);
    const left = 46; const right = 706; const top = 10; const bottom = 132;
    const x = (value: number) => left + value / Math.max(1, maxUpdate) * (right - left);
    const y = (value: number) => bottom - Math.max(0, Math.min(ceiling, value)) / ceiling * (bottom - top);
    this.benchmarkTopologyChart.append(
      this.svg("line", { x1: String(left), y1: String(bottom), x2: String(right), y2: String(bottom), class: "lab-axis" }),
      this.svg("line", { x1: String(left), y1: String(top), x2: String(left), y2: String(bottom), class: "lab-axis" }),
    );
    for (const [text, px, py, anchor] of [
      ["0", left, 152, "start"], [String(maxUpdate), right, 152, "end"],
      [`${Math.round(ceiling * 100)}%`, left - 6, top + 4, "end"], ["0%", left - 6, bottom + 4, "end"],
    ] as const) {
      const label = this.svg("text", { x: String(px), y: String(py), "text-anchor": anchor, class: "lab-axis-label" });
      label.textContent = text;
      this.benchmarkTopologyChart.append(label);
    }
    measured.forEach((benchmark) => {
      const index = benchmarks.indexOf(benchmark);
      const seriesClass: string = SERIES_CLASSES[index] ?? "series-a";
      const points = benchmark.checkpoints.filter(
        (checkpoint) => checkpoint.livingCells !== undefined && checkpoint.edgeCount !== undefined,
      );
      this.benchmarkTopologyChart.append(
        this.svg("polyline", {
          points: points.map((checkpoint) => `${x(checkpoint.update).toFixed(2)},${y((checkpoint.livingCells ?? 0) / referenceCells).toFixed(2)}`).join(" "),
          class: `lab-series ${seriesClass}`,
        }),
        this.svg("polyline", {
          points: points.map((checkpoint) => `${x(checkpoint.update).toFixed(2)},${y((checkpoint.edgeCount ?? 0) / referenceEdges).toFixed(2)}`).join(" "),
          class: `lab-series topology-edges ${seriesClass}`,
        }),
      );
    });
  }

  private renderGpus(gpus: GpuSnapshot[]): void {
    this.gpuHost.replaceChildren(...gpus.map((gpu) => {
      const lane = document.createElement("article");
      lane.className = "gpu-lane";
      const usage = gpu.memoryTotalMiB > 0 ? gpu.memoryUsedMiB / gpu.memoryTotalMiB : 0;
      lane.innerHTML = `
        <div><strong>${this.escape(gpu.name)}</strong><span>${gpu.utilizationPercent.toFixed(0)}% · ${gpu.powerWatts.toFixed(0)} W</span></div>
        <div class="gpu-memory"><i style="width:${Math.min(100, usage * 100).toFixed(1)}%"></i></div>
        <small>${(gpu.memoryUsedMiB / 1024).toFixed(1)} / ${(gpu.memoryTotalMiB / 1024).toFixed(1)} GiB · ${gpu.processes.length} compute process${gpu.processes.length === 1 ? "" : "es"}</small>`;
      return lane;
    }));
  }

  private renderRuns(runs: RunSnapshot[], gpus: GpuSnapshot[]): void {
    const gpuNames = new Map(gpus.map((gpu) => [gpu.uuid, gpu.name.replace("NVIDIA GeForce ", "")]));
    const rows = runs.map((run) => {
      const row = document.createElement("tr");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = this.selectedRuns.has(run.id);
      checkbox.setAttribute("aria-label", `Compare ${run.id}`);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked && this.selectedRuns.size >= SERIES_CLASSES.length) {
          checkbox.checked = false;
          this.status.value = `compare at most ${SERIES_CLASSES.length} runs`;
          return;
        }
        if (checkbox.checked) this.selectedRuns.add(run.id);
        else this.selectedRuns.delete(run.id);
        this.renderRunDiagnostics(this.snapshot?.runs ?? []);
        void this.refreshHistories();
      });
      const compare = document.createElement("td");
      compare.append(checkbox);
      row.append(compare);
      const values = [
        run.id,
        run.task === "tiny_stories" ? "Tokens" : "Characters",
        run.architecture.toUpperCase(),
        run.gpuUuid ? (gpuNames.get(run.gpuUuid) ?? run.gpuUuid.slice(0, 12)) : "—",
        run.latestTrain?.update?.toLocaleString() ?? "—",
        this.number(run.latestTrain?.rollingLoss ?? run.latestTrain?.loss, 3),
        this.number(run.latestHeldOut?.loss, 3),
        this.number(
          run.latestTrain?.targetTokensPerSecond
            ?? run.latestTrain?.targetCharactersPerSecond,
          0,
        ),
      ];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      const state = document.createElement("td");
      state.className = `run-state ${run.status}`;
      state.textContent = run.status;
      if (run.status === "running" && this.snapshot?.controlEnabled) {
        const stop = document.createElement("button");
        stop.type = "button";
        stop.textContent = "Stop";
        stop.addEventListener("click", () => void this.stop(run.id, stop));
        state.append(stop);
      }
      row.append(state);
      return row;
    });
    if (rows.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td colspan="10">No persisted runs found.</td>';
      rows.push(row);
    }
    this.runsHost.replaceChildren(...rows);
  }

  private renderRunDiagnostics(runs: RunSnapshot[]): void {
    const selected = runs.filter((run) => this.selectedRuns.has(run.id));
    const rows = selected.map((run) => {
      const diagnostic = run.latestDiagnostics;
      const heldOut = run.latestHeldOut;
      const row = document.createElement("tr");
      const graph = diagnostic
        ? `${diagnostic.livingCells ?? "—"} cells · ${diagnostic.edgeCount ?? "—"} physical · ${diagnostic.conductingEdgeCount ?? "—"} conducting`
        : "—";
      const routing = diagnostic
        ? `${diagnostic.minimumOutputHops ?? "—"}/${diagnostic.medianOutputHops ?? "—"} hops · ${diagnostic.tokenReachableOutputs ?? 0}/${diagnostic.contextReachableOutputs ?? 0}/${diagnostic.reachableOutputs ?? 0} token/context/graph${(diagnostic.tokenReachableOutputs ?? 0) === 0 && Number(run.configuration.messageSteps ?? 0) < Number(diagnostic.minimumOutputHops ?? 0) ? ` · insufficient ${run.configuration.messageSteps ?? "—"} < ${diagnostic.minimumOutputHops ?? "—"}` : ""}`
        : "—";
      const lifecycle = diagnostic
        ? `${String(run.configuration.lifecycleProfile ?? (run.configuration.lifecycle ? "baseline" : "off"))} · ${diagnostic.stunnedCells ?? 0} stunned · +${diagnostic.cumulativeBirths ?? 0}/−${diagnostic.cumulativeDeaths ?? 0} cells · ${diagnostic.cumulativeStuns ?? 0}/${diagnostic.cumulativeRecoveries ?? 0} stun/recover`
        : "—";
      const structure = diagnostic
        ? `${run.configuration.structure === false ? "fixed" : "adaptive"} · +${diagnostic.cumulativeGrownEdges ?? 0}/−${diagnostic.cumulativePrunedEdges ?? 0} edges · ${diagnostic.pruneEligibleEdges ?? 0} eligible · gen ${diagnostic.generation ?? 0}`
        : "—";
      const sample = heldOut?.generationSample === undefined
        ? "—"
        : `${heldOut.generationPrompt ?? ""} → ${this.compactSample(heldOut.generationSample)} · ${this.percent(heldOut.generationUniqueTokenRatio)} unique`;
      for (const value of [run.id, graph, routing, lifecycle, structure, sample]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      return row;
    });
    if (rows.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td colspan="6">Select a run to inspect measured organism diagnostics.</td>';
      rows.push(row);
    }
    this.diagnosticsHost.replaceChildren(...rows);
  }

  private async refreshHistories(): Promise<void> {
    const selected = [...this.selectedRuns];
    const responses: [string, MetricRecord[]][] = await Promise.all(selected.map(async (runId) => {
      const response = await fetch(`/api/lab/runs/${encodeURIComponent(runId)}/metrics?limit=600`, { cache: "no-store" });
      if (!response.ok) return [runId, [] as MetricRecord[]];
      const payload = await response.json() as { records: MetricRecord[] };
      return [runId, payload.records.filter((record) => record.type === "train")];
    }));
    this.histories = new Map(responses);
    this.drawChart();
  }

  private drawChart(): void {
    this.chart.replaceChildren();
    this.legend.replaceChildren();
    const selected = [...this.selectedRuns]
      .map((id) => ({ id, records: this.histories.get(id) ?? [] }))
      .filter((series) => series.records.length > 0);
    if (selected.length === 0) {
      const label = this.svg("text", { x: "360", y: "98", class: "chart-empty" });
      label.textContent = "Select a run with metrics";
      this.chart.append(label);
      return;
    }
    const points = selected.flatMap((series) => series.records.map((record) => ({
      update: record.update,
      loss: record.rollingLoss ?? record.loss ?? Number.NaN,
    }))).filter((point) => Number.isFinite(point.loss));
    const minUpdate = Math.min(...points.map((point) => point.update));
    const maxUpdate = Math.max(...points.map((point) => point.update));
    const minLoss = Math.min(...points.map((point) => point.loss));
    const maxLoss = Math.max(...points.map((point) => point.loss));
    const left = 46; const right = 706; const top = 12; const bottom = 164;
    const x = (value: number) => left + (value - minUpdate) / Math.max(1, maxUpdate - minUpdate) * (right - left);
    const y = (value: number) => bottom - (value - minLoss) / Math.max(0.001, maxLoss - minLoss) * (bottom - top);
    this.chart.append(
      this.svg("line", { x1: String(left), y1: String(bottom), x2: String(right), y2: String(bottom), class: "lab-axis" }),
      this.svg("line", { x1: String(left), y1: String(top), x2: String(left), y2: String(bottom), class: "lab-axis" }),
    );
    const labels = [
      [String(minUpdate), left, 184, "start"], [String(maxUpdate), right, 184, "end"],
      [maxLoss.toFixed(2), left - 6, top + 4, "end"], [minLoss.toFixed(2), left - 6, bottom + 4, "end"],
    ] as const;
    for (const [text, px, py, anchor] of labels) {
      const label = this.svg("text", { x: String(px), y: String(py), "text-anchor": anchor, class: "lab-axis-label" });
      label.textContent = text;
      this.chart.append(label);
    }
    selected.forEach((series, index) => {
      const seriesClass: string = SERIES_CLASSES[index] ?? "series-a";
      const values = series.records.map((record) => ({
        update: record.update, loss: record.rollingLoss ?? record.loss ?? Number.NaN,
      })).filter((point) => Number.isFinite(point.loss));
      const path = this.svg("polyline", {
        points: values.map((point) => `${x(point.update).toFixed(2)},${y(point.loss).toFixed(2)}`).join(" "),
        class: `lab-series ${seriesClass}`,
      });
      this.chart.append(path);
      const legend = document.createElement("span");
      legend.className = seriesClass;
      legend.textContent = series.id;
      this.legend.append(legend);
    });
  }

  private async launch(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    this.launchButton.disabled = true;
    this.launchStatus.value = "launching";
    const form = new FormData(this.form);
    const lifecycleProfile = String(form.get("lifecycleProfile"));
    const body = {
      runId: String(form.get("runId")), gpuUuid: String(form.get("gpuUuid")),
      task: String(form.get("task")),
      architecture: String(form.get("architecture")),
      fieldSize: 68,
      batchSize: Number(form.get("batchSize")), contextLength: 64,
      messageSteps: Number(form.get("messageSteps")),
      updates: Number(form.get("updates")), seed: Number(form.get("seed")),
      amp: String(form.get("amp")), lifecycle: lifecycleProfile !== "off",
      lifecycleProfile, structure: String(form.get("structure")) === "adaptive",
    };
    try {
      const response = await fetch("/api/lab/runs", {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
      });
      const payload = await response.json() as { runId?: string; detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? `launch failed (${response.status})`);
      this.launchStatus.value = `${payload.runId} launched`;
      if (payload.runId) this.selectedRuns.add(payload.runId);
      await this.refresh();
    } catch (error) {
      this.launchStatus.value = error instanceof Error ? error.message : "launch failed";
    } finally {
      this.launchButton.disabled = !(this.snapshot?.controlEnabled ?? false);
    }
  }

  private async stop(runId: string, button: HTMLButtonElement): Promise<void> {
    button.disabled = true;
    const response = await fetch(`/api/lab/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" });
    const payload = await response.json() as { status?: string; detail?: string };
    this.status.value = response.ok ? `${runId}: ${payload.status}` : (payload.detail ?? "stop failed");
    await this.refresh();
  }

  private populateSelect(select: HTMLSelectElement, choices: { value: string; label: string }[]): void {
    const current = select.value;
    const signature = choices.map((choice) => `${choice.value}:${choice.label}`).join("|");
    if (select.dataset.signature === signature) return;
    select.replaceChildren(...choices.map((choice) => {
      const option = document.createElement("option");
      option.value = choice.value;
      option.textContent = choice.label;
      return option;
    }));
    select.dataset.signature = signature;
    if (choices.some((choice) => choice.value === current)) select.value = current;
  }

  private svg(name: string, attributes: Record<string, string>): SVGElement {
    const element = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    return element;
  }

  private number(value: number | undefined, digits: number): string {
    return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
  }

  private percent(value: number | undefined): string {
    return value === undefined || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(1)}%`;
  }

  private escape(value: string): string {
    const span = document.createElement("span");
    span.textContent = value;
    return span.innerHTML;
  }

  private compactSample(value: string): string {
    const visible = value.replaceAll("\n", " ↵ ").replaceAll("\t", " ⇥ ").trim();
    return visible.length <= 72 ? visible : `${visible.slice(0, 69)}…`;
  }
}
