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
  type: "train" | "held_out" | "training_audit" | "random_context_audit" | "full_corpus_context_audit" | "trajectory_audit" | "diagnostic" | "failure" | "phase" | "retry" | "resume" | "checkpoint";
  update: number;
  loss?: number;
  rollingLoss?: number;
  phaseRollingLoss?: number;
  accuracy?: number;
  rollingAccuracy?: number;
  phaseRollingAccuracy?: number;
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
  lifecycleActive?: boolean;
  lifecycleReason?: string;
  lifecycleWarmupRemaining?: number;
  structureUnlocked?: boolean;
  structureUnlockReason?: string;
  structuralWarmupRemaining?: number;
  structurePlateauRemaining?: number;
  structuralInterval?: number;
  graphReferenceLoss?: number;
  graphReferenceAccuracy?: number;
  graphSilencedLoss?: number;
  graphSilencedAccuracy?: number;
  graphSilencedLossDelta?: number;
  graphSilencedAccuracyDelta?: number;
  sourceRotatedLoss?: number;
  sourceRotatedAccuracy?: number;
  sourceRotatedLossDelta?: number;
  sourceRotatedAccuracyDelta?: number;
  weightReassignedLossDelta?: number;
  weightReassignedAccuracyDelta?: number;
  broadcastSilencedLossDelta?: number;
  broadcastSilencedAccuracyDelta?: number;
  broadcastAblationApplicable?: boolean;
  generationPrompt?: string;
  generationSample?: string;
  generationUniqueTokenRatio?: number;
  generationSpecialTokenRatio?: number;
  generationUnknownTokenRatio?: number;
  generationTokenIds?: number[];
  positionIndices?: number[];
  positionAccuracy?: number[];
  unigramBaselineAccuracy?: number;
  bigramBaselineAccuracy?: number;
  unigramBaselineLoss?: number;
  bigramBaselineLoss?: number;
  validationUnknownTokenRate?: number;
  evaluationSeed?: number | null;
  evaluationBatches?: number;
  evaluatedTokens?: number;
  electricalStateTokens?: number;
  coldStateLoss?: number;
  coldStateAccuracy?: number;
  stateCarryAccuracyDelta?: number;
  stateCarryLossDelta?: number;
  initialStateTokens?: number;
  stateRetention?: number;
  stateHorizon?: Array<{
    windows: number;
    tokens: number;
    loss: number;
    accuracy: number;
  }>;
  stateLanes?: number;
  stateLane?: number;
  stateLaneStreamTokens?: number;
  activeStateLanes?: number;
  coldStateLanes?: number;
  experienceTrajectoryCount?: number;
  laneStreamDomains?: Array<{
    tokens: number;
    lanes: number;
    firstLane?: number;
    uniqueCursorPhases?: number;
    cursorPhaseCoverage?: number;
    minimumCursorPhaseLanes?: number;
    maximumCursorPhaseLanes?: number;
  }>;
  minimumLaneStreamTokens?: number;
  maximumLaneStreamTokens?: number;
  uniqueCursorPhases?: number;
  cursorPhaseCoverage?: number;
  minimumCursorPhaseLanes?: number;
  maximumCursorPhaseLanes?: number;
  minimumElectricalStateTokens?: number;
  maximumElectricalStateTokens?: number;
  failureType?: string;
  failureMessage?: string;
  organismId?: string;
  phaseIndex?: number;
  phaseName?: string;
  topologyProfile?: string;
  trainingStreamTokens?: number;
  fullTrainingStreamTokens?: number;
  trainingShardTokens?: number | null;
  classBiasGradientNorm?: number;
  outputReadoutGradientNorm?: number;
  tokenEncoderGradientNorm?: number;
  cellRuleGradientNorm?: number;
  synapseGradientNorm?: number;
  totalGradientNorm?: number;
  gradientClipScale?: number;
  gradientClip?: number;
  randomOffsetAuxiliaryWeight?: number;
  randomOffsetAuxiliaryScope?: "active_shard" | "full_corpus";
  randomOffsetAuxiliaryLoss?: number | null;
  randomOffsetAuxiliaryAccuracy?: number | null;
  evaluationSplit?: "validation" | "training" | "trajectory" | "random_context" | "full_corpus_context";
  trajectoryLane?: number | null;
  trajectoryStreamTokens?: number | null;
}

interface OrganismPhase {
  index: number;
  name: string;
  startUpdate: number;
  targetUpdate: number;
  structure: boolean;
  lifecycleProfile: string;
  trainingShardTokens?: number;
  stateLanes?: number;
  gradientClip?: number;
  randomOffsetAuxiliaryWeight?: number;
  randomOffsetAuxiliaryScope?: "active_shard" | "full_corpus";
  startGrownEdges?: number;
  startPrunedEdges?: number;
  startBirths?: number;
  startDeaths?: number;
}

interface RunSnapshot {
  id: string;
  task: string;
  architecture: string;
  gpuUuid: string | null;
  pid: number | null;
  status: "running" | "checkpointed" | "stopped" | "failed";
  configuration: Record<string, unknown>;
  organismId: string | null;
  phaseHistory: OrganismPhase[];
  parentCheckpoint?: { runId: string; update: number; sha256: string } | null;
  branchRootRunId?: string | null;
  branchDepth?: number;
  commit: string | null;
  latestTrain: MetricRecord | null;
  latestHeldOut: MetricRecord | null;
  latestTrainingAudit: MetricRecord | null;
  latestRandomContextAudit?: MetricRecord | null;
  latestFullCorpusContextAudit?: MetricRecord | null;
  latestTrajectoryAudit: MetricRecord | null;
  latestTrajectoryAudits?: MetricRecord[];
  latestDiagnostics: MetricRecord | null;
  latestFailure: MetricRecord | null;
  hasCheckpoint: boolean;
}

interface LaboratorySnapshot {
  controlEnabled: boolean;
  capabilities: {
    tasks: string[];
    architectures: string[];
    ampModes: string[];
    lifecycleProfiles: string[];
    topologyProfiles?: string[];
    tokenizerProfiles?: string[];
    checkpointEvaluation?: boolean;
    trainingShardAudit?: boolean;
    randomContextAudit?: boolean;
    fullCorpusContextAudit?: boolean;
    trainingShardCurriculum?: boolean;
    stateLaneExpansion?: boolean;
    stateLaneDomains?: boolean;
    maximumStateLanes?: number;
    phaseBalancedLaneExpansion?: boolean;
    trajectoryLaneAudit?: boolean;
    checkpointFork?: boolean;
    sameLineageRetry?: boolean;
    samePhaseResume?: boolean;
    phaseGradientClip?: boolean;
    persistentStateTraining?: boolean;
    randomOffsetAuxiliary?: boolean;
    randomOffsetAuxiliaryScope?: boolean;
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
  learningRateScale: number;
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
  private readonly tokenizerSelect = required<HTMLSelectElement>("#lab-tokenizer-profile");
  private readonly vocabularySelect = required<HTMLSelectElement>("#lab-vocabulary-size");
  private readonly messageStepsSelect = required<HTMLSelectElement>("#lab-message-steps");
  private readonly broadcastGainInput = required<HTMLInputElement>("#lab-broadcast-gain");
  private readonly stateLanesInput = required<HTMLInputElement>("#lab-state-lanes");
  private readonly lifecycleProfileSelect = required<HTMLSelectElement>("#lab-lifecycle-profile");
  private readonly launchButton = required<HTMLButtonElement>("#lab-launch");
  private readonly launchStatus = required<HTMLOutputElement>("#lab-launch-status");
  private readonly continueForm = required<HTMLFormElement>("#laboratory-continue-form");
  private readonly continueRunSelect = required<HTMLSelectElement>("#lab-continue-run");
  private readonly continueForkRunInput = required<HTMLInputElement>("#lab-continue-fork-run");
  private readonly continueGpuSelect = required<HTMLSelectElement>("#lab-continue-gpu");
  private readonly continueLifecycleSelect = required<HTMLSelectElement>("#lab-continue-lifecycle-profile");
  private readonly continueStructureSelect = required<HTMLSelectElement>("#lab-continue-structure");
  private readonly continueTrainingShardSelect = required<HTMLSelectElement>("#lab-continue-training-shard");
  private readonly continueStateLanesInput = required<HTMLInputElement>("#lab-continue-state-lanes");
  private readonly continueGradientClipInput = required<HTMLInputElement>("#lab-continue-gradient-clip");
  private readonly continueRandomOffsetAuxiliaryInput = required<HTMLInputElement>("#lab-continue-random-offset-auxiliary");
  private readonly continueRandomOffsetAuxiliaryScopeSelect = required<HTMLSelectElement>("#lab-continue-random-offset-scope");
  private readonly continueButton = required<HTMLButtonElement>("#lab-continue");
  private readonly continueStatus = required<HTMLOutputElement>("#lab-continue-status");
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
    this.continueForm.addEventListener("submit", (event) => void this.continueOrganism(event));
    this.continueRunSelect.addEventListener("change", () => this.syncContinuationPhase(true));
    this.taskSelect.addEventListener("change", () => {
      this.messageStepsSelect.value = this.taskSelect.value === "tiny_stories" ? "12" : "2";
      this.broadcastGainInput.value = this.taskSelect.value === "tiny_stories" ? "0.35" : "0.3";
      this.syncTokenizer();
    });
    this.tokenizerSelect.addEventListener("change", () => this.syncTokenizer());
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
      this.tokenizerSelect,
      (snapshot.capabilities.tokenizerProfiles ?? ["wordpiece", "byte"]).map((name) => ({
        value: name,
        label: name === "byte" ? "UTF-8 bytes (complete)" : "wordpieces (may contain <unk>)",
      })),
    );
    this.populateSelect(
      this.lifecycleProfileSelect,
      snapshot.capabilities.lifecycleProfiles.map((name) => ({ value: name, label: name })),
    );
    this.populateSelect(
      this.continueGpuSelect,
      snapshot.gpus.map((gpu) => ({ value: gpu.uuid, label: gpu.name })),
    );
    this.populateSelect(
      this.continueRunSelect,
      snapshot.runs.filter((run) => run.hasCheckpoint && run.status !== "running").map((run) => {
        const phase = (run.phaseHistory ?? []).at(-1);
        return {
          value: run.id,
          label: `${run.id} · p${phase?.index ?? 0} ${phase?.name ?? "legacy"}`,
        };
      }),
    );
    this.populateSelect(
      this.continueLifecycleSelect,
      snapshot.capabilities.lifecycleProfiles.map((name) => ({ value: name, label: name })),
    );
    this.syncContinuationPhase();
    for (const control of this.form.elements) {
      if (control instanceof HTMLInputElement || control instanceof HTMLSelectElement || control instanceof HTMLButtonElement) {
        control.disabled = !snapshot.controlEnabled;
      }
    }
    this.syncTokenizer();
    const maximumStateLanes = snapshot.capabilities.maximumStateLanes ?? 32;
    this.stateLanesInput.max = String(maximumStateLanes);
    if (Number(this.stateLanesInput.value) > maximumStateLanes) {
      this.stateLanesInput.value = String(maximumStateLanes);
    }
    const canContinue = snapshot.controlEnabled && this.continueRunSelect.options.length > 0;
    for (const control of this.continueForm.elements) {
      if (control instanceof HTMLInputElement || control instanceof HTMLSelectElement || control instanceof HTMLButtonElement) {
        control.disabled = !canContinue;
      }
    }
    this.continueTrainingShardSelect.disabled = !(
      canContinue && snapshot.capabilities.trainingShardCurriculum
    );
    if (!snapshot.capabilities.trainingShardCurriculum) {
      this.continueTrainingShardSelect.value = "preserve";
    }
    this.continueStateLanesInput.disabled = !(
      canContinue && snapshot.capabilities.stateLaneExpansion
    );
    if (!snapshot.capabilities.stateLaneExpansion) {
      this.continueStateLanesInput.value = "";
    }
    this.continueGradientClipInput.disabled = !(
      canContinue && snapshot.capabilities.phaseGradientClip
    );
    if (!snapshot.capabilities.phaseGradientClip) {
      this.continueGradientClipInput.value = "";
    }
    this.continueRandomOffsetAuxiliaryInput.disabled = !(
      canContinue && snapshot.capabilities.randomOffsetAuxiliary
    );
    this.continueRandomOffsetAuxiliaryScopeSelect.disabled = !(
      canContinue && snapshot.capabilities.randomOffsetAuxiliaryScope
    );
    if (!snapshot.capabilities.randomOffsetAuxiliary) {
      this.continueRandomOffsetAuxiliaryInput.value = "";
    }
    if (!snapshot.capabilities.randomOffsetAuxiliaryScope) {
      this.continueRandomOffsetAuxiliaryScopeSelect.value = "preserve";
    }
    this.continueForkRunInput.disabled = !(
      canContinue && snapshot.capabilities.checkpointFork
    );
    if (!snapshot.capabilities.checkpointFork) this.continueForkRunInput.value = "";
    this.launchStatus.value = snapshot.controlEnabled
      ? "new runs receive immutable manifests"
      : "launch control disabled on server";
    this.continueStatus.value = canContinue
      ? "same organism: continuation retains cells, connectome, weights, optimizer, stream cursor, and electrical memory; state ablations use disposable evaluation copies"
      : snapshot.controlEnabled
        ? "no stopped checkpoint is available"
        : "continuation control disabled on server";
  }

  private renderBenchmarks(benchmarks: BenchmarkSnapshot[]): void {
    const rows = benchmarks.map((benchmark) => {
      const final = benchmark.checkpoints.at(-1);
      const peak = benchmark.checkpoints.length
        ? Math.max(...benchmark.checkpoints.map((checkpoint) => checkpoint.heldOutAccuracy))
        : undefined;
      const tokenControl = [
        "token_routing", "token_memory", "token_context", "token_stream",
        "token_pipeline",
        "token_settling",
        "token_settled_pipeline",
        "token_grammar",
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
        this.number(
          run.latestTrain?.phaseRollingLoss
            ?? run.latestTrain?.rollingLoss
            ?? run.latestTrain?.loss,
          3,
        ),
        this.percent(
          run.latestTrain?.phaseRollingAccuracy
            ?? run.latestTrain?.rollingAccuracy
            ?? run.latestTrain?.accuracy,
        ),
        this.number(run.latestHeldOut?.loss, 3),
        this.percent(run.latestHeldOut?.accuracy),
        this.number(
          run.latestTrain?.targetTokensPerSecond
            ?? run.latestTrain?.targetCharactersPerSecond,
          0,
        ),
        this.lineagePhase(run),
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
      } else if (run.status === "failed" && run.latestFailure) {
        const failure = document.createElement("small");
        failure.textContent = `${run.latestFailure.failureType ?? "failure"}: ${run.latestFailure.failureMessage ?? "no detail"}`;
        state.append(failure);
        if (
          run.hasCheckpoint
          && this.snapshot?.controlEnabled
          && this.snapshot.capabilities.sameLineageRetry
        ) {
          const retry = document.createElement("button");
          retry.type = "button";
          retry.textContent = "Retry same organism";
          retry.addEventListener("click", () => void this.retryCheckpoint(run, retry));
          state.append(retry);
        }
      }
      const currentPhase = (run.phaseHistory ?? []).at(-1);
      if (
        run.status !== "running" && run.status !== "failed" && run.hasCheckpoint
        && this.snapshot?.controlEnabled
        && this.snapshot.capabilities.samePhaseResume
        && Number(currentPhase?.targetUpdate ?? 0) > Number(run.latestTrain?.update ?? 0)
      ) {
        const resume = document.createElement("button");
        resume.type = "button";
        resume.textContent = "Resume same phase";
        resume.addEventListener("click", () => void this.resumeCheckpoint(run, resume));
        state.append(resume);
      }
      if (
        run.status !== "running" && run.hasCheckpoint
        && this.snapshot?.controlEnabled
        && this.snapshot.capabilities.checkpointEvaluation
      ) {
        const evaluate = document.createElement("button");
        evaluate.type = "button";
        evaluate.textContent = "Evaluate validation";
        evaluate.addEventListener("click", () => void this.evaluateCheckpoint(
          run, evaluate, "validation",
        ));
        state.append(evaluate);
        const phase = (run.phaseHistory ?? []).at(-1);
        if (
          this.snapshot.capabilities.trainingShardAudit
          && (phase?.trainingShardTokens ?? 0) > 0
        ) {
          const shard = document.createElement("button");
          shard.type = "button";
          shard.textContent = "Audit warm shard trajectory";
          shard.addEventListener("click", () => void this.evaluateCheckpoint(
            run, shard, "training",
          ));
          state.append(shard);
          if (this.snapshot.capabilities.randomContextAudit) {
            const randomContexts = document.createElement("button");
            randomContexts.type = "button";
            randomContexts.textContent = "Probe cold contexts (read-only)";
            randomContexts.addEventListener("click", () => void this.evaluateCheckpoint(
              run, randomContexts, "random_context",
            ));
            state.append(randomContexts);
          }
          if (
            run.task === "tiny_stories"
            && this.snapshot.capabilities.fullCorpusContextAudit
          ) {
            const fullCorpusContexts = document.createElement("button");
            fullCorpusContexts.type = "button";
            fullCorpusContexts.textContent = "Probe full corpus (read-only)";
            fullCorpusContexts.addEventListener("click", () => void this.evaluateCheckpoint(
              run, fullCorpusContexts, "full_corpus_context",
            ));
            state.append(fullCorpusContexts);
          }
          const domains = run.latestDiagnostics?.laneStreamDomains ?? [];
          let legacyFirstLane = 0;
          const targets = this.snapshot.capabilities.trajectoryLaneAudit
            ? domains.map((domain) => {
              const target = {
                ...domain,
                firstLane: domain.firstLane ?? legacyFirstLane,
              };
              legacyFirstLane += domain.lanes;
              return target;
            })
            : [];
          if (targets.length > 0) {
            for (const domain of targets) {
              const trajectory = document.createElement("button");
              trajectory.type = "button";
              trajectory.textContent = `Audit lane ${domain.firstLane} · ${domain.tokens.toLocaleString()} tokens`;
              trajectory.addEventListener("click", () => void this.evaluateCheckpoint(
                run, trajectory, "trajectory", domain.firstLane,
              ));
              state.append(trajectory);
            }
          } else {
            const trajectory = document.createElement("button");
            trajectory.type = "button";
            trajectory.textContent = "Audit next trajectory";
            trajectory.addEventListener("click", () => void this.evaluateCheckpoint(
              run, trajectory, "trajectory",
            ));
            state.append(trajectory);
          }
        }
      }
      row.append(state);
      return row;
    });
    if (rows.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td colspan="13">No persisted runs found.</td>';
      rows.push(row);
    }
    this.runsHost.replaceChildren(...rows);
  }

  private renderRunDiagnostics(runs: RunSnapshot[]): void {
    const selected = runs.filter((run) => this.selectedRuns.has(run.id));
    const rows = selected.map((run) => {
      const diagnostic = run.latestDiagnostics;
      const heldOut = run.latestHeldOut;
      const shardAudit = run.latestTrainingAudit;
      const randomContextAudit = run.latestRandomContextAudit;
      const fullCorpusContextAudit = run.latestFullCorpusContextAudit;
      const trajectoryAudits = run.latestTrajectoryAudits?.length
        ? run.latestTrajectoryAudits
        : run.latestTrajectoryAudit ? [run.latestTrajectoryAudit] : [];
      const row = document.createElement("tr");
      const graph = diagnostic
        ? `${diagnostic.livingCells ?? "—"} cells · ${diagnostic.edgeCount ?? "—"} physical · ${diagnostic.conductingEdgeCount ?? "—"} conducting`
        : "—";
      const minimumStateAge = diagnostic?.minimumElectricalStateTokens;
      const maximumStateAge = diagnostic?.maximumElectricalStateTokens;
      const stateAge = minimumStateAge !== undefined && maximumStateAge !== undefined
        ? `state age ${minimumStateAge.toLocaleString()}–${maximumStateAge.toLocaleString()} tokens`
        : diagnostic?.electricalStateTokens === undefined
          ? "state age unreported"
          : `state age ${diagnostic.electricalStateTokens.toLocaleString()} tokens`;
      const phaseCoverage = diagnostic?.uniqueCursorPhases === undefined
        ? ""
        : ` · ${diagnostic.activeStateLanes ?? 0}/${diagnostic.stateLanes ?? run.configuration.stateLanes ?? 1} active lanes · ${diagnostic.uniqueCursorPhases}/${run.configuration.contextLength ?? 64} cursor phases${diagnostic.minimumCursorPhaseLanes === undefined ? "" : ` · ${diagnostic.minimumCursorPhaseLanes}–${diagnostic.maximumCursorPhaseLanes ?? diagnostic.minimumCursorPhaseLanes} lanes/phase`}`;
      const streamDomains = diagnostic?.laneStreamDomains?.length
        ? ` · domains ${diagnostic.laneStreamDomains.map((domain) => `${domain.lanes}×${domain.tokens.toLocaleString()}${domain.uniqueCursorPhases === undefined ? "" : ` @ ${domain.uniqueCursorPhases}/${run.configuration.contextLength ?? 64} phases`}`).join(" + ")}`
        : "";
      const laneAccuracy = this.laneAccuracySpread(run);
      const gradientSummary = run.latestTrain?.classBiasGradientNorm === undefined
        ? ""
        : ` · grad bias/readout/token/rule/edge ${this.scientific(run.latestTrain.classBiasGradientNorm)}/${this.scientific(run.latestTrain.outputReadoutGradientNorm)}/${this.scientific(run.latestTrain.tokenEncoderGradientNorm)}/${this.scientific(run.latestTrain.cellRuleGradientNorm)}/${this.scientific(run.latestTrain.synapseGradientNorm)} · total ${this.scientific(run.latestTrain.totalGradientNorm)} → ceiling ${this.number(run.latestTrain.gradientClip ?? Number(run.configuration.gradientClip ?? 1), 3)} × clip ${this.number(run.latestTrain.gradientClipScale, 3)}${this.gradientPressure(run)}`;
      const randomOffsetAuxiliary = Number(
        run.latestTrain?.randomOffsetAuxiliaryWeight
        ?? run.configuration.randomOffsetAuxiliaryWeight
        ?? 0,
      );
      const auxiliarySummary = randomOffsetAuxiliary > 0
        ? ` · random-offset aux ${String(run.latestTrain?.randomOffsetAuxiliaryScope ?? run.configuration.randomOffsetAuxiliaryScope ?? "active_shard").replace("_", " ")} ×${this.number(randomOffsetAuxiliary, 2)} loss ${this.number(run.latestTrain?.randomOffsetAuxiliaryLoss ?? undefined, 3)} acc ${this.percent(run.latestTrain?.randomOffsetAuxiliaryAccuracy ?? undefined)}`
        : this.snapshot?.capabilities.persistentStateTraining
          ? " · persistent-state gradients only"
          : " · random-offset aux off";
      const shardCausality = shardAudit?.graphReferenceAccuracy === undefined
        ? ""
        : ` · warm shard trajectory ref ${this.percent(shardAudit.graphReferenceAccuracy)} · silence Δacc ${this.signedPercent(shardAudit.graphSilencedAccuracyDelta)} / Δloss ${this.signedNumber(shardAudit.graphSilencedLossDelta)} · rotate Δacc ${this.signedPercent(shardAudit.sourceRotatedAccuracyDelta)} / Δloss ${this.signedNumber(shardAudit.sourceRotatedLossDelta)} · reassign Δacc ${this.signedPercent(shardAudit.weightReassignedAccuracyDelta)} / Δloss ${this.signedNumber(shardAudit.weightReassignedLossDelta)}`;
      const randomContextCausality = randomContextAudit?.graphReferenceAccuracy === undefined
        ? ""
        : ` · read-only cold-context probe ref ${this.percent(randomContextAudit.graphReferenceAccuracy)} / loss ${this.number(randomContextAudit.graphReferenceLoss, 3)} · silence Δacc ${this.signedPercent(randomContextAudit.graphSilencedAccuracyDelta)} · rotate Δacc ${this.signedPercent(randomContextAudit.sourceRotatedAccuracyDelta)} · reassign Δacc ${this.signedPercent(randomContextAudit.weightReassignedAccuracyDelta)}`;
      const fullCorpusContextCausality = fullCorpusContextAudit?.graphReferenceAccuracy === undefined
        ? ""
        : ` · full-corpus cold probe ref ${this.percent(fullCorpusContextAudit.graphReferenceAccuracy)} / loss ${this.number(fullCorpusContextAudit.graphReferenceLoss, 3)} · silence Δacc ${this.signedPercent(fullCorpusContextAudit.graphSilencedAccuracyDelta)} · rotate Δacc ${this.signedPercent(fullCorpusContextAudit.sourceRotatedAccuracyDelta)} · reassign Δacc ${this.signedPercent(fullCorpusContextAudit.weightReassignedAccuracyDelta)}`;
      const trajectoryCausality = trajectoryAudits
        .filter((audit) => audit.graphReferenceAccuracy !== undefined)
        .map((audit) => ` · trajectory lane ${audit.trajectoryLane ?? "—"}${audit.trajectoryStreamTokens == null ? "" : ` @ ${audit.trajectoryStreamTokens.toLocaleString()} tokens`} ref ${this.percent(audit.graphReferenceAccuracy)} · silence Δacc ${this.signedPercent(audit.graphSilencedAccuracyDelta)} / Δloss ${this.signedNumber(audit.graphSilencedLossDelta)} · rotate Δacc ${this.signedPercent(audit.sourceRotatedAccuracyDelta)} / Δloss ${this.signedNumber(audit.sourceRotatedLossDelta)} · reassign Δacc ${this.signedPercent(audit.weightReassignedAccuracyDelta)} / Δloss ${this.signedNumber(audit.weightReassignedLossDelta)}`)
        .join("");
      const routing = diagnostic
        ? `${String(run.configuration.tokenizerProfile ?? "legacy tokenizer")} · ${String(run.configuration.streamMode ?? "windowed")} · retention ${String(run.configuration.stateRetention ?? "1 legacy")} · ${String(run.configuration.stateLanes ?? 1)} state lane${Number(run.configuration.stateLanes ?? 1) === 1 ? "" : "s"} · ${stateAge}${phaseCoverage}${streamDomains}${laneAccuracy} · ${diagnostic.minimumOutputHops ?? "—"}/${diagnostic.medianOutputHops ?? "—"} hops · ${diagnostic.tokenReachableOutputs ?? 0}/${diagnostic.contextReachableOutputs ?? 0}/${diagnostic.reachableOutputs ?? 0} token/context/graph · broadcast ${String(run.configuration.broadcastGain ?? "legacy")}${auxiliarySummary}${(diagnostic.tokenReachableOutputs ?? 0) === 0 && Number(run.configuration.messageSteps ?? 0) < Number(diagnostic.minimumOutputHops ?? 0) ? ` · insufficient ${run.configuration.messageSteps ?? "—"} < ${diagnostic.minimumOutputHops ?? "—"}` : ""}${gradientSummary}${heldOut?.graphReferenceAccuracy === undefined ? "" : ` · validation causal ref ${this.percent(heldOut.graphReferenceAccuracy)} · silence Δacc ${this.signedPercent(heldOut.graphSilencedAccuracyDelta)} / Δloss ${this.signedNumber(heldOut.graphSilencedLossDelta)} · rotate topology Δacc ${this.signedPercent(heldOut.sourceRotatedAccuracyDelta)} / Δloss ${this.signedNumber(heldOut.sourceRotatedLossDelta)}${heldOut.weightReassignedLossDelta === undefined ? "" : ` · reassign weights Δacc ${this.signedPercent(heldOut.weightReassignedAccuracyDelta)} / Δloss ${this.signedNumber(heldOut.weightReassignedLossDelta)}`}${heldOut.broadcastAblationApplicable ? ` · silence broadcast Δacc ${this.signedPercent(heldOut.broadcastSilencedAccuracyDelta)} / Δloss ${this.signedNumber(heldOut.broadcastSilencedLossDelta)}` : ""}`}${shardCausality}${randomContextCausality}${fullCorpusContextCausality}${trajectoryCausality}`
        : "—";
      const lifecycle = diagnostic
        ? `${String(run.configuration.lifecycleProfile ?? (run.configuration.lifecycle ? "baseline" : "off"))} · ${diagnostic.lifecycleReason ?? (diagnostic.lifecycleActive ? "active" : "inactive")}${(diagnostic.lifecycleWarmupRemaining ?? 0) > 0 ? ` · ${diagnostic.lifecycleWarmupRemaining} warm-up updates` : ""} · ${diagnostic.stunnedCells ?? 0} stunned · +${diagnostic.cumulativeBirths ?? 0}/−${diagnostic.cumulativeDeaths ?? 0} cells · ${diagnostic.cumulativeStuns ?? 0}/${diagnostic.cumulativeRecoveries ?? 0} stun/recover`
        : "—";
      const topologyProfile = String(
        run.configuration.topologyProfile
        ?? (run.configuration.structure === false ? "fixed" : "adaptive")
      );
      const topologyStatus = topologyProfile === "fixed"
        ? "fixed conducting"
        : `${topologyProfile.replace("_", "-")} ${diagnostic?.structureUnlocked ? "active" : "locked"}`;
      const phase = (run.phaseHistory ?? []).at(-1);
      const phaseTurnover = diagnostic && phase?.startGrownEdges !== undefined
        ? ` · phase +${Math.max(0, (diagnostic.cumulativeGrownEdges ?? 0) - phase.startGrownEdges)}/−${Math.max(0, (diagnostic.cumulativePrunedEdges ?? 0) - (phase.startPrunedEdges ?? 0))}`
        : "";
      const structure = diagnostic
        ? `${topologyStatus} · ${diagnostic.structureUnlockReason ?? "reason unreported"}${(diagnostic.structuralWarmupRemaining ?? 0) > 0 ? ` · ${diagnostic.structuralWarmupRemaining} warm-up updates` : !diagnostic.structureUnlocked && (diagnostic.structurePlateauRemaining ?? 0) > 0 ? ` · ≤${diagnostic.structurePlateauRemaining} plateau updates` : ""} · lifetime +${diagnostic.cumulativeGrownEdges ?? 0}/−${diagnostic.cumulativePrunedEdges ?? 0}${phaseTurnover} edges · ${diagnostic.pruneEligibleEdges ?? 0} eligible · gen ${diagnostic.generation ?? 0}`
        : "—";
      const sample = heldOut?.generationSample === undefined
        ? "—"
        : `${heldOut.generationPrompt ?? ""} → ${this.compactSample(heldOut.generationSample)} · ${this.percent(heldOut.generationUniqueTokenRatio)} unique · ${this.percent(heldOut.generationSpecialTokenRatio)} special · ${this.percent(heldOut.generationUnknownTokenRatio)} unknown${heldOut.validationUnknownTokenRate === undefined ? "" : ` · validation ${this.percent(heldOut.validationUnknownTokenRate)} unknown`}${heldOut.evaluatedTokens === undefined ? "" : ` · audit ${heldOut.evaluatedTokens.toLocaleString()} tokens @ seed ${heldOut.evaluationSeed ?? "evolving"}`}`;
      const positionAccuracy = heldOut ? this.positionBands(heldOut) : "—";
      for (const value of [
        `${run.id} · ${this.lineagePhase(run)}`, graph, routing, lifecycle, structure, positionAccuracy, sample,
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      return row;
    });
    if (rows.length === 0) {
      const row = document.createElement("tr");
      row.innerHTML = '<td colspan="7">Select a run to inspect measured organism diagnostics.</td>';
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
      return [runId, payload.records.filter((record) => record.type === "train" || record.type === "phase")];
    }));
    this.histories = new Map(responses);
    this.drawChart();
    this.renderRunDiagnostics(this.snapshot?.runs ?? []);
  }

  private gradientPressure(run: RunSnapshot): string {
    const phaseIndex = run.latestTrain?.phaseIndex;
    const scales = (this.histories.get(run.id) ?? [])
      .filter((record) => (
        record.type === "train"
        && record.gradientClipScale !== undefined
        && (phaseIndex === undefined || record.phaseIndex === phaseIndex)
      ))
      .map((record) => record.gradientClipScale as number)
      .filter(Number.isFinite)
      .slice(-160)
      .sort((left, right) => left - right);
    if (scales.length < 2) return "";
    const percentile = (fraction: number): number => (
      scales[Math.min(scales.length - 1, Math.floor((scales.length - 1) * fraction))]
      ?? scales[0]
      ?? 1
    );
    const severe = scales.filter((scale) => scale <= 0.1).length / scales.length;
    return ` · clip pressure n${scales.length} median ×${this.number(percentile(0.5), 3)} / p10 ×${this.number(percentile(0.1), 3)} / severe ${this.percent(severe)}`;
  }

  private laneAccuracySpread(run: RunSnapshot): string {
    const laneCount = Number(run.configuration.stateLanes ?? 1);
    if (laneCount <= 1) return "";
    const phaseIndex = run.latestTrain?.phaseIndex;
    const records = (this.histories.get(run.id) ?? [])
      .filter((record) => (
        record.type === "train"
        && record.accuracy !== undefined
        && (phaseIndex === undefined || record.phaseIndex === phaseIndex)
      ))
      .slice(-(laneCount * 4));
    const lanes = Array.from({ length: laneCount }, () => [] as number[]);
    const domainByLane = new Array<number | undefined>(laneCount);
    for (const record of records) {
      const lane = record.stateLane ?? ((record.update - 1) % laneCount);
      if (lane >= 0 && lane < laneCount) {
        lanes[lane]?.push(record.accuracy as number);
        if (record.stateLaneStreamTokens !== undefined) {
          domainByLane[lane] = record.stateLaneStreamTokens;
        }
      }
    }
    const laneMean = (values: number[]): number => (
      values.reduce((sum, value) => sum + value, 0) / values.length
    );
    const previousPhase = run.phaseHistory.at(-2);
    const previousLaneCount = Math.min(
      laneCount, Number(previousPhase?.stateLanes ?? laneCount),
    );
    const appendedLanes = lanes.slice(previousLaneCount);
    const appendedAccuracies = appendedLanes
      .filter((values) => values.length > 0)
      .map(laneMean);
    const appended = appendedLanes.length > 0
      && appendedAccuracies.length === appendedLanes.length
      ? ` · appended lane acc ${this.percent(laneMean(appendedAccuracies))} mean / ${this.percent(Math.min(...appendedAccuracies))}–${this.percent(Math.max(...appendedAccuracies))} (${appendedLanes.length} lanes)`
      : "";
    if (lanes.some((values) => values.length === 0)) return appended;
    const accuracies = lanes.map(laneMean);
    const mean = accuracies.reduce((sum, value) => sum + value, 0) / accuracies.length;
    const domainAccuracies = new Map<number, number[]>();
    domainByLane.forEach((domain, lane) => {
      if (domain === undefined) return;
      const values = domainAccuracies.get(domain) ?? [];
      values.push(accuracies[lane] as number);
      domainAccuracies.set(domain, values);
    });
    const domains = domainAccuracies.size <= 1
      ? ""
      : ` · domain acc ${[...domainAccuracies.entries()].sort((a, b) => a[0] - b[0]).map(([domain, values]) => (
        `${domain.toLocaleString()} ${this.percent(values.reduce((sum, value) => sum + value, 0) / values.length)}`
      )).join(" / ")}`;
    return `${appended} · lane acc ${this.percent(mean)} mean / ${this.percent(Math.min(...accuracies))}–${this.percent(Math.max(...accuracies))}${domains}`;
  }

  private drawChart(): void {
    this.chart.replaceChildren();
    this.legend.replaceChildren();
    const selected = [...this.selectedRuns]
      .map((id) => ({ id, records: this.histories.get(id) ?? [] }))
      .filter((series) => series.records.some((record) => record.type === "train"));
    if (selected.length === 0) {
      const label = this.svg("text", { x: "360", y: "98", class: "chart-empty" });
      label.textContent = "Select a run with metrics";
      this.chart.append(label);
      return;
    }
    const points = selected.flatMap((series) => series.records.filter(
      (record) => record.type === "train",
    ).map((record) => ({
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
      const values = series.records.filter((record) => record.type === "train").map((record) => ({
        update: record.update, loss: record.rollingLoss ?? record.loss ?? Number.NaN,
      })).filter((point) => Number.isFinite(point.loss));
      const path = this.svg("polyline", {
        points: values.map((point) => `${x(point.update).toFixed(2)},${y(point.loss).toFixed(2)}`).join(" "),
        class: `lab-series ${seriesClass}`,
      });
      this.chart.append(path);
      for (const phase of series.records.filter((record) => record.type === "phase")) {
        const phaseX = x(phase.update);
        this.chart.append(this.svg("line", {
          x1: phaseX.toFixed(2), y1: String(top), x2: phaseX.toFixed(2), y2: String(bottom),
          class: `phase-boundary ${seriesClass}`,
        }));
        const phaseLabel = this.svg("text", {
          x: (phaseX + 3).toFixed(2), y: String(top + 10), class: "lab-axis-label",
        });
        phaseLabel.textContent = `p${phase.phaseIndex ?? "?"}`;
        this.chart.append(phaseLabel);
      }
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
    const topologyProfile = String(form.get("topologyProfile"));
    const body = {
      runId: String(form.get("runId")), gpuUuid: String(form.get("gpuUuid")),
      task: String(form.get("task")),
      architecture: String(form.get("architecture")),
      tokenizerProfile: this.tokenizerSelect.value,
      fieldSize: 68,
      batchSize: Number(form.get("batchSize")), contextLength: 64,
      vocabularySize: Number(this.vocabularySelect.value),
      streamMode: String(form.get("streamMode")),
      stateRetention: Number(form.get("stateRetention")),
      stateLanes: Number(form.get("stateLanes")),
      messageSteps: Number(form.get("messageSteps")),
      broadcastGain: Number(form.get("broadcastGain")),
      updates: Number(form.get("updates")), seed: Number(form.get("seed")),
      learningRateScale: Number(form.get("learningRateScale")),
      amp: String(form.get("amp")), lifecycle: lifecycleProfile !== "off",
      lifecycleProfile, topologyProfile, structure: topologyProfile !== "fixed",
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

  private syncTokenizer(): void {
    const enabled = this.snapshot?.controlEnabled ?? false;
    const tokenTask = this.taskSelect.value === "tiny_stories";
    const byteComplete = tokenTask && this.tokenizerSelect.value === "byte";
    if (byteComplete) this.vocabularySelect.value = "256";
    this.tokenizerSelect.disabled = !enabled || !tokenTask;
    this.vocabularySelect.disabled = !enabled || !tokenTask || byteComplete;
  }

  private async stop(runId: string, button: HTMLButtonElement): Promise<void> {
    button.disabled = true;
    const response = await fetch(`/api/lab/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" });
    const payload = await response.json() as { status?: string; detail?: string };
    this.status.value = response.ok ? `${runId}: ${payload.status}` : (payload.detail ?? "stop failed");
    await this.refresh();
  }

  private async evaluateCheckpoint(
    run: RunSnapshot, button: HTMLButtonElement,
    evaluationSplit: "validation" | "training" | "trajectory" | "random_context" | "full_corpus_context",
    trajectoryLane?: number,
  ): Promise<void> {
    const gpuUuid = run.gpuUuid ?? this.snapshot?.gpus[0]?.uuid;
    if (!gpuUuid) {
      this.status.value = `${run.id}: no GPU available for evaluation`;
      return;
    }
    button.disabled = true;
    const response = await fetch(
      `/api/lab/runs/${encodeURIComponent(run.id)}/evaluate`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          gpuUuid, stateHorizons: false, evaluationSplit, trajectoryLane,
        }),
      },
    );
    const payload = await response.json() as { status?: string; detail?: string };
    this.status.value = response.ok
      ? `${run.id}: ${payload.status ?? "evaluating"} ${evaluationSplit}${trajectoryLane === undefined ? "" : ` lane ${trajectoryLane}`}`
      : (payload.detail ?? "evaluation failed");
    await this.refresh();
  }

  private async retryCheckpoint(
    run: RunSnapshot, button: HTMLButtonElement,
  ): Promise<void> {
    const gpuUuid = run.gpuUuid ?? this.snapshot?.gpus[0]?.uuid;
    if (!gpuUuid) {
      this.status.value = `${run.id}: no GPU available for retry`;
      return;
    }
    button.disabled = true;
    this.status.value = `${run.id}: verifying same-lineage checkpoint`;
    const response = await fetch(
      `/api/lab/runs/${encodeURIComponent(run.id)}/retry`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ gpuUuid }),
      },
    );
    const payload = await response.json() as {
      status?: string; organismId?: string; checkpointSha256?: string; detail?: string;
    };
    this.status.value = response.ok
      ? `${run.id}: ${payload.status ?? "running"} same organism ${payload.organismId ?? ""} · checkpoint ${payload.checkpointSha256?.slice(0, 12) ?? "verified"}`
      : (payload.detail ?? "same-lineage retry failed");
    await this.refresh();
  }

  private async resumeCheckpoint(
    run: RunSnapshot, button: HTMLButtonElement,
  ): Promise<void> {
    const gpuUuid = run.gpuUuid ?? this.snapshot?.gpus[0]?.uuid;
    if (!gpuUuid) {
      this.status.value = `${run.id}: no GPU available for same-phase resume`;
      return;
    }
    button.disabled = true;
    const response = await fetch(
      `/api/lab/runs/${encodeURIComponent(run.id)}/resume`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ gpuUuid }),
      },
    );
    const payload = await response.json() as {
      status?: string; phaseIndex?: number; targetUpdate?: number; detail?: string;
    };
    this.status.value = response.ok
      ? `${run.id}: ${payload.status ?? "running"} same phase ${payload.phaseIndex ?? "—"} → ${payload.targetUpdate?.toLocaleString() ?? "target"}`
      : (payload.detail ?? "same-phase resume failed");
    await this.refresh();
  }

  private async continueOrganism(event: SubmitEvent): Promise<void> {
    event.preventDefault();
    this.continueButton.disabled = true;
    this.continueStatus.value = "continuing checkpoint lineage";
    const form = new FormData(this.continueForm);
    let runId = String(form.get("runId"));
    const forkRunId = String(form.get("forkRunId") ?? "").trim();
    const lifecycleProfile = String(form.get("lifecycleProfile"));
    const topologyProfile = String(form.get("topologyProfile"));
    const phaseName = String(form.get("phaseName") ?? "").trim();
    const shardSelection = String(form.get("trainingShardTokens") ?? "preserve");
    const laneSelection = String(form.get("stateLanes") ?? "").trim();
    const gradientClipSelection = String(form.get("gradientClip") ?? "").trim();
    const randomOffsetAuxiliarySelection = String(
      form.get("randomOffsetAuxiliaryWeight") ?? "",
    ).trim();
    const randomOffsetAuxiliaryScope = String(
      form.get("randomOffsetAuxiliaryScope") ?? "preserve",
    );
    const body = {
      gpuUuid: String(form.get("gpuUuid")),
      additionalUpdates: Number(form.get("additionalUpdates")),
      lifecycle: lifecycleProfile !== "off",
      lifecycleProfile,
      topologyProfile,
      structure: topologyProfile !== "fixed",
      phaseName: phaseName || null,
      trainingShardTokens: shardSelection === "preserve" ? null : Number(shardSelection),
      stateLanes: laneSelection === "" ? null : Number(laneSelection),
      gradientClip: gradientClipSelection === "" ? null : Number(gradientClipSelection),
      randomOffsetAuxiliaryWeight: randomOffsetAuxiliarySelection === ""
        ? null
        : Number(randomOffsetAuxiliarySelection),
      randomOffsetAuxiliaryScope: randomOffsetAuxiliaryScope === "preserve"
        ? null
        : randomOffsetAuxiliaryScope,
    };
    try {
      if (forkRunId) {
        const forkResponse = await fetch(
          `/api/lab/runs/${encodeURIComponent(runId)}/fork`,
          {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ forkRunId }),
          },
        );
        const forkPayload = await forkResponse.json() as {
          runId?: string; organismId?: string; detail?: string;
        };
        if (!forkResponse.ok) {
          throw new Error(forkPayload.detail ?? `checkpoint fork failed (${forkResponse.status})`);
        }
        runId = forkPayload.runId ?? forkRunId;
        this.continueStatus.value = `${runId}: exact checkpoint fork created; continuing same organism`;
      }
      const response = await fetch(`/api/lab/runs/${encodeURIComponent(runId)}/continue`, {
        method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
      });
      const payload = await response.json() as {
        runId?: string; organismId?: string; phaseIndex?: number; detail?: string;
      };
      if (!response.ok) throw new Error(payload.detail ?? `continuation failed (${response.status})`);
      this.continueStatus.value = `${payload.runId} · same organism · phase ${payload.phaseIndex}`;
      if (payload.runId) this.selectedRuns.add(payload.runId);
      this.continueForkRunInput.value = "";
      await this.refresh();
    } catch (error) {
      this.continueStatus.value = error instanceof Error ? error.message : "continuation failed";
    } finally {
      this.continueButton.disabled = !(this.snapshot?.controlEnabled ?? false);
    }
  }

  private syncContinuationPhase(force = false): void {
    const run = this.snapshot?.runs.find((candidate) => candidate.id === this.continueRunSelect.value);
    if (!run) return;
    if (force || this.continueForm.dataset.runId !== run.id) {
      this.continueForm.dataset.runId = run.id;
      this.continueStructureSelect.value = String(
        run.configuration.topologyProfile
        ?? (run.configuration.structure === false ? "fixed" : "adaptive")
      );
      this.continueLifecycleSelect.value = String(
        run.configuration.lifecycleProfile ?? (run.configuration.lifecycle ? "baseline" : "off"),
      );
      this.continueTrainingShardSelect.value = "preserve";
      this.continueStateLanesInput.value = "";
      this.continueGradientClipInput.value = "";
      this.continueRandomOffsetAuxiliaryInput.value = "";
      this.continueRandomOffsetAuxiliaryScopeSelect.value = "preserve";
    }
    const currentLanes = Number(run.configuration.stateLanes ?? 1);
    const maximumStateLanes = this.snapshot?.capabilities.maximumStateLanes ?? 32;
    this.continueStateLanesInput.min = String(currentLanes);
    this.continueStateLanesInput.max = String(maximumStateLanes);
    if (
      this.continueStateLanesInput.value !== ""
      && Number(this.continueStateLanesInput.value) < currentLanes
    ) {
      this.continueStateLanesInput.value = "";
    }
    this.continueGradientClipInput.placeholder = `blank preserves ${Number(run.configuration.gradientClip ?? 1)}`;
    const legacyAuxiliary = Number(run.configuration.randomOffsetAuxiliaryWeight ?? 0);
    this.continueRandomOffsetAuxiliaryInput.placeholder = legacyAuxiliary > 0
      ? `legacy ×${legacyAuxiliary} will be disabled on continuation`
      : "persistent-state training only";
    this.continueRandomOffsetAuxiliaryScopeSelect.title = "fresh-state contexts are read-only audits";
  }

  private lineagePhase(run: RunSnapshot): string {
    const phase = (run.phaseHistory ?? []).at(-1);
    const lineage = run.organismId ? run.organismId.replace(/^organism-/, "").slice(0, 8) : "legacy";
    const curriculum = phase?.trainingShardTokens
      ? ` · repeat ${phase.trainingShardTokens.toLocaleString()}`
      : phase?.trainingShardTokens === 0 ? " · full stream" : "";
    const auxiliary = Number(
      phase?.randomOffsetAuxiliaryWeight
      ?? run.configuration.randomOffsetAuxiliaryWeight
      ?? 0,
    );
    const auxiliaryScope = String(
      phase?.randomOffsetAuxiliaryScope
      ?? run.configuration.randomOffsetAuxiliaryScope
      ?? "active_shard"
    ).replace("_", " ");
    const auxiliaryLabel = auxiliary > 0
      ? ` · random aux ${auxiliaryScope} ×${auxiliary}`
      : "";
    const lanes = phase?.stateLanes ?? Number(run.configuration.stateLanes ?? 1);
    const branch = run.parentCheckpoint
      ? ` · exact fork d${run.branchDepth ?? "?"} ← ${run.parentCheckpoint.runId}@${run.parentCheckpoint.update.toLocaleString()} sha ${run.parentCheckpoint.sha256.slice(0, 12)}`
      : "";
    return `organism ${lineage} · p${phase?.index ?? 0} ${phase?.name ?? "training"}${curriculum}${auxiliaryLabel} · ${lanes} lane${lanes === 1 ? "" : "s"}${branch}`;
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

  private scientific(value: number | undefined): string {
    return value === undefined || !Number.isFinite(value) ? "—" : value.toExponential(1);
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

  private positionBands(record: MetricRecord): string {
    const indices = record.positionIndices ?? [];
    const accuracy = record.positionAccuracy ?? [];
    const bands = [
      { label: "p0–3", values: [] as number[] },
      { label: "p4–15", values: [] as number[] },
      { label: "p16+", values: [] as number[] },
    ];
    indices.forEach((position, offset) => {
      const value = accuracy[offset];
      if (value === undefined || !Number.isFinite(value)) return;
      const band = position < 4 ? bands[0] : position < 16 ? bands[1] : bands[2];
      band?.values.push(value);
    });
    const populated = bands.filter((band) => band.values.length > 0);
    const positionSummary = populated.map((band) => (
      `${band.label} ${this.percent(
        band.values.reduce((sum, value) => sum + value, 0) / band.values.length,
      )}`
    )).join(" · ");
    const baselineSummary = record.unigramBaselineAccuracy === undefined
      ? ""
      : `uni ${this.percent(record.unigramBaselineAccuracy)} / ppl ${this.perplexity(record.unigramBaselineLoss)} · bi ${this.percent(record.bigramBaselineAccuracy)} / ppl ${this.perplexity(record.bigramBaselineLoss)}${record.accuracy === undefined ? "" : ` · model−uni ${this.signedPercent(record.accuracy - record.unigramBaselineAccuracy)} · model−bi ${this.signedPercent(record.accuracy - (record.bigramBaselineAccuracy ?? 0))}`}`;
    const curriculumSummary = record.trainingStreamTokens === undefined
      ? ""
      : record.trainingShardTokens
        ? `curriculum repeated ${record.trainingStreamTokens.toLocaleString()} / ${record.fullTrainingStreamTokens?.toLocaleString() ?? "—"} training tokens`
        : `curriculum full ${record.trainingStreamTokens.toLocaleString()} training tokens`;
    const stateSummary = record.coldStateAccuracy === undefined
      ? ""
      : `saved-state evaluation copy ${this.percent(record.accuracy)} · zero-state ablation copy ${this.percent(record.coldStateAccuracy)} · state Δacc ${this.signedPercent(record.stateCarryAccuracyDelta)} / Δloss ${this.signedNumber(record.stateCarryLossDelta ?? (record.coldStateLoss === undefined || record.loss === undefined ? undefined : record.coldStateLoss - record.loss))} · checkpoint state age ${(record.initialStateTokens ?? 0).toLocaleString()}`;
    const horizonSummary = (record.stateHorizon ?? []).map(
      (point) => `h${point.windows} ${this.percent(point.accuracy)}`,
    ).join(" · ");
    if (!positionSummary && !baselineSummary && !curriculumSummary && !stateSummary && !horizonSummary) return "—";
    return [curriculumSummary, baselineSummary, stateSummary, horizonSummary, positionSummary].filter(Boolean).join(" | ");
  }

  private signedPercent(value: number | undefined): string {
    if (value === undefined || !Number.isFinite(value)) return "—";
    return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
  }

  private signedNumber(value: number | undefined): string {
    if (value === undefined || !Number.isFinite(value)) return "—";
    return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
  }

  private perplexity(loss: number | undefined): string {
    if (loss === undefined || !Number.isFinite(loss)) return "—";
    return Math.exp(Math.min(20, loss)).toFixed(1);
  }
}
