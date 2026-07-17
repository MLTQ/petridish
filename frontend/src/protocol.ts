export interface EdgeSnapshot {
  source: number[];
  destination: number[];
  weight: number[];
  age: number[];
  utility: number[];
  flow: number[];
  credit: number[];
}

export interface StructuralEvent {
  type: "grown" | "pruned" | "born" | "died";
  source: number;
  destination: number;
  tick: number;
  cause?: "starvation" | "overload" | "maintenance";
}

export interface MnistTaskSnapshot {
  kind: "mnist";
  phase: "input" | "forward" | "feedback" | "structural";
  target: number;
  prediction: number;
  accuracy: number;
  testAccuracy: number | null;
  confidence: number;
  loss: number;
  reward: number;
  epoch: number;
  seenExamples: number;
  trainingStep: number;
  trialStep: number;
  trialSteps: number;
  generation: number;
  structuralWarmupRemaining: number;
  lifecycleWarmupRemaining: number;
  lifecycleActive: boolean;
  lifecycleReason: string;
  learningPhase: "readout" | "rule" | "synapse" | "structure";
  structureUnlockReason: string;
  curriculumStage: number;
  curriculumStageCount: number;
  curriculumExamples: number;
  curriculumTargetAccuracy: number | null;
  curriculumStageAccuracy: number;
  curriculumStageUpdates: number;
  births: number;
  deaths: number;
  deathCauses: Record<"starvation" | "overload" | "maintenance", number>;
  cumulativeBirths: number;
  cumulativeDeaths: number;
  cumulativeDeathCauses: Record<"starvation" | "overload" | "maintenance", number>;
  image: number[];
}

export interface SequenceTaskSnapshot {
  kind: "sequence";
  taskKey: "associative_recall" | "tiny_language" | "tiny_shakespeare";
  title: string;
  description: string;
  phase: "token" | "feedback" | "structural";
  vocabulary: string[];
  tokens: string[];
  tokenIds: number[];
  targets: (string | null)[];
  targetIds: number[];
  predictions: string[];
  predictionIds: number[];
  position: number;
  accuracy: number;
  testAccuracy: number | null;
  confidence: number;
  loss: number;
  perplexity: number;
  recallPairs: number;
  recallMaxPairs: number;
  stageAccuracy: number;
  datasetName: string | null;
  datasetCharacters: number;
  contextLength: number;
  interactive: boolean;
  interactivePrompt: string;
  generatedText: string;
  nextTokenPrediction: string;
  sourceUrl: string | null;
  reward: number;
  seenExamples: number;
  trainingStep: number;
  trialStep: number;
  trialSteps: number;
  generation: number;
  structuralWarmupRemaining: number;
  lifecycleWarmupRemaining: number;
  lifecycleActive: boolean;
  lifecycleReason: string;
  learningPhase: string;
  structureUnlockReason: string;
  births: number;
  deaths: number;
  deathCauses: Record<"starvation" | "overload" | "maintenance", number>;
  cumulativeBirths: number;
  cumulativeDeaths: number;
  cumulativeDeathCauses: Record<"starvation" | "overload" | "maintenance", number>;
}

export type TaskSnapshot = MnistTaskSnapshot | SequenceTaskSnapshot;

export interface HyperparameterSnapshot {
  key: string;
  label: string;
  group: string;
  value: number;
  min: number;
  max: number;
  step: number;
  integer: boolean;
  choices?: number[];
}

export interface ExperimentSnapshot {
  type: "snapshot";
  experiment: "mnist" | "associative_recall" | "tiny_language" | "tiny_shakespeare";
  tick: number;
  runtime: {
    mode: "visualization" | "headless";
    running: boolean;
    stepsPerFrame: number;
    lastComputeSeconds: number;
    trainingUpdatesPerSecond: number;
    trainingExamplesPerSecond: number;
    reportIntervalSeconds: number;
    computePhase: "idle" | "forward" | "backward" | "optimizer" | "credit" | "lifecycle" | "evaluation" | "headless";
    computeProgress: number;
    computeTotal: number;
    controlRevision: number;
    savedOrganisms: { id: string; label: string }[];
    loadedOrganism: string | null;
  };
  field: {
    width: number;
    height: number;
    channels: string[];
    indices: number[] | null;
    cells: number[][];
  };
  edges: EdgeSnapshot;
  events: StructuralEvent[];
  task: TaskSnapshot;
  metrics: {
    reward: number;
    rollingReward: number;
    loss: number | null;
    livingCells: number;
    meanEnergy: number;
    meanAge: number;
    stressedCells: number;
    turnoverEvents: number;
    edgeCount: number;
    visibleEdgeCount: number;
    meanWeight: number;
    synapseUpdateRatio: number | null;
    structureLocked: boolean;
    meanAttentionEntropy?: number;
    minimumOutputHops?: number | null;
    medianOutputHops?: number | null;
    reachableOutputs?: number;
    temporallyReachableOutputs?: number;
    activeParameters?: number;
    parametersPerLivingCell?: number;
    device: string;
  };
  configuration?: {
    parameters: HyperparameterSnapshot[];
  };
}

export type ExperimentCommand =
  | { type: "play" | "pause" }
  | { type: "step"; count?: number }
  | { type: "reset"; seed?: number }
  | { type: "speed"; steps: number }
  | { type: "training"; enabled: boolean }
  | { type: "lesion"; x: number; y: number; radius: number }
  | { type: "evaluate"; batches?: number }
  | { type: "lifecycle" }
  | { type: "experiment"; name: ExperimentSnapshot["experiment"] }
  | { type: "load"; organism: string }
  | { type: "prompt"; text: string }
  | { type: "generate" }
  | { type: "configure"; values: Record<string, number> };

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage = ExperimentSnapshot | ErrorMessage;
