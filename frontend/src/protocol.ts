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
}

export interface XorTaskSnapshot {
  kind: "xor";
  phase: "cue" | "delay" | "response" | "rest";
  bitA: number;
  bitB: number;
  target: number;
  prediction: number | null;
  accuracy: number;
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
  image: number[];
}

export type TaskSnapshot = XorTaskSnapshot | MnistTaskSnapshot;

export interface HyperparameterSnapshot {
  key: string;
  label: string;
  group: string;
  value: number;
  min: number;
  max: number;
  step: number;
  integer: boolean;
}

export interface ExperimentSnapshot {
  type: "snapshot";
  experiment: "xor" | "mnist";
  tick: number;
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
  | { type: "experiment"; name: "xor" | "mnist" }
  | { type: "step"; count?: number }
  | { type: "reset"; seed?: number }
  | { type: "speed"; steps: number }
  | { type: "lesion"; x: number; y: number; radius: number }
  | { type: "stimulate"; region: "sensor_a" | "sensor_b" | "motor_zero" | "motor_one"; amount?: number; duration?: number }
  | { type: "reward"; amount: number }
  | { type: "evaluate"; batches?: number }
  | { type: "rewire" }
  | { type: "configure"; values: Record<string, number> };

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage = ExperimentSnapshot | ErrorMessage;
