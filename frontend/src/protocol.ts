export interface EdgeSnapshot {
  source: number[];
  destination: number[];
  weight: number[];
  age: number[];
  utility: number[];
}

export interface StructuralEvent {
  type: "grown" | "pruned";
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
  phase: "seed" | "sensing" | "developing" | "reading";
  target: number;
  prediction: number;
  accuracy: number;
  testAccuracy: number | null;
  confidence: number;
  loss: number;
  epoch: number;
  seenExamples: number;
  trainingStep: number;
  assemblyStep: number;
  assemblySteps: number;
  tokenRow: number;
  routingRound: number;
  image: number[];
}

export type TaskSnapshot = XorTaskSnapshot | MnistTaskSnapshot;

export interface ExperimentSnapshot {
  type: "snapshot";
  experiment: "xor" | "mnist";
  tick: number;
  field: {
    width: number;
    height: number;
    channels: string[];
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
    meanWeight: number;
    device: string;
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
  | { type: "rewire" };

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage = ExperimentSnapshot | ErrorMessage;
