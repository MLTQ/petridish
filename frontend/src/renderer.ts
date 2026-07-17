import { Application, Container, Graphics } from "pixi.js";

import type { ExperimentSnapshot } from "./protocol";

export type FieldLayer =
  | "activation"
  | "energy"
  | "stimulation"
  | "load"
  | "credit"
  | "utility"
  | "genotype_norm"
  | "emission"
  | "stress"
  | "age"
  | "lineage"
  | "alive";

type DishPointerHandler = (x: number, y: number, painting: boolean) => void;

const COLORS = {
  background: 0x071012,
  positive: 0x54e6c1,
  negative: 0xef6fa9,
  traffic: 0xf3df79,
  newborn: 0xb0f4ff,
  prune: 0xff5b63,
  selected: 0xffffff,
  sensor: 0x70a5ff,
  motor: 0xffb15c,
  neutral: 0x213437,
};

export class DishRenderer {
  private readonly app = new Application();
  private readonly edgeLayer = new Graphics();
  private readonly eventLayer = new Graphics();
  private readonly cellLayer = new Graphics();
  private readonly markerLayer = new Graphics();
  private snapshot: ExperimentSnapshot | null = null;
  private layer: FieldLayer = "activation";
  private edgeThreshold = 0.08;
  private selectedCell: number | null = null;
  private pixiReady = false;
  private fallbackCanvas: HTMLCanvasElement | null = null;
  private fallbackContext: CanvasRenderingContext2D | null = null;
  private siteRows = new Map<number, number>();

  public constructor(
    private readonly host: HTMLElement,
    private readonly onPointer: DishPointerHandler,
  ) {}

  public async initialize(): Promise<void> {
    const initialized = this.app
      .init({
        antialias: true,
        background: COLORS.background,
        preference: "webgl",
        resizeTo: this.host,
        resolution: Math.min(window.devicePixelRatio, 2),
        autoDensity: true,
      })
      .then(() => true)
      .catch(() => false);
    this.pixiReady = await Promise.race([
      initialized,
      new Promise<false>((resolve) => window.setTimeout(() => resolve(false), 1_500)),
    ]);

    let canvas: HTMLCanvasElement;
    if (this.pixiReady) {
      canvas = this.app.canvas;
      this.host.appendChild(canvas);
      const stage = new Container();
      stage.addChild(this.edgeLayer, this.cellLayer, this.eventLayer, this.markerLayer);
      this.app.stage.addChild(stage);
    } else {
      canvas = document.createElement("canvas");
      this.fallbackCanvas = canvas;
      this.fallbackContext = canvas.getContext("2d", { alpha: false });
      if (!this.fallbackContext) throw new Error("Neither WebGL nor Canvas2D is available.");
      this.host.appendChild(canvas);
      this.resizeFallback();
    }
    canvas.setAttribute("aria-label", "Live spatial neural population");
    canvas.setAttribute("role", "img");

    const handlePointer = (event: PointerEvent, painting: boolean): void => {
      const position = this.fieldPosition(event);
      if (position) this.onPointer(position.x, position.y, painting);
    };
    canvas.addEventListener("pointerdown", (event) => handlePointer(event, false));
    canvas.addEventListener("pointermove", (event) => {
      if ((event.buttons & 1) === 1) handlePointer(event, true);
    });
    new ResizeObserver(() => {
      if (!this.pixiReady) this.resizeFallback();
      this.draw();
    }).observe(this.host);
  }

  public render(snapshot: ExperimentSnapshot): void {
    this.snapshot = snapshot;
    this.siteRows.clear();
    const indices = snapshot.field.indices;
    for (let row = 0; row < snapshot.field.cells.length; row += 1) {
      this.siteRows.set(indices?.[row] ?? row, row);
    }
    this.draw();
  }

  public setLayer(layer: FieldLayer): void {
    this.layer = layer;
    this.draw();
  }

  public setEdgeThreshold(value: number): void {
    this.edgeThreshold = value;
    this.draw();
  }

  public selectCell(index: number): void {
    this.selectedCell = index;
    this.draw();
  }

  private draw(): void {
    const snapshot = this.snapshot;
    if (!snapshot || this.renderWidth <= 0 || this.renderHeight <= 0) return;
    const geometry = this.geometry(snapshot);
    if (!this.pixiReady) {
      this.drawFallback(snapshot, geometry);
      return;
    }
    this.edgeLayer.clear();
    this.eventLayer.clear();
    this.cellLayer.clear();
    this.markerLayer.clear();
    this.drawEdges(snapshot, geometry);
    this.drawCells(snapshot, geometry);
    this.drawEvents(snapshot, geometry);
  }

  private geometry(snapshot: ExperimentSnapshot): DishGeometry {
    const size = Math.min(this.renderWidth, this.renderHeight);
    return {
      size,
      x: (this.renderWidth - size) / 2,
      y: (this.renderHeight - size) / 2,
      cell: size / Math.max(snapshot.field.width, snapshot.field.height),
    };
  }

  private get renderWidth(): number {
    return this.pixiReady ? this.app.screen.width : this.host.clientWidth;
  }

  private get renderHeight(): number {
    return this.pixiReady ? this.app.screen.height : this.host.clientHeight;
  }

  private drawEdges(snapshot: ExperimentSnapshot, geometry: DishGeometry): void {
    const edges = snapshot.edges;
    const measured = snapshot.task.kind === "mnist" && snapshot.task.phase === "feedback"
      ? edges.credit.map((value) => Math.abs(value))
      : edges.flow.map((value) => Math.abs(value));
    const maximum = Math.max(1e-9, ...measured);
    for (let index = 0; index < edges.source.length; index += 1) {
      const weight = edges.weight[index] ?? 0;
      if (Math.abs(weight) < this.edgeThreshold) continue;
      const source = edges.source[index];
      const destination = edges.destination[index];
      if (source === undefined || destination === undefined) continue;
      const start = this.cellCenter(source, snapshot, geometry);
      const end = this.cellCenter(destination, snapshot, geometry);
      const activity = (measured[index] ?? 0) / maximum;
      const alpha = 0.04 + 0.7 * Math.sqrt(activity);
      const width = Math.max(0.35, geometry.cell * (0.06 + 0.2 * Math.sqrt(activity)));
      this.edgeLayer.moveTo(start.x, start.y).lineTo(end.x, end.y).stroke({
        color: weight >= 0 ? COLORS.positive : COLORS.negative,
        alpha,
        width,
      });
    }
  }

  private drawCells(snapshot: ExperimentSnapshot, geometry: DishGeometry): void {
    const aliveIndex = this.channelIndex(snapshot, "alive");
    const sensorIndex = this.channelIndex(snapshot, "sensor_id");
    const motorIndex = this.channelIndex(snapshot, "motor_id");
    const scale = this.layerScale(snapshot);
    const gap = Math.max(0.15, geometry.cell * 0.12);
    for (let row = 0; row < snapshot.field.cells.length; row += 1) {
      const cell = snapshot.field.cells[row];
      if (!cell) continue;
      const site = snapshot.field.indices?.[row] ?? row;
      const x = geometry.x + (site % snapshot.field.width) * geometry.cell + gap;
      const y = geometry.y + Math.floor(site / snapshot.field.width) * geometry.cell + gap;
      const size = Math.max(0.7, geometry.cell - 2 * gap);
      const alive = aliveIndex >= 0 ? (cell[aliveIndex] ?? 0) : 1;
      this.cellLayer.rect(x, y, size, size).fill({
        color: this.cellColor(cell, snapshot, scale),
        alpha: 0.12 + 0.88 * alive,
      });
      const sensor = sensorIndex >= 0 ? (cell[sensorIndex] ?? 0) : 0;
      const motor = motorIndex >= 0 ? (cell[motorIndex] ?? 0) : 0;
      if (sensor !== 0 || motor !== 0 || this.selectedCell === site) {
        this.markerLayer.rect(x, y, size, size).stroke({
          color: this.selectedCell === site ? COLORS.selected : sensor !== 0 ? COLORS.sensor : COLORS.motor,
          alpha: 1,
          width: this.selectedCell === site ? 1.6 : Math.max(0.7, geometry.cell * 0.16),
        });
      }
    }
  }

  private drawEvents(snapshot: ExperimentSnapshot, geometry: DishGeometry): void {
    for (const event of snapshot.events) {
      if (event.type === "grown" || event.type === "pruned") {
        const start = this.cellCenter(event.source, snapshot, geometry);
        const end = this.cellCenter(event.destination, snapshot, geometry);
        this.eventLayer.moveTo(start.x, start.y).lineTo(end.x, end.y).stroke({
          color: event.type === "grown" ? COLORS.newborn : COLORS.prune,
          alpha: 0.92,
          width: Math.max(1, geometry.cell * 0.22),
        });
      } else {
        const center = this.cellCenter(event.destination, snapshot, geometry);
        this.eventLayer.circle(center.x, center.y, Math.max(1.2, geometry.cell * 0.45)).stroke({
          color: event.type === "born" ? COLORS.newborn : COLORS.prune,
          alpha: 0.95,
          width: 1.2,
        });
      }
    }
  }

  private layerScale(snapshot: ExperimentSnapshot): number {
    const index = this.channelIndex(snapshot, this.layer);
    if (index < 0 || this.layer === "alive" || this.layer === "energy") return 1;
    let maximum = 0;
    for (const cell of snapshot.field.cells) maximum = Math.max(maximum, Math.abs(cell[index] ?? 0));
    return Math.max(1e-9, maximum);
  }

  private cellColor(cell: number[], snapshot: ExperimentSnapshot, scale: number): number {
    const value = (name: string): number => {
      const index = this.channelIndex(snapshot, name);
      return index >= 0 ? (cell[index] ?? 0) : 0;
    };
    const raw = value(this.layer);
    if (this.layer === "energy") return blend(COLORS.neutral, 0x7ee787, clamp01(raw));
    if (this.layer === "alive") return blend(COLORS.neutral, 0xd8f7ee, clamp01(raw));
    if (
      this.layer === "stimulation"
      || this.layer === "load"
      || this.layer === "emission"
      || this.layer === "stress"
    ) {
      return blend(COLORS.neutral, COLORS.traffic, clamp01(raw / scale));
    }
    const magnitude = clamp01(Math.abs(raw) / scale);
    return blend(COLORS.neutral, raw >= 0 ? COLORS.positive : COLORS.negative, 0.12 + 0.88 * magnitude);
  }

  private channelIndex(snapshot: ExperimentSnapshot, name: string): number {
    return snapshot.field.channels.indexOf(name);
  }

  private cellCenter(index: number, snapshot: ExperimentSnapshot, geometry: DishGeometry): Point {
    return {
      x: geometry.x + (index % snapshot.field.width + 0.5) * geometry.cell,
      y: geometry.y + (Math.floor(index / snapshot.field.width) + 0.5) * geometry.cell,
    };
  }

  private fieldPosition(event: PointerEvent): Point | null {
    const snapshot = this.snapshot;
    if (!snapshot) return null;
    const canvas = this.pixiReady ? this.app.canvas : this.fallbackCanvas;
    if (!canvas) return null;
    const bounds = canvas.getBoundingClientRect();
    const canvasX = (event.clientX - bounds.left) * (this.renderWidth / bounds.width);
    const canvasY = (event.clientY - bounds.top) * (this.renderHeight / bounds.height);
    const geometry = this.geometry(snapshot);
    const x = (canvasX - geometry.x) / geometry.cell;
    const y = (canvasY - geometry.y) / geometry.cell;
    if (x < 0 || y < 0 || x >= snapshot.field.width || y >= snapshot.field.height) return null;
    event.preventDefault();
    return { x, y };
  }

  private resizeFallback(): void {
    const canvas = this.fallbackCanvas;
    const context = this.fallbackContext;
    if (!canvas || !context) return;
    const density = Math.min(window.devicePixelRatio, 2);
    canvas.width = Math.max(1, Math.round(this.host.clientWidth * density));
    canvas.height = Math.max(1, Math.round(this.host.clientHeight * density));
    context.setTransform(density, 0, 0, density, 0, 0);
  }

  private drawFallback(snapshot: ExperimentSnapshot, geometry: DishGeometry): void {
    const context = this.fallbackContext;
    if (!context) return;
    context.fillStyle = cssColor(COLORS.background);
    context.fillRect(0, 0, this.renderWidth, this.renderHeight);
    const edges = snapshot.edges;
    const measured = snapshot.task.kind === "mnist" && snapshot.task.phase === "feedback"
      ? edges.credit.map((value) => Math.abs(value))
      : edges.flow.map((value) => Math.abs(value));
    const maximum = Math.max(1e-9, ...measured);
    for (let index = 0; index < edges.source.length; index += 1) {
      const weight = edges.weight[index] ?? 0;
      if (Math.abs(weight) < this.edgeThreshold) continue;
      const source = edges.source[index];
      const destination = edges.destination[index];
      if (source === undefined || destination === undefined) continue;
      const start = this.cellCenter(source, snapshot, geometry);
      const end = this.cellCenter(destination, snapshot, geometry);
      const activity = (measured[index] ?? 0) / maximum;
      context.globalAlpha = 0.04 + 0.7 * Math.sqrt(activity);
      context.strokeStyle = cssColor(weight >= 0 ? COLORS.positive : COLORS.negative);
      context.lineWidth = Math.max(0.35, geometry.cell * (0.06 + 0.2 * Math.sqrt(activity)));
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
    }

    const aliveIndex = this.channelIndex(snapshot, "alive");
    const sensorIndex = this.channelIndex(snapshot, "sensor_id");
    const motorIndex = this.channelIndex(snapshot, "motor_id");
    const scale = this.layerScale(snapshot);
    const gap = Math.max(0.15, geometry.cell * 0.12);
    for (let row = 0; row < snapshot.field.cells.length; row += 1) {
      const cell = snapshot.field.cells[row];
      if (!cell) continue;
      const site = snapshot.field.indices?.[row] ?? row;
      const x = geometry.x + (site % snapshot.field.width) * geometry.cell + gap;
      const y = geometry.y + Math.floor(site / snapshot.field.width) * geometry.cell + gap;
      const size = Math.max(0.7, geometry.cell - 2 * gap);
      context.globalAlpha = 0.12 + 0.88 * (aliveIndex >= 0 ? (cell[aliveIndex] ?? 0) : 1);
      context.fillStyle = cssColor(this.cellColor(cell, snapshot, scale));
      context.fillRect(x, y, size, size);
      const sensor = sensorIndex >= 0 ? (cell[sensorIndex] ?? 0) : 0;
      const motor = motorIndex >= 0 ? (cell[motorIndex] ?? 0) : 0;
      if (sensor !== 0 || motor !== 0 || this.selectedCell === site) {
        context.globalAlpha = 1;
        context.strokeStyle = cssColor(
          this.selectedCell === site ? COLORS.selected : sensor !== 0 ? COLORS.sensor : COLORS.motor,
        );
        context.lineWidth = this.selectedCell === site ? 1.6 : Math.max(0.7, geometry.cell * 0.16);
        context.strokeRect(x, y, size, size);
      }
    }
    for (const event of snapshot.events) {
      const center = this.cellCenter(event.destination, snapshot, geometry);
      context.globalAlpha = 0.95;
      context.strokeStyle = cssColor(
        event.type === "grown" || event.type === "born" ? COLORS.newborn : COLORS.prune,
      );
      context.lineWidth = 1.2;
      context.beginPath();
      if (event.type === "grown" || event.type === "pruned") {
        const start = this.cellCenter(event.source, snapshot, geometry);
        context.moveTo(start.x, start.y);
        context.lineTo(center.x, center.y);
      } else {
        context.arc(center.x, center.y, Math.max(1.2, geometry.cell * 0.45), 0, 2 * Math.PI);
      }
      context.stroke();
    }
    context.globalAlpha = 1;
  }
}

interface Point {
  x: number;
  y: number;
}

interface DishGeometry extends Point {
  size: number;
  cell: number;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function blend(from: number, to: number, amount: number): number {
  const t = clamp01(amount);
  const red = Math.round(((from >> 16) & 255) * (1 - t) + ((to >> 16) & 255) * t);
  const green = Math.round(((from >> 8) & 255) * (1 - t) + ((to >> 8) & 255) * t);
  const blue = Math.round((from & 255) * (1 - t) + (to & 255) * t);
  return (red << 16) | (green << 8) | blue;
}

function cssColor(color: number): string {
  return `#${color.toString(16).padStart(6, "0")}`;
}
