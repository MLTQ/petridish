import { Application, Container, Graphics } from "pixi.js";

import type { ExperimentSnapshot } from "./protocol";

export type FieldLayer =
  | "phase"
  | "activation"
  | "energy"
  | "axon_growth"
  | "dendrite_growth"
  | "reward_trace"
  | "alive";

type DishPointerHandler = (x: number, y: number, painting: boolean) => void;

const COLORS = {
  background: 0x071012,
  grid: 0x132427,
  positive: 0x54e6c1,
  negative: 0xef6fa9,
  signal: 0xf3df79,
  newborn: 0xb0f4ff,
  prune: 0xff5b63,
  selected: 0xffffff,
  sensor: 0x70a5ff,
  motor: 0xffb15c,
};

export class DishRenderer {
  private readonly app = new Application();
  private readonly edgeLayer = new Graphics();
  private readonly broadcastLayer = new Graphics();
  private readonly eventLayer = new Graphics();
  private readonly cellLayer = new Graphics();
  private readonly markerLayer = new Graphics();
  private readonly pulseLayer = new Graphics();
  private snapshot: ExperimentSnapshot | null = null;
  private layer: FieldLayer = "phase";
  private edgeThreshold = 0.08;
  private selectedCell: number | null = null;
  private pixiReady = false;
  private fallbackCanvas: HTMLCanvasElement | null = null;
  private fallbackContext: CanvasRenderingContext2D | null = null;

  public constructor(
    private readonly host: HTMLElement,
    private readonly onPointer: DishPointerHandler,
  ) {}

  public async initialize(): Promise<void> {
    const pixiInitialization = this.app
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
      pixiInitialization,
      new Promise<false>((resolve) => window.setTimeout(() => resolve(false), 1_500)),
    ]);

    let canvas: HTMLCanvasElement;
    if (this.pixiReady) {
      canvas = this.app.canvas;
      this.host.appendChild(canvas);
      const stage = new Container();
      stage.addChild(
        this.cellLayer,
        this.broadcastLayer,
        this.edgeLayer,
        this.eventLayer,
        this.markerLayer,
        this.pulseLayer,
      );
      this.app.stage.addChild(stage);
    } else {
      canvas = document.createElement("canvas");
      this.fallbackCanvas = canvas;
      this.fallbackContext = canvas.getContext("2d", { alpha: false });
      if (!this.fallbackContext) throw new Error("Neither WebGL nor Canvas2D is available.");
      this.host.appendChild(canvas);
      this.resizeFallback();
    }
    canvas.setAttribute("aria-label", "Live neural cellular automata field");
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
    this.broadcastLayer.clear();
    this.eventLayer.clear();
    this.cellLayer.clear();
    this.markerLayer.clear();
    this.pulseLayer.clear();
    this.drawEdges(snapshot, geometry);
    this.drawEvents(snapshot, geometry);
    this.drawCells(snapshot, geometry);
    this.drawBroadcasts(snapshot, geometry);
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
    const activationIndex = this.channelIndex(snapshot, "activation");
    for (let index = 0; index < edges.source.length; index += 1) {
      const weight = edges.weight[index] ?? 0;
      if (Math.abs(weight) < this.edgeThreshold) continue;
      const source = edges.source[index];
      const destination = edges.destination[index];
      if (source === undefined || destination === undefined) continue;
      const start = this.cellCenter(source, snapshot, geometry);
      const end = this.cellCenter(destination, snapshot, geometry);
      const age = edges.age[index] ?? 100;
      const young = age < this.newEdgeAge(snapshot);
      const color = young ? COLORS.newborn : weight >= 0 ? COLORS.positive : COLORS.negative;
      const alpha = young ? 0.78 : Math.min(0.54, 0.13 + Math.abs(weight) * 0.26);
      const width = young ? 1.7 : 0.45 + Math.min(2.1, Math.abs(weight) * 1.2);
      this.edgeLayer.moveTo(start.x, start.y).lineTo(end.x, end.y).stroke({ color, alpha, width });

      const activation = snapshot.field.cells[source]?.[activationIndex] ?? 0;
      if (Math.abs(activation) > 0.12 && index % 2 === snapshot.tick % 2) {
        const progress = (snapshot.tick * 0.055 + index * 0.137) % 1;
        const x = start.x + (end.x - start.x) * progress;
        const y = start.y + (end.y - start.y) * progress;
        this.pulseLayer.circle(x, y, Math.max(1.1, geometry.cell * 0.085)).fill({
          color: COLORS.signal,
          alpha: Math.min(0.95, 0.5 + Math.abs(activation) * 0.45),
        });
      }
    }
  }

  private drawEvents(snapshot: ExperimentSnapshot, geometry: DishGeometry): void {
    for (const event of snapshot.events) {
      const age = snapshot.tick - event.tick;
      if (age < 0 || age > 36) continue;
      const start = this.cellCenter(event.source, snapshot, geometry);
      const end = this.cellCenter(event.destination, snapshot, geometry);
      this.eventLayer.moveTo(start.x, start.y).lineTo(end.x, end.y).stroke({
        color: event.type === "grown" ? COLORS.newborn : COLORS.prune,
        alpha: Math.max(0, 0.65 * (1 - age / 36)),
        width: event.type === "grown" ? 1.8 : 1.4,
      });
    }
  }

  private drawBroadcasts(snapshot: ExperimentSnapshot, geometry: DishGeometry): void {
    if (snapshot.experiment !== "mnist") return;
    const axonIndex = this.channelIndex(snapshot, "axon_growth");
    const receptorIndex = this.channelIndex(snapshot, "dendrite_growth");
    for (let index = 0; index < snapshot.field.cells.length; index += 1) {
      const cell = snapshot.field.cells[index];
      if (!cell) continue;
      const axon = cell[axonIndex] ?? 0;
      const receptor = cell[receptorIndex] ?? 0;
      if (Math.max(axon, receptor) < 0.62) continue;
      const center = this.cellCenter(index, snapshot, geometry);
      if (axon >= 0.62) {
        this.broadcastLayer.circle(center.x, center.y, geometry.cell * (0.35 + 0.36 * axon)).stroke({
          color: 0x5ad9ff,
          alpha: Math.min(0.34, (axon - 0.58) * 0.8),
          width: 1,
        });
      }
      if (receptor >= 0.62) {
        this.broadcastLayer.circle(center.x, center.y, geometry.cell * (0.28 + 0.28 * receptor)).stroke({
          color: 0xc28cff,
          alpha: Math.min(0.3, (receptor - 0.58) * 0.72),
          width: 1,
        });
      }
    }
  }

  private newEdgeAge(snapshot: ExperimentSnapshot): number {
    return snapshot.experiment === "mnist" ? 2 : 36;
  }

  private drawCells(snapshot: ExperimentSnapshot, geometry: DishGeometry): void {
    const aliveIndex = this.channelIndex(snapshot, "alive");
    const sensorIndex = this.channelIndex(snapshot, "sensor_id");
    const motorIndex = this.channelIndex(snapshot, "motor_id");
    const gap = Math.max(0.3, geometry.cell * 0.045);
    for (let index = 0; index < snapshot.field.cells.length; index += 1) {
      const cell = snapshot.field.cells[index];
      if (!cell) continue;
      const xIndex = index % snapshot.field.width;
      const yIndex = Math.floor(index / snapshot.field.width);
      const x = geometry.x + xIndex * geometry.cell + gap;
      const y = geometry.y + yIndex * geometry.cell + gap;
      const alive = cell[aliveIndex] ?? 0;
      const color = this.cellColor(cell, snapshot);
      this.cellLayer.rect(x, y, geometry.cell - 2 * gap, geometry.cell - 2 * gap).fill({
        color,
        alpha: 0.16 + 0.84 * alive,
      });

      const sensor = cell[sensorIndex] ?? 0;
      const motor = cell[motorIndex] ?? 0;
      if (sensor !== 0 || motor !== 0) {
        this.markerLayer.rect(x, y, geometry.cell - 2 * gap, geometry.cell - 2 * gap).stroke({
          color: sensor !== 0 ? COLORS.sensor : COLORS.motor,
          alpha: 0.95,
          width: Math.max(1, geometry.cell * 0.1),
        });
      }
      if (this.selectedCell === index) {
        this.markerLayer.rect(x - 1, y - 1, geometry.cell - 2 * gap + 2, geometry.cell - 2 * gap + 2).stroke({
          color: COLORS.selected,
          alpha: 1,
          width: 2,
        });
      }
    }
  }

  private cellColor(cell: number[], snapshot: ExperimentSnapshot): number {
    const value = (name: string): number => cell[this.channelIndex(snapshot, name)] ?? 0;
    const activation = value("activation");
    if (this.layer === "phase") {
      const phase = Math.atan2(value("phase_sin"), value("phase_cos"));
      const hue = ((phase / (2 * Math.PI)) * 360 + 360) % 360;
      return hslToHex(hue, 68, 25 + Math.min(37, Math.abs(activation) * 42));
    }
    if (this.layer === "activation") {
      const magnitude = Math.min(1, Math.abs(activation));
      return blend(0x102225, activation >= 0 ? COLORS.positive : COLORS.negative, 0.18 + magnitude * 0.82);
    }
    if (this.layer === "energy") return blend(0x101c1c, 0x7ee787, Math.max(0, Math.min(1, value("energy"))));
    if (this.layer === "reward_trace") {
      const reward = value("reward_trace");
      return blend(0x16181d, reward >= 0 ? 0xf1c75b : 0xd65a87, Math.min(1, Math.abs(reward)));
    }
    if (this.layer === "alive") return blend(0x10191a, 0xd8f7ee, Math.max(0, Math.min(1, value("alive"))));
    const growth = Math.max(0, Math.min(1, value(this.layer)));
    return blend(0x111b20, this.layer === "axon_growth" ? 0x5ad9ff : 0xc28cff, growth);
  }

  private channelIndex(snapshot: ExperimentSnapshot, name: string): number {
    const index = snapshot.field.channels.indexOf(name);
    return index >= 0 ? index : 0;
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
    const scaleX = this.renderWidth / bounds.width;
    const scaleY = this.renderHeight / bounds.height;
    const canvasX = (event.clientX - bounds.left) * scaleX;
    const canvasY = (event.clientY - bounds.top) * scaleY;
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
    const activationIndex = this.channelIndex(snapshot, "activation");
    const edges = snapshot.edges;
    context.lineCap = "round";
    for (let index = 0; index < edges.source.length; index += 1) {
      const weight = edges.weight[index] ?? 0;
      if (Math.abs(weight) < this.edgeThreshold) continue;
      const source = edges.source[index];
      const destination = edges.destination[index];
      if (source === undefined || destination === undefined) continue;
      const start = this.cellCenter(source, snapshot, geometry);
      const end = this.cellCenter(destination, snapshot, geometry);
      const young = (edges.age[index] ?? 100) < this.newEdgeAge(snapshot);
      context.globalAlpha = young ? 0.78 : Math.min(0.54, 0.13 + Math.abs(weight) * 0.26);
      context.strokeStyle = cssColor(young ? COLORS.newborn : weight >= 0 ? COLORS.positive : COLORS.negative);
      context.lineWidth = young ? 1.7 : 0.45 + Math.min(2.1, Math.abs(weight) * 1.2);
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();

      const activation = snapshot.field.cells[source]?.[activationIndex] ?? 0;
      if (Math.abs(activation) > 0.12 && index % 2 === snapshot.tick % 2) {
        const progress = (snapshot.tick * 0.055 + index * 0.137) % 1;
        context.globalAlpha = Math.min(0.95, 0.5 + Math.abs(activation) * 0.45);
        context.fillStyle = cssColor(COLORS.signal);
        context.beginPath();
        context.arc(
          start.x + (end.x - start.x) * progress,
          start.y + (end.y - start.y) * progress,
          Math.max(1.1, geometry.cell * 0.085),
          0,
          2 * Math.PI,
        );
        context.fill();
      }
    }

    for (const event of snapshot.events) {
      const age = snapshot.tick - event.tick;
      if (age < 0 || age > 36) continue;
      const start = this.cellCenter(event.source, snapshot, geometry);
      const end = this.cellCenter(event.destination, snapshot, geometry);
      context.globalAlpha = Math.max(0, 0.65 * (1 - age / 36));
      context.strokeStyle = cssColor(event.type === "grown" ? COLORS.newborn : COLORS.prune);
      context.lineWidth = event.type === "grown" ? 1.8 : 1.4;
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
    }

    const aliveIndex = this.channelIndex(snapshot, "alive");
    const sensorIndex = this.channelIndex(snapshot, "sensor_id");
    const motorIndex = this.channelIndex(snapshot, "motor_id");
    const gap = Math.max(0.3, geometry.cell * 0.045);
    for (let index = 0; index < snapshot.field.cells.length; index += 1) {
      const cell = snapshot.field.cells[index];
      if (!cell) continue;
      const x = geometry.x + (index % snapshot.field.width) * geometry.cell + gap;
      const y = geometry.y + Math.floor(index / snapshot.field.width) * geometry.cell + gap;
      const size = geometry.cell - 2 * gap;
      context.globalAlpha = 0.16 + 0.84 * (cell[aliveIndex] ?? 0);
      context.fillStyle = cssColor(this.cellColor(cell, snapshot));
      context.fillRect(x, y, size, size);
      const sensor = cell[sensorIndex] ?? 0;
      const motor = cell[motorIndex] ?? 0;
      if (sensor !== 0 || motor !== 0 || this.selectedCell === index) {
        context.globalAlpha = 0.95;
        context.strokeStyle = cssColor(
          this.selectedCell === index ? COLORS.selected : sensor !== 0 ? COLORS.sensor : COLORS.motor,
        );
        context.lineWidth = this.selectedCell === index ? 2 : Math.max(1, geometry.cell * 0.1);
        context.strokeRect(x, y, size, size);
      }
      if (snapshot.experiment === "mnist") {
        const axon = cell[this.channelIndex(snapshot, "axon_growth")] ?? 0;
        const receptor = cell[this.channelIndex(snapshot, "dendrite_growth")] ?? 0;
        if (Math.max(axon, receptor) >= 0.62) {
          const center = this.cellCenter(index, snapshot, geometry);
          context.globalAlpha = Math.min(0.3, (Math.max(axon, receptor) - 0.58) * 0.75);
          context.strokeStyle = cssColor(axon >= receptor ? 0x5ad9ff : 0xc28cff);
          context.lineWidth = 1;
          context.beginPath();
          context.arc(center.x, center.y, geometry.cell * 0.62, 0, 2 * Math.PI);
          context.stroke();
        }
      }
    }
    this.drawFallbackEdgeOverlay(snapshot, geometry, context, activationIndex);
    context.globalAlpha = 1;
  }

  private drawFallbackEdgeOverlay(
    snapshot: ExperimentSnapshot,
    geometry: DishGeometry,
    context: CanvasRenderingContext2D,
    activationIndex: number,
  ): void {
    const edges = snapshot.edges;
    for (let index = 0; index < edges.source.length; index += 1) {
      const weight = edges.weight[index] ?? 0;
      if (Math.abs(weight) < this.edgeThreshold) continue;
      const source = edges.source[index];
      const destination = edges.destination[index];
      if (source === undefined || destination === undefined) continue;
      const start = this.cellCenter(source, snapshot, geometry);
      const end = this.cellCenter(destination, snapshot, geometry);
      const young = (edges.age[index] ?? 100) < this.newEdgeAge(snapshot);
      context.globalAlpha = young ? 0.72 : Math.min(0.48, 0.1 + Math.abs(weight) * 0.22);
      context.strokeStyle = cssColor(young ? COLORS.newborn : weight >= 0 ? COLORS.positive : COLORS.negative);
      context.lineWidth = young ? 1.5 : 0.4 + Math.min(1.8, Math.abs(weight));
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();

      const activation = snapshot.field.cells[source]?.[activationIndex] ?? 0;
      if (Math.abs(activation) > 0.12 && index % 2 === snapshot.tick % 2) {
        const progress = (snapshot.tick * 0.055 + index * 0.137) % 1;
        context.globalAlpha = Math.min(0.95, 0.5 + Math.abs(activation) * 0.45);
        context.fillStyle = cssColor(COLORS.signal);
        context.beginPath();
        context.arc(
          start.x + (end.x - start.x) * progress,
          start.y + (end.y - start.y) * progress,
          Math.max(1.1, geometry.cell * 0.085),
          0,
          2 * Math.PI,
        );
        context.fill();
      }
    }
    for (const event of snapshot.events) {
      const age = snapshot.tick - event.tick;
      if (age < 0 || age > 36) continue;
      const start = this.cellCenter(event.source, snapshot, geometry);
      const end = this.cellCenter(event.destination, snapshot, geometry);
      context.globalAlpha = Math.max(0, 0.65 * (1 - age / 36));
      context.strokeStyle = cssColor(event.type === "grown" ? COLORS.newborn : COLORS.prune);
      context.lineWidth = event.type === "grown" ? 1.8 : 1.4;
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
    }
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

function blend(from: number, to: number, amount: number): number {
  const t = Math.max(0, Math.min(1, amount));
  const red = Math.round(((from >> 16) & 255) * (1 - t) + ((to >> 16) & 255) * t);
  const green = Math.round(((from >> 8) & 255) * (1 - t) + ((to >> 8) & 255) * t);
  const blue = Math.round((from & 255) * (1 - t) + (to & 255) * t);
  return (red << 16) | (green << 8) | blue;
}

function hslToHex(hue: number, saturation: number, lightness: number): number {
  const s = saturation / 100;
  const l = lightness / 100;
  const chroma = (1 - Math.abs(2 * l - 1)) * s;
  const x = chroma * (1 - Math.abs(((hue / 60) % 2) - 1));
  const match = l - chroma / 2;
  let red = 0;
  let green = 0;
  let blue = 0;
  if (hue < 60) [red, green] = [chroma, x];
  else if (hue < 120) [red, green] = [x, chroma];
  else if (hue < 180) [green, blue] = [chroma, x];
  else if (hue < 240) [green, blue] = [x, chroma];
  else if (hue < 300) [red, blue] = [x, chroma];
  else [red, blue] = [chroma, x];
  return (
    Math.round((red + match) * 255) << 16
    | Math.round((green + match) * 255) << 8
    | Math.round((blue + match) * 255)
  );
}

function cssColor(color: number): string {
  return `#${color.toString(16).padStart(6, "0")}`;
}
