import type { ExperimentSnapshot } from "./protocol";

export class HistoryChart {
  private readonly rewards: number[] = [];
  private readonly accuracies: number[] = [];

  public constructor(
    private readonly rewardLine: SVGPolylineElement,
    private readonly accuracyLine: SVGPolylineElement,
    private readonly svg: SVGSVGElement,
  ) {}

  public update(snapshot: ExperimentSnapshot): void {
    const objective = 1 - Math.min(1, (snapshot.metrics.loss ?? Math.log(10)) / Math.log(10));
    this.rewards.push(objective);
    this.accuracies.push(snapshot.task.accuracy);
    if (this.rewards.length > 160) this.rewards.shift();
    if (this.accuracies.length > 160) this.accuracies.shift();
    this.rewardLine.setAttribute("points", this.points(this.rewards, -0.25, 1));
    this.accuracyLine.setAttribute("points", this.points(this.accuracies, 0, 1));
    this.svg.setAttribute(
      "aria-label",
      `Normalized loss objective ${objective.toFixed(3)}; training accuracy ${(snapshot.task.accuracy * 100).toFixed(0)} percent`,
    );
  }

  private points(values: number[], minimum: number, maximum: number): string {
    if (values.length === 0) return "";
    const span = maximum - minimum;
    return values
      .map((value, index) => {
        const x = values.length === 1 ? 300 : (index / (values.length - 1)) * 300;
        const normalized = Math.max(0, Math.min(1, (value - minimum) / span));
        return `${x.toFixed(1)},${(86 - normalized * 80).toFixed(1)}`;
      })
      .join(" ");
  }
}
