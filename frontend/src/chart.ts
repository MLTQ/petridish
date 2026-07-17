import type { ExperimentSnapshot } from "./protocol";

export class HistoryChart {
  private readonly rewards: number[] = [];
  private readonly accuracies: number[] = [];
  private experiment: ExperimentSnapshot["experiment"] | null = null;

  public constructor(
    private readonly rewardLine: SVGPolylineElement,
    private readonly accuracyLine: SVGPolylineElement,
    private readonly svg: SVGSVGElement,
  ) {}

  public update(snapshot: ExperimentSnapshot): void {
    if (this.experiment !== snapshot.experiment) this.reset(snapshot.experiment);
    const objective = snapshot.experiment === "mnist"
      ? 1 - Math.min(1, (snapshot.metrics.loss ?? Math.log(10)) / Math.log(10))
      : snapshot.metrics.rollingReward;
    this.rewards.push(objective);
    this.accuracies.push(snapshot.task.accuracy);
    if (this.rewards.length > 160) this.rewards.shift();
    if (this.accuracies.length > 160) this.accuracies.shift();
    this.rewardLine.setAttribute("points", this.points(this.rewards, -0.25, 1));
    this.accuracyLine.setAttribute("points", this.points(this.accuracies, 0, 1));
    this.svg.setAttribute(
      "aria-label",
      snapshot.experiment === "mnist"
        ? `Normalized loss objective ${objective.toFixed(3)}; training accuracy ${(snapshot.task.accuracy * 100).toFixed(0)} percent`
        : `Reward ${snapshot.metrics.rollingReward.toFixed(3)}; accuracy ${(snapshot.task.accuracy * 100).toFixed(0)} percent`,
    );
  }

  private reset(experiment: ExperimentSnapshot["experiment"]): void {
    this.experiment = experiment;
    this.rewards.length = 0;
    this.accuracies.length = 0;
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
