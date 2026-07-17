import type { ExperimentCommand, ExperimentSnapshot, ServerMessage } from "./protocol";

type SnapshotHandler = (snapshot: ExperimentSnapshot) => void;
type StatusHandler = (status: "connecting" | "connected" | "disconnected") => void;
type ErrorHandler = (message: string) => void;

export class ExperimentSocket {
  private socket: WebSocket | null = null;
  private retryTimer: number | null = null;
  private closed = false;

  public constructor(
    private readonly onSnapshot: SnapshotHandler,
    private readonly onStatus: StatusHandler,
    private readonly onError: ErrorHandler,
  ) {}

  public connect(): void {
    this.closed = false;
    this.onStatus("connecting");
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    this.socket = new WebSocket(`${protocol}//${window.location.host}/ws`);
    this.socket.addEventListener("open", () => this.onStatus("connected"));
    this.socket.addEventListener("message", (event) => this.receive(String(event.data)));
    this.socket.addEventListener("close", () => {
      this.onStatus("disconnected");
      if (!this.closed) {
        this.retryTimer = window.setTimeout(() => this.connect(), 1200);
      }
    });
    this.socket.addEventListener("error", () => this.onError("The experiment stream could not connect."));
  }

  public send(command: ExperimentCommand): void {
    if (this.socket?.readyState !== WebSocket.OPEN) {
      this.onError("The experiment server is not connected yet.");
      return;
    }
    this.socket.send(JSON.stringify(command));
  }

  public close(): void {
    this.closed = true;
    if (this.retryTimer !== null) window.clearTimeout(this.retryTimer);
    this.socket?.close();
  }

  private receive(raw: string): void {
    try {
      const message = JSON.parse(raw) as ServerMessage;
      if (message.type === "snapshot") this.onSnapshot(message);
      else this.onError(message.message);
    } catch {
      this.onError("The server sent an unreadable snapshot.");
    }
  }
}
