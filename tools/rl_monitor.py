#!/usr/bin/env python3
"""
rl_monitor -- Real-time CLI dashboard for Prime-RL + Dynamo smoke tests.

Watches orchestrator and trainer log files and optionally rollout batch files
to display a live summary of RL training progress.

Usage:
    python tools/rl_monitor.py --output-dir /tmp/dynamo_smoke_outputs
    python tools/rl_monitor.py --orch-log /tmp/smoke_orchestrator.log --trainer-log /tmp/smoke_trainer.log
"""
import argparse
import json
import os
import re
import struct
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Optional rich import for pretty display
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


@dataclass
class StepData:
    step: int = 0
    time_s: float = 0.0
    reward: float = 0.0
    seq_len: float = 0.0
    async_level: int = 0
    off_policy: int = 0


@dataclass
class TrainerStepData:
    step: int = 0
    time_s: float = 0.0
    loss: float = 0.0
    entropy: float = 0.0
    mismatch_kl: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    throughput: float = 0.0
    mfu: float = 0.0
    peak_mem_gib: float = 0.0


@dataclass
class RLMonitorState:
    # Orchestrator
    orch_steps: list[StepData] = field(default_factory=list)
    orch_status: str = "not started"
    orch_paused_reason: str = ""
    orch_max_steps: int = 0
    orch_exit_code: int | None = None

    # Trainer
    trainer_steps: list[TrainerStepData] = field(default_factory=list)
    trainer_status: str = "not started"
    trainer_exit_code: int | None = None

    # Dynamo
    dynamo_healthy: bool = False
    dynamo_model: str = ""

    # Rollouts
    rollout_files: list[str] = field(default_factory=list)
    broadcast_files: list[str] = field(default_factory=list)

    # Weight updates
    weight_updates: list[dict] = field(default_factory=list)


# --- Log Parsers ---

# Orchestrator patterns
RE_ORCH_STEP = re.compile(
    r"Step (\d+) \| Time: ([\d.]+)s \| Reward: ([\d.]+) \| Seq\. Length: ([\d.]+) tokens/sample"
    r"(?: \| Async Level: (\d+))?(?: \| Max\. Off-Policy Level: (\d+))?"
)
RE_ORCH_PAUSED = re.compile(r"Orchestrator paused: (.+)")
RE_ORCH_RESUMED = re.compile(r"Orchestrator resumed: (.+)")
RE_ORCH_LOOP = re.compile(r"Starting orchestrator loop \(max_steps=(\d+)\)")
RE_ORCH_WEIGHT_PAUSE = re.compile(r"Pausing inference engines for weight update")
RE_ORCH_WEIGHT_RESUME = re.compile(r"All inference engines resumed")
RE_ORCH_FINISHED = re.compile(r"Orchestrator finished")
RE_ORCH_FATAL = re.compile(r"Fatal error in orchestrate")

# Trainer patterns
RE_TRAINER_STEP = re.compile(
    r"Step (\d+) \| Time: ([\d.]+)s \| Loss: ([\d.]+)"
    r" \| Entropy: ([\d.]+)"
    r" \| Mismatch KL: ([\d.]+)"
    r" \| Grad\. Norm: ([\d.]+)"
    r" \| LR: ([\d.e+-]+)"
    r" \| Throughput: ([\d.]+) tokens/s"
    r" \| MFU: ([\d.]+)%"
    r" \| Peak Mem\.: ([\d.]+) GiB"
)
RE_TRAINER_LOOP = re.compile(r"Starting training loop \(max_steps=(\d+)\)")
RE_TRAINER_FATAL = re.compile(r"Fatal error in train")


def parse_orch_log(log_path: Path, state: RLMonitorState):
    """Parse the orchestrator log file and update state."""
    if not log_path.exists():
        return

    state.orch_status = "running"
    text = log_path.read_text(errors="replace")

    m = RE_ORCH_LOOP.search(text)
    if m:
        state.orch_max_steps = int(m.group(1))

    state.orch_steps.clear()
    for m in RE_ORCH_STEP.finditer(text):
        state.orch_steps.append(
            StepData(
                step=int(m.group(1)),
                time_s=float(m.group(2)),
                reward=float(m.group(3)),
                seq_len=float(m.group(4)),
                async_level=int(m.group(5)) if m.group(5) else 0,
                off_policy=int(m.group(6)) if m.group(6) else 0,
            )
        )

    # Check paused/resumed
    state.orch_paused_reason = ""
    paused_matches = list(RE_ORCH_PAUSED.finditer(text))
    resumed_matches = list(RE_ORCH_RESUMED.finditer(text))
    if paused_matches and (not resumed_matches or paused_matches[-1].start() > resumed_matches[-1].start()):
        state.orch_paused_reason = paused_matches[-1].group(1)
        state.orch_status = "paused"

    # Track weight updates
    state.weight_updates.clear()
    for m in RE_ORCH_WEIGHT_PAUSE.finditer(text):
        state.weight_updates.append({"action": "pause", "pos": m.start()})
    for m in RE_ORCH_WEIGHT_RESUME.finditer(text):
        state.weight_updates.append({"action": "resume", "pos": m.start()})

    if RE_ORCH_FINISHED.search(text):
        state.orch_status = "finished"
        state.orch_exit_code = 0
    elif RE_ORCH_FATAL.search(text):
        state.orch_status = "FATAL ERROR"
        state.orch_exit_code = 1


def parse_trainer_log(log_path: Path, state: RLMonitorState):
    """Parse the trainer log file and update state."""
    if not log_path.exists():
        return

    state.trainer_status = "running"
    text = log_path.read_text(errors="replace")

    state.trainer_steps.clear()
    for m in RE_TRAINER_STEP.finditer(text):
        state.trainer_steps.append(
            TrainerStepData(
                step=int(m.group(1)),
                time_s=float(m.group(2)),
                loss=float(m.group(3)),
                entropy=float(m.group(4)),
                mismatch_kl=float(m.group(5)),
                grad_norm=float(m.group(6)),
                lr=float(m.group(7)),
                throughput=float(m.group(8)),
                mfu=float(m.group(9)),
                peak_mem_gib=float(m.group(10)),
            )
        )

    if RE_TRAINER_FATAL.search(text):
        state.trainer_status = "FATAL ERROR"
        state.trainer_exit_code = 1
    elif text.strip().endswith("wandb: Find logs at:") or "Shutting down" in text:
        state.trainer_status = "finished"
        state.trainer_exit_code = 0


def scan_rollouts(output_dir: Path, state: RLMonitorState):
    """Scan the output directory for rollout and broadcast files."""
    run_dir = output_dir / "run_default"
    if not run_dir.exists():
        return

    rollout_dir = run_dir / "rollouts"
    broadcast_dir = run_dir / "broadcasts"

    state.rollout_files = sorted(
        [p.parent.name for p in rollout_dir.glob("step_*/rollouts.bin")]
    ) if rollout_dir.exists() else []

    state.broadcast_files = sorted(
        [p.name for p in broadcast_dir.iterdir() if p.is_dir()]
    ) if broadcast_dir.exists() else []


def check_dynamo_health(state: RLMonitorState):
    """Check if Dynamo is healthy (non-blocking HTTP check)."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request("http://localhost:8000/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            state.dynamo_healthy = data.get("status") == "healthy"
    except Exception:
        state.dynamo_healthy = False


# --- Display ---


def render_plain(state: RLMonitorState) -> str:
    """Render state as plain text."""
    lines = []
    lines.append("=" * 72)
    lines.append("  PRIME-RL + DYNAMO  |  RL Training Monitor")
    lines.append("=" * 72)

    # Dynamo status
    dyn_icon = "OK" if state.dynamo_healthy else "DOWN"
    lines.append(f"\n  Dynamo:  [{dyn_icon}] http://localhost:8000")

    # Orchestrator
    orch_progress = f"{len(state.orch_steps)}/{state.orch_max_steps}" if state.orch_max_steps else f"{len(state.orch_steps)}/?"
    lines.append(f"\n  Orchestrator:  [{state.orch_status}]  Steps: {orch_progress}")
    if state.orch_paused_reason:
        lines.append(f"    Paused: {state.orch_paused_reason}")

    if state.orch_steps:
        lines.append(f"\n  {'Step':>5}  {'Time':>8}  {'Reward':>8}  {'Seq Len':>8}  {'Async':>5}  {'OffPol':>6}")
        lines.append(f"  {'----':>5}  {'------':>8}  {'------':>8}  {'-------':>8}  {'-----':>5}  {'------':>6}")
        for s in state.orch_steps:
            lines.append(
                f"  {s.step:>5}  {s.time_s:>7.1f}s  {s.reward:>8.4f}  {s.seq_len:>7.1f}t  {s.async_level:>5}  {s.off_policy:>6}"
            )

    # Reward summary
    if state.orch_steps:
        rewards = [s.reward for s in state.orch_steps]
        lines.append(f"\n  Reward:  avg={sum(rewards)/len(rewards):.4f}  max={max(rewards):.4f}  min={min(rewards):.4f}")

    # Trainer
    trainer_progress = f"{len(state.trainer_steps)}/?" if not state.orch_max_steps else f"{len(state.trainer_steps)}/{state.orch_max_steps}"
    lines.append(f"\n  Trainer:  [{state.trainer_status}]  Steps: {trainer_progress}")

    if state.trainer_steps:
        lines.append(f"\n  {'Step':>5}  {'Time':>8}  {'Loss':>10}  {'Entropy':>8}  {'KL':>8}  {'GradN':>7}  {'tok/s':>7}  {'Mem':>6}")
        lines.append(f"  {'----':>5}  {'------':>8}  {'------':>10}  {'-------':>8}  {'------':>8}  {'-----':>7}  {'-----':>7}  {'---':>6}")
        for s in state.trainer_steps:
            lines.append(
                f"  {s.step:>5}  {s.time_s:>7.1f}s  {s.loss:>10.6f}  {s.entropy:>8.4f}  {s.mismatch_kl:>8.4f}  {s.grad_norm:>7.4f}  {s.throughput:>6.0f}  {s.peak_mem_gib:>5.1f}G"
            )

    # Loss summary
    if state.trainer_steps:
        losses = [s.loss for s in state.trainer_steps]
        lines.append(f"\n  Loss:  avg={sum(losses)/len(losses):.6f}  last={losses[-1]:.6f}")

    # Weight updates
    n_updates = len([w for w in state.weight_updates if w["action"] == "resume"])
    lines.append(f"\n  Weight Updates: {n_updates} completed")

    # Filesystem
    lines.append(f"  Rollout Batches: {', '.join(state.rollout_files) or 'none'}")
    lines.append(f"  Broadcast Dirs:  {', '.join(state.broadcast_files) or 'none'}")

    lines.append(f"\n  Last refresh: {time.strftime('%H:%M:%S')}")
    lines.append("=" * 72)

    return "\n".join(lines)


def render_rich(state: RLMonitorState, console: "Console") -> "Panel":
    """Render state using rich for colored terminal output."""
    from rich.columns import Columns
    from rich.layout import Layout

    # -- Header --
    dyn_color = "green" if state.dynamo_healthy else "red"
    dyn_text = "HEALTHY" if state.dynamo_healthy else "DOWN"

    # -- Orchestrator table --
    orch_table = Table(title="Orchestrator", title_style="bold cyan", expand=True, show_edge=False, pad_edge=False)
    orch_table.add_column("Step", justify="right", style="bold")
    orch_table.add_column("Time", justify="right")
    orch_table.add_column("Reward", justify="right", style="green")
    orch_table.add_column("Seq Len", justify="right")
    orch_table.add_column("Async", justify="right")

    for s in state.orch_steps:
        reward_style = "green bold" if s.reward > 0.1 else "yellow" if s.reward > 0 else "dim"
        orch_table.add_row(str(s.step), f"{s.time_s:.1f}s", f"[{reward_style}]{s.reward:.4f}[/]", f"{s.seq_len:.0f}t", str(s.async_level))

    # -- Trainer table --
    train_table = Table(title="Trainer", title_style="bold magenta", expand=True, show_edge=False, pad_edge=False)
    train_table.add_column("Step", justify="right", style="bold")
    train_table.add_column("Time", justify="right")
    train_table.add_column("Loss", justify="right", style="yellow")
    train_table.add_column("Entropy", justify="right")
    train_table.add_column("KL", justify="right")
    train_table.add_column("GradN", justify="right")
    train_table.add_column("tok/s", justify="right")
    train_table.add_column("Mem", justify="right")

    for s in state.trainer_steps:
        train_table.add_row(
            str(s.step), f"{s.time_s:.1f}s", f"{s.loss:.6f}", f"{s.entropy:.4f}",
            f"{s.mismatch_kl:.4f}", f"{s.grad_norm:.4f}", f"{s.throughput:.0f}", f"{s.peak_mem_gib:.1f}G"
        )

    # -- Status line --
    orch_progress = f"{len(state.orch_steps)}/{state.orch_max_steps}" if state.orch_max_steps else "?"
    trainer_progress = f"{len(state.trainer_steps)}/{state.orch_max_steps}" if state.orch_max_steps else "?"
    n_updates = len([w for w in state.weight_updates if w["action"] == "resume"])

    status_parts = [
        f"[{dyn_color}]Dynamo: {dyn_text}[/]",
        f"Orch: [{state.orch_status}] {orch_progress}",
        f"Train: [{state.trainer_status}] {trainer_progress}",
        f"Wt Updates: {n_updates}",
        f"Rollouts: {len(state.rollout_files)}",
    ]
    if state.orch_paused_reason:
        status_parts.append(f"[yellow]PAUSED: {state.orch_paused_reason[:50]}[/]")

    status_text = "  |  ".join(status_parts)

    # Compose
    group = Table.grid(padding=(1, 0))
    group.add_row(Text(status_text, style="dim"))
    group.add_row(orch_table)
    group.add_row(train_table)

    # Summaries
    if state.orch_steps:
        rewards = [s.reward for s in state.orch_steps]
        group.add_row(Text(
            f"Reward  avg={sum(rewards)/len(rewards):.4f}  max={max(rewards):.4f}  min={min(rewards):.4f}",
            style="green"
        ))
    if state.trainer_steps:
        losses = [s.loss for s in state.trainer_steps]
        group.add_row(Text(
            f"Loss    avg={sum(losses)/len(losses):.6f}  last={losses[-1]:.6f}",
            style="yellow"
        ))

    group.add_row(Text(f"Broadcasts: {', '.join(state.broadcast_files) or 'none'}", style="dim"))
    group.add_row(Text(f"Updated: {time.strftime('%H:%M:%S')}", style="dim"))

    return Panel(group, title="[bold]Prime-RL + Dynamo RL Monitor[/]", border_style="bright_blue")


def main():
    parser = argparse.ArgumentParser(description="RL Training Monitor for Prime-RL + Dynamo")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/dynamo_smoke_outputs"),
                        help="Base output directory (trainer output_dir)")
    parser.add_argument("--orch-log", type=Path, default=Path("/tmp/smoke_orchestrator.log"),
                        help="Path to orchestrator log file")
    parser.add_argument("--trainer-log", type=Path, default=Path("/tmp/smoke_trainer.log"),
                        help="Path to trainer log file")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Refresh interval in seconds")
    parser.add_argument("--no-rich", action="store_true",
                        help="Force plain text output (no colors)")
    parser.add_argument("--once", action="store_true",
                        help="Print once and exit (no live refresh)")
    args = parser.parse_args()

    state = RLMonitorState()
    use_rich = HAS_RICH and not args.no_rich and sys.stdout.isatty()

    def refresh():
        parse_orch_log(args.orch_log, state)
        parse_trainer_log(args.trainer_log, state)
        scan_rollouts(args.output_dir, state)
        check_dynamo_health(state)

    if args.once:
        refresh()
        if use_rich:
            console = Console()
            console.print(render_rich(state, console))
        else:
            print(render_plain(state))
        return

    if use_rich:
        console = Console()
        with Live(render_rich(state, console), console=console, refresh_per_second=1, screen=True) as live:
            try:
                while True:
                    refresh()
                    live.update(render_rich(state, console))
                    # Exit when both processes are done
                    if state.orch_status in ("finished", "FATAL ERROR") and state.trainer_status in (
                        "finished",
                        "FATAL ERROR",
                    ):
                        time.sleep(1)  # One final refresh
                        refresh()
                        live.update(render_rich(state, console))
                        break
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                pass
    else:
        try:
            while True:
                refresh()
                os.system("clear" if os.name == "posix" else "cls")
                print(render_plain(state))
                if state.orch_status in ("finished", "FATAL ERROR") and state.trainer_status in (
                    "finished",
                    "FATAL ERROR",
                ):
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
