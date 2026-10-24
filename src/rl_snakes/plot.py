"""Plot every metric in a run's metrics.jsonl.

`uv run snake-plot runs/<run>` writes metrics.png; add `--live` to watch it in a window instead.
The run's plot.json sets the layout: groups under "stacked" are drawn as stacked areas (shares of a whole),
and the "main" group gets a full-width plot on top.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

SMOOTH = 0.9  # EMA factor for the smoothed line drawn over the raw values
COLS = 3
DPI = 100


def load(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {metric: (episodes, values)}; metrics may appear in only some rows."""
    series: dict[str, tuple[list, list]] = {}
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {}
    with open(path) as f:
        for line in f:
            if not line.endswith("\n"):  # the logger may be mid-write
                break
            row = json.loads(line)
            episode = row.pop("episode")
            for key, value in row.items():
                xs, ys = series.setdefault(key, ([], []))
                xs.append(episode)
                ys.append(value)
    return {k: (np.array(xs), np.array(ys)) for k, (xs, ys) in series.items()}


def load_layout(run_dir: Path) -> dict:
    path = run_dir / "plot.json"
    layout = json.loads(path.read_text()) if path.exists() else {}
    return {"stacked": set(layout.get("stacked", [])), "main": layout.get("main")}


def ema(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values)
    acc = values[0]
    for i, v in enumerate(values):
        acc = SMOOTH * acc + (1 - SMOOTH) * v
        out[i] = acc
    return out


def draw(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    stacked: set[str] = frozenset(),
    main: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    # "death/wall" and "death/self" share the "death" subplot; a plain key gets one to itself.
    groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for key in sorted(series):
        group, _, line = key.partition("/")
        groups.setdefault(group, {})[line] = series[key]

    # The main group spans the full width and two rows' height; the rest fill a grid below it.
    top = 2 if main in groups else 0
    rest = [g for g in groups if g != main]
    rows = top + math.ceil(len(rest) / COLS)
    fig = Figure(figsize=figsize or (5 * COLS, 3 * rows), dpi=DPI, layout="constrained")
    grid = fig.add_gridspec(rows, COLS)
    slots = {g: grid[top + i // COLS, i % COLS] for i, g in enumerate(rest)}
    if top:
        slots = {main: grid[:top, :], **slots}

    for group, slot in slots.items():
        lines = groups[group]
        ax = fig.add_subplot(slot)
        if group in stacked:
            # Raw values, no EMA, logged together on the same episodes; normalized so counts work too.
            xs = next(iter(lines.values()))[0]
            ys = np.stack([ys for _, ys in lines.values()])
            shares = ys / np.maximum(ys.sum(axis=0), 1e-12)
            ax.stackplot(xs, *shares, labels=list(lines), alpha=0.8)
            ax.set_ylim(0, 1)
            ax.set_xlim(xs[0], xs[-1])
        else:
            for j, (line, (xs, ys)) in enumerate(lines.items()):
                color = f"C{j}"
                ax.plot(xs, ys, alpha=0.3, linewidth=0.8, color=color)
                ax.plot(xs, ema(ys), linewidth=1.5, color=color, label=line or None)
        if len(lines) > 1:
            ax.legend(fontsize="small")
        ax.set_title(group)
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
    return fig


def plot_run(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "metrics.png"
    series = load(run_dir)
    if series:
        draw(series, **load_layout(run_dir)).savefig(out)
    return out


def live(run_dir: str | Path, size: tuple[int, int] = (1500, 900)) -> None:
    """Show the run's metrics in a resizable window, redrawn whenever metrics.jsonl changes."""
    import pygame

    run_dir = Path(run_dir)
    path = run_dir / "metrics.jsonl"
    pygame.init()
    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    pygame.display.set_caption(f"metrics: {run_dir.name}")
    screen.fill("white")
    pygame.display.flip()

    last_mtime, dirty = None, True
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.VIDEORESIZE:
                dirty = True

        mtime = path.stat().st_mtime if path.exists() else None
        if mtime != last_mtime or dirty:
            last_mtime, dirty = mtime, False
            series = load(run_dir)
            if series:
                w, h = screen.get_size()
                canvas = FigureCanvasAgg(draw(series, **load_layout(run_dir), figsize=(w / DPI, h / DPI)))
                canvas.draw()
                image = pygame.image.frombuffer(canvas.buffer_rgba(), canvas.get_width_height(), "RGBA")
                screen.blit(image, (0, 0))
                pygame.display.flip()
        time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--live", action="store_true", help="watch the run in a window instead of writing a png")
    args = parser.parse_args()
    if args.live:
        live(args.run_dir)
    else:
        print(plot_run(args.run_dir))


if __name__ == "__main__":
    main()
