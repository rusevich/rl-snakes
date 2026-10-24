"""Append training metrics to `runs/<run>/metrics.jsonl` and periodically redraw `metrics.png`.

With `live=True` (the default) a separate window also shows the metrics as they come in.
Groups named in `stacked` (e.g. `["death"]`) are drawn as stacked areas instead of lines,
and the `main` group (e.g. `"return"`) gets a full-width plot on top.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from rl_snakes.plot import plot_run


class Logger:
    def __init__(
        self,
        run_dir: str | Path | None = None,
        plot_every: int = 50,
        live: bool = True,
        stacked: list[str] = (),
        main: str | None = None,
    ):
        self.run_dir = Path(run_dir or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Saved with the run, so the live window and later `snake-plot` calls draw it the same way.
        (self.run_dir / "plot.json").write_text(json.dumps({"stacked": list(stacked), "main": main}))
        self.file = open(self.run_dir / "metrics.jsonl", "a")
        self.plot_every = plot_every
        self.n_logs = 0
        if live:
            # Its own process, so the window stays responsive and doesn't share pygame with the env renderer.
            subprocess.Popen([sys.executable, "-m", "rl_snakes.plot", "--live", str(self.run_dir)])

    def log(self, episode: int, **metrics: float | dict[str, float]) -> None:
        """Record scalar metrics at x-axis position `episode`; any new key gets its own subplot.

        A dict value, e.g. `death={"wall": 0.5, "self": 0.5}`, is stored as `death/wall`, `death/self`
        and drawn as one subplot with a line per entry.
        """
        row = {"episode": episode}
        for key, value in metrics.items():
            if isinstance(value, dict):
                row.update({f"{key}/{k}": float(v) for k, v in value.items()})
            else:
                row[key] = float(value)
        self.file.write(json.dumps(row) + "\n")
        self.file.flush()
        self.n_logs += 1
        if self.n_logs % self.plot_every == 0:
            self.plot()

    def plot(self) -> None:
        plot_run(self.run_dir)

    def close(self) -> None:
        self.file.close()
        self.plot()
