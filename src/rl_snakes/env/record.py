from pathlib import Path

import gymnasium as gym
from PIL import Image


class RecordGif(gym.Wrapper):
    """Save every episode of the wrapped env as a GIF at `path`.

    Works with `render_mode="human"` (records what the window shows) or `"rgb_array"`
    (headless). A `{episode}` placeholder in `path` is filled with the episode count
    of this wrapper; without it, each episode overwrites the previous GIF.
    """

    def __init__(self, env: gym.Env, path: str | Path, fps: int | None = None):
        super().__init__(env)
        assert env.render_mode is not None, "RecordGif needs a render_mode"
        self.path = str(path)
        self.fps = fps or env.metadata["render_fps"]
        self.frames = []
        self.episode = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [self._frame()]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._frame())
        if terminated or truncated:
            self.save()
            self.episode += 1
        return obs, reward, terminated, truncated, info

    def save(self) -> Path:
        path = Path(self.path.format(episode=self.episode))
        path.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(frame) for frame in self.frames]
        # Hold the final frame a bit longer so the death is visible before the loop restarts.
        durations = [1000 // self.fps] * (len(images) - 1) + [1000]
        images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)
        return path

    def _frame(self):
        if self.env.render_mode != "human":  # human mode already drew this step
            self.env.render()
        return self.env.unwrapped._renderer.frame()
