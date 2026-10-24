import numpy as np
import pygame

from rl_snakes.env.snake import DIRECTIONS

BACKGROUND = (17, 17, 27)
TILE_DARK = (30, 30, 46)
TILE_LIGHT = (36, 36, 54)
TEXT = (205, 214, 244)
SUBTEXT = (127, 132, 156)
HEAD = (166, 227, 161)
TAIL = (64, 140, 90)
APPLE = (243, 139, 168)
LEAF = (166, 227, 161)
EYE = (245, 245, 250)
PUPIL = (17, 17, 27)

HUD_HEIGHT = 44
MARGIN = 12


def _lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


class SnakeRenderer:
    def __init__(self, width: int, height: int, mode: str, fps: int):
        self.width, self.height, self.mode, self.fps = width, height, mode, fps
        self.cell = max(16, min(40, 640 // max(width, height)))
        size = (width * self.cell + 2 * MARGIN, height * self.cell + HUD_HEIGHT + MARGIN)

        if mode == "human":
            pygame.display.init()
            pygame.display.set_caption("Snake")
            # reuse the open window: calling set_mode again nudges it on some WMs
            surface = pygame.display.get_surface()
            self.surface = surface if surface is not None and surface.get_size() == size else pygame.display.set_mode(size)
            self.clock = pygame.time.Clock()
        else:
            self.surface = pygame.Surface(size)
        pygame.font.init()
        self.font = pygame.font.Font(None, 30)
        self.board = self._draw_board()

    def draw(self, snake, apple, direction: int, score: int):
        self.surface.fill(BACKGROUND)
        self.surface.blit(self.board, (MARGIN, HUD_HEIGHT))
        self._draw_hud(score, len(snake))
        if apple is not None:
            self._draw_apple(apple)
        self._draw_snake(list(snake), direction)

        if self.mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        return self.frame()

    def frame(self) -> np.ndarray:
        """The last drawn frame as an (H, W, 3) uint8 array, in either mode."""
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def close(self):
        if self.mode == "human":
            pygame.display.quit()

    # --- drawing -------------------------------------------------------------

    def _draw_board(self) -> pygame.Surface:
        board = pygame.Surface((self.width * self.cell, self.height * self.cell))
        for y in range(self.height):
            for x in range(self.width):
                color = TILE_LIGHT if (x + y) % 2 else TILE_DARK
                board.fill(color, (x * self.cell, y * self.cell, self.cell, self.cell))
        return board

    def _draw_hud(self, score: int, length: int):
        label = self.font.render(f"SCORE  {score}", True, TEXT)
        self.surface.blit(label, (MARGIN, (HUD_HEIGHT - label.get_height()) // 2))
        label = self.font.render(f"LENGTH  {length}", True, SUBTEXT)
        self.surface.blit(label, (self.surface.get_width() - MARGIN - label.get_width(), (HUD_HEIGHT - label.get_height()) // 2))

    def _center(self, cell) -> tuple[float, float]:
        x, y = cell
        return MARGIN + (x + 0.5) * self.cell, HUD_HEIGHT + (y + 0.5) * self.cell

    def _draw_snake(self, snake, direction: int):
        thickness = self.cell * 0.76
        n = len(snake)
        colors = [_lerp(HEAD, TAIL, i / max(n - 1, 1)) for i in range(n)]

        # Bridges between neighbouring segments make the body look continuous.
        for i in range(n - 1, 0, -1):
            (x1, y1), (x2, y2) = self._center(snake[i]), self._center(snake[i - 1])
            left, top = min(x1, x2) - thickness / 2, min(y1, y2) - thickness / 2
            rect = pygame.Rect(left, top, abs(x1 - x2) + thickness, abs(y1 - y2) + thickness)
            pygame.draw.rect(self.surface, colors[i], rect, border_radius=int(thickness / 3))

        head = pygame.Rect(0, 0, self.cell * 0.86, self.cell * 0.86)
        head.center = self._center(snake[0])
        pygame.draw.rect(self.surface, HEAD, head, border_radius=int(self.cell / 3))
        self._draw_eyes(snake[0], direction)

    def _draw_eyes(self, head, direction: int):
        cx, cy = self._center(head)
        dx, dy = DIRECTIONS[direction]
        px, py = -dy, dx  # perpendicular to the heading
        for side in (-1, 1):
            ex = cx + dx * self.cell * 0.15 + px * side * self.cell * 0.2
            ey = cy + dy * self.cell * 0.15 + py * side * self.cell * 0.2
            pygame.draw.circle(self.surface, EYE, (ex, ey), self.cell * 0.12)
            pygame.draw.circle(self.surface, PUPIL, (ex + dx * self.cell * 0.04, ey + dy * self.cell * 0.04), self.cell * 0.06)

    def _draw_apple(self, apple):
        cx, cy = self._center(apple)
        pygame.draw.circle(self.surface, APPLE, (cx, cy + self.cell * 0.04), self.cell * 0.34)
        leaf = pygame.Rect(0, 0, self.cell * 0.22, self.cell * 0.12)
        leaf.center = (cx + self.cell * 0.1, cy - self.cell * 0.32)
        pygame.draw.ellipse(self.surface, LEAF, leaf)
