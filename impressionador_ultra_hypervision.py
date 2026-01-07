#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Impressionador ULTRA • HyperVision 3D (Windows-friendly) 🚀🌀

Um showcase de efeitos no terminal feito para impressionar.

✅ Mantém os clássicos (Essência):
  1) Donut 3D
  2) Matrix Rain
  3) Mandelbrot

✅ Extras + 3D Fake:
  4) Starfield
  5) DOOM Fire
  6) Plasma
  7) Metaballs
  8) Game of Life
  9) Tunnel (Motion Blur)
 10) Wireframe Cube
 11) Terrain (Normal Shading)

✅ NOVO (integrado no projeto):
 12) Run Tunnel (corrida infinita sem saída + ilusões)
 13) Moiré Vortex (op-art)
 14) Spiral Trance (hipnose)

✅ Recursos "de demo":
- AUTO-SHOW (apresentação automática) com transições wipe/glitch
- HUD avançado (FPS, resolução, viewport, tema, cor, velocidade, pausa, qualidade)
- Screenshot (S) e Recorder ANSI (O)
- Qualidade adaptativa em modo Mono para cenas pesadas
- Safe Mode (X) para reduzir intensidade rapidamente

Controles (durante as cenas):
  1..9  -> troca rápida
  J     -> cena 10
  K     -> cena 11
  L     -> cena 12
  M     -> cena 13
  N     -> cena 14
  I     -> alterna variações do efeito atual (quando suportado)
  T     -> tema (Normal/Suave/Neon/Psicodélico/Mono)
  C     -> cor ON/OFF
  +/-   -> velocidade
  P     -> pausa
  H     -> ajuda
  F     -> HUD ON/OFF
  A     -> AUTO-SHOW ON/OFF
  S     -> screenshot (salva .txt em screens/)
  O     -> recorder ON/OFF (salva .ans em recordings/)
  X     -> SAFE MODE (tema Suave + cor ON + velocidade 1.0)
  R     -> reset da cena
  Q/ESC -> menu
  Ctrl+C -> sair

Aviso: cenas 12–14 podem ser bem hipnóticas. Use X (Safe Mode) se necessário.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
import shutil
from datetime import datetime

IS_WINDOWS = (os.name == 'nt')
CSI = "\x1b["

# Enable VT on Windows
if IS_WINDOWS:
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, mode.value | 0x0004)
    except Exception:
        pass


def _w(s: str) -> None:
    sys.stdout.write(s)


def flush() -> None:
    sys.stdout.flush()


def clear() -> None:
    _w(CSI + '2J' + CSI + 'H')


def home() -> None:
    _w(CSI + 'H')


def hide_cursor() -> None:
    _w(CSI + '?25l')


def show_cursor() -> None:
    _w(CSI + '?25h')


def reset() -> str:
    return CSI + '0m'


def bold() -> str:
    return CSI + '1m'


def dim() -> str:
    return CSI + '2m'


def fg256(n: int) -> str:
    return f"{CSI}38;5;{n}m"


def fg_rgb(r: int, g: int, b: int) -> str:
    return f"{CSI}38;2;{r};{g};{b}m"


def term_size() -> tuple[int, int]:
    s = shutil.get_terminal_size((160, 50))
    return s.columns, s.lines


def read_key() -> str:
    if not IS_WINDOWS:
        return ''
    try:
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ('\x00', '\xe0') and msvcrt.kbhit():
                return ch + msvcrt.getwch()
            return ch
    except Exception:
        return ''
    return ''


def viewport(cols: int, lines: int, header: int = 4, margin: int = 1):
    usable_h = max(12, lines - header)
    usable_w = max(30, cols - 1)
    vw = max(30, usable_w - 2 * margin)
    vh = max(12, usable_h - 1)
    left = max(0, (usable_w - vw) // 2)
    top = max(0, (usable_h - vh) // 2)
    return vw, vh, left, top


def pad_lines(lines_list: list[str], left_pad: int) -> list[str]:
    if left_pad <= 0:
        return lines_list
    pad = ' ' * left_pad
    return [pad + ln for ln in lines_list]


def frame_to_screen(header_lines: list[str], content_lines: list[str], top_pad: int) -> str:
    out: list[str] = []
    out.extend(header_lines)
    out.extend([''] * top_pad)
    out.extend(content_lines)
    return '\n'.join(out)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def clampi(x: float, lo: int = 0, hi: int = 255) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return int(x)


def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    h = (h % 1.0) * 6.0
    i = int(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (clampi(r * 255), clampi(g * 255), clampi(b * 255))


def vnorm(v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = v
    l = math.sqrt(x * x + y * y + z * z) + 1e-9
    return (x / l, y / l, z / l)


def vdot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def strip_ansi(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s[i] == '\x1b':
            j = i + 1
            while j < len(s) and s[j] != 'm':
                j += 1
            i = j + 1
        else:
            out.append(s[i])
            i += 1
    return ''.join(out)


def save_screenshot(lines: list[str], folder: str = 'screens') -> str:
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(folder, f'shot_{ts}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([strip_ansi(l) for l in lines]))
    return path


def recorder_open(folder: str = 'recordings'):
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(folder, f'session_{ts}.ans')
    f = open(path, 'w', encoding='utf-8')
    return path, f


THEMES = [
    ("Normal", "cores clássicas e sóbrias"),
    ("Suave", "menos saturação (confortável)"),
    ("Neon", "vibrante"),
    ("Psicodélico", "arco-íris dinâmico"),
    ("Mono", "preto e branco"),
]


class Theme:
    def __init__(self):
        self.idx = 0
        self.color_enabled = True

    @property
    def name(self) -> str:
        return THEMES[self.idx][0]

    @property
    def desc(self) -> str:
        return THEMES[self.idx][1]

    def next_theme(self) -> None:
        self.idx = (self.idx + 1) % len(THEMES)

    def set_theme(self, name: str) -> None:
        for i, (n, _) in enumerate(THEMES):
            if n.lower() == name.lower():
                self.idx = i
                return

    def toggle_color(self) -> None:
        self.color_enabled = not self.color_enabled

    def is_mono(self) -> bool:
        return self.name == 'Mono' or (not self.color_enabled)

    def ui_title(self, s: str) -> str:
        if self.is_mono():
            return bold() + s + reset()
        col = 39
        if self.name == 'Suave':
            col = 81
        elif self.name == 'Neon':
            col = 207
        elif self.name == 'Psicodélico':
            col = 226
        return bold() + fg256(col) + s + reset()

    def ui_subtle(self, s: str) -> str:
        if self.is_mono():
            return dim() + s + reset()
        return dim() + fg256(245) + s + reset()

    def ui_ok(self, s: str) -> str:
        if self.is_mono():
            return s
        return fg256(46) + s + reset()

    def donut(self, lum: float, t: float) -> str:
        if self.is_mono():
            return ''
        lum = clamp(lum)
        if self.name == 'Normal':
            r = int(20 + 35 * lum)
            g = int(120 + 95 * lum)
            b = int(170 + 75 * lum)
        elif self.name == 'Suave':
            r = int(25 + 25 * lum)
            g = int(120 + 65 * lum)
            b = int(155 + 55 * lum)
        elif self.name == 'Neon':
            r = int(80 + 140 * lum)
            g = int(40 + 200 * lum)
            b = int(160 + 80 * lum)
        else:
            r, g, b = hsv_to_rgb((t * 0.12 + lum * 0.35) % 1.0, 0.95, 0.45 + 0.55 * lum)
        return fg_rgb(r, g, b)

    def mandel(self, k: float, t: float) -> str:
        if self.is_mono():
            return ''
        k = clamp(k)
        if self.name == 'Normal':
            if k < 0.6:
                u = k / 0.6
                r = int(0 + 50 * u)
                g = int(70 + 150 * u)
                b = int(170 + 60 * u)
            else:
                u = (k - 0.6) / 0.4
                r = int(50 + 180 * u)
                g = int(220 + 20 * u)
                b = int(230 - 170 * u)
        elif self.name == 'Suave':
            r, g, b = hsv_to_rgb(0.60 + 0.10 * k, 0.45, 0.55 + 0.35 * (k ** 0.5))
        elif self.name == 'Neon':
            r, g, b = hsv_to_rgb(0.75 + 0.55 * k, 0.95, 0.35 + 0.65 * (k ** 0.45))
        else:
            r, g, b = hsv_to_rgb((t * 0.08 + 2.8 * k) % 1.0, 0.98, 0.25 + 0.75 * (k ** 0.35))
        return fg_rgb(r, g, b)

    def matrix(self, intensity: float, t: float, highlight: float = 0.0) -> str:
        if self.is_mono():
            return ''
        intensity = clamp(intensity)
        if self.name in ('Normal', 'Suave'):
            if highlight > 0.7:
                return fg256(120)
            if highlight > 0.35:
                return fg256(82)
            return fg256(46)
        if self.name == 'Neon':
            r, g, b = hsv_to_rgb(0.40 + 0.08 * math.sin(t * 0.8), 0.95, 0.35 + 0.65 * intensity)
            return fg_rgb(r, g, b)
        r, g, b = hsv_to_rgb((t * 0.15 + intensity * 0.9) % 1.0, 0.98, 0.25 + 0.75 * intensity)
        return fg_rgb(r, g, b)

    def fire(self, idx: int, t: float) -> str:
        if self.is_mono():
            return ''
        idx = max(0, min(35, idx))
        if self.name == 'Normal':
            palette = [16, 52, 88, 124, 160, 196, 202, 208, 214, 220, 226]
            return fg256(palette[min(len(palette)-1, idx // 4)])
        if self.name == 'Suave':
            palette = [16, 52, 88, 124, 160, 196, 202, 208, 214]
            return fg256(palette[min(len(palette)-1, idx // 5)])
        if self.name == 'Neon':
            r, g, b = hsv_to_rgb(0.02 + 0.08 * (idx / 35), 0.95, 0.30 + 0.70 * (idx / 35))
            return fg_rgb(r, g, b)
        r, g, b = hsv_to_rgb((t * 0.2 + (idx / 35) * 0.35) % 1.0, 0.98, 0.25 + 0.75 * (idx / 35))
        return fg_rgb(r, g, b)

    def fx(self, k: float, t: float, base_h: float = 0.62) -> str:
        if self.is_mono():
            return ''
        k = clamp(k)
        if self.name == 'Normal':
            r, g, b = hsv_to_rgb(base_h + 0.10 * k, 0.55, 0.18 + 0.82 * k)
        elif self.name == 'Suave':
            r, g, b = hsv_to_rgb(base_h + 0.08 * k, 0.35, 0.25 + 0.75 * k)
        elif self.name == 'Neon':
            r, g, b = hsv_to_rgb(base_h + 0.55 * k, 0.95, 0.18 + 0.82 * k)
        else:
            r, g, b = hsv_to_rgb((t * 0.12 + 1.8 * k) % 1.0, 0.98, 0.12 + 0.88 * k)
        return fg_rgb(r, g, b)


class Scene:
    key = '0'
    name = 'Scene'
    heavy = False
    supports_mode = False

    def reset(self, vw: int, vh: int) -> None:
        pass

    def step(self, vw: int, vh: int, t: float, dt: float, speed: float, theme: Theme, quality: float) -> list[str]:
        raise NotImplementedError


# Helpers for adaptive mono scaling

def upscale_mono(small: list[str], rw: int, rh: int, vw: int, vh: int) -> list[str]:
    out = []
    for y in range(vh):
        sy = int(y / max(1, vh - 1) * (rh - 1))
        row = small[sy]
        parts = []
        for x in range(vw):
            sx = int(x / max(1, vw - 1) * (rw - 1))
            parts.append(row[sx])
        out.append(''.join(parts))
    return out


# 1) Donut
class Donut(Scene):
    key = '1'
    name = 'Donut 3D'

    def __init__(self):
        self.A = 0.0
        self.B = 0.0

    def reset(self, vw, vh):
        self.A = 0.0
        self.B = 0.0

    def step(self, vw, vh, t, dt, speed, theme, quality):
        chars = '.,-~:;=!*#$@'
        zbuffer = [0.0] * (vw * vh)
        output = [' '] * (vw * vh)
        lum = [0.0] * (vw * vh)
        A = self.A
        B = self.B
        cosA, sinA = math.cos(A), math.sin(A)
        cosB, sinB = math.cos(B), math.sin(B)
        R1, R2 = 1.0, 2.0
        K2 = 5.0
        K1 = vw * K2 * 3 / (8 * (R1 + R2))
        theta = 0.0
        while theta < 2 * math.pi:
            costheta = math.cos(theta)
            sintheta = math.sin(theta)
            phi = 0.0
            while phi < 2 * math.pi:
                cosphi = math.cos(phi)
                sinphi = math.sin(phi)
                circlex = R2 + R1 * costheta
                circley = R1 * sintheta
                x = circlex * (cosB * cosphi + sinA * sinB * sinphi) - circley * cosA * sinB
                y = circlex * (sinB * cosphi - sinA * cosB * sinphi) + circley * cosA * cosB
                z = K2 + cosA * circlex * sinphi + circley * sinA
                ooz = 1.0 / z
                xp = int(vw / 2 + K1 * ooz * x)
                yp = int(vh / 2 - K1 * ooz * y)
                L = (cosphi * costheta * sinB
                     - cosA * costheta * sinphi
                     - sinA * sintheta
                     + cosB * (cosA * sintheta - costheta * sinA * sinphi))
                if L > 0 and 0 <= xp < vw and 0 <= yp < vh:
                    idx = xp + vw * yp
                    if ooz > zbuffer[idx]:
                        zbuffer[idx] = ooz
                        li = int(L * (len(chars) - 1))
                        li = max(0, min(li, len(chars) - 1))
                        output[idx] = chars[li]
                        lum[idx] = min(1.0, L)
                phi += 0.07
            theta += 0.03
        lines = []
        for y in range(vh):
            start = y * vw
            row_chars = output[start:start + vw]
            if theme.is_mono():
                lines.append(''.join(row_chars))
            else:
                row = []
                for i, ch in enumerate(row_chars):
                    if ch == ' ':
                        row.append(' ')
                    else:
                        row.append(theme.donut(lum[start + i], t) + ch + reset())
                lines.append(''.join(row))
        self.A += 0.04 * speed
        self.B += 0.02 * speed
        return lines


# 2) Matrix
class Matrix(Scene):
    key = '2'
    name = 'Matrix Rain'

    def __init__(self):
        self.drops = []
        self.speeds = []
        self.last = []
        self.charset = (
            "アイウエオカキクケコサシスセソタチツテトナニヌネノ"
            "ハヒフヘホマミムメモヤユヨラリルレロワヲン"
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

    def reset(self, vw, vh):
        self.drops = [random.randint(-vh, vh) for _ in range(vw)]
        self.speeds = [random.uniform(0.7, 2.8) for _ in range(vw)]
        self.last = [time.time()] * vw

    def step(self, vw, vh, t, dt, speed, theme, quality):
        if len(self.drops) != vw:
            self.reset(vw, vh)
        now = time.time()
        canvas = [list(' ' * vw) for _ in range(vh)]
        head = [0.0] * vw
        for x in range(vw):
            if now - self.last[x] >= (0.02 / (self.speeds[x] * speed)):
                self.last[x] = now
                self.drops[x] += 1
                if self.drops[x] > vh + random.randint(0, vh // 2 + 1):
                    self.drops[x] = random.randint(-vh, 0)
                    self.speeds[x] = random.uniform(0.7, 2.8)
            y = self.drops[x]
            head[x] = clamp(1.0 - abs((vh / 2 - y) / (vh / 2 + 1e-9)))
            if 0 <= y < vh:
                canvas[y][x] = random.choice(self.charset)
            for tr in range(1, 12):
                yy = y - tr
                if 0 <= yy < vh and random.random() < 0.65:
                    canvas[yy][x] = random.choice(self.charset)
        if theme.is_mono():
            return [''.join(canvas[y]) for y in range(vh)]
        lines = []
        for y in range(vh):
            row = canvas[y]
            line = []
            for x, ch in enumerate(row):
                if ch == ' ':
                    line.append(' ')
                else:
                    r = random.random()
                    hl = 0.9 if r < 0.03 else (0.5 if r < 0.08 else 0.0)
                    line.append(theme.matrix(head[x], t, hl) + ch + reset())
            lines.append(''.join(line))
        return lines


# 3) Mandelbrot
class Mandel(Scene):
    key = '3'
    name = 'Mandelbrot'
    heavy = True

    def __init__(self):
        self.phase = 0.0
        self.cx, self.cy = -0.5, 0.0
        self.base_scale = 3.0
        self.ramp = " .:-=+*#%@"

    def reset(self, vw, vh):
        self.phase = 0.0

    def _render_mono(self, vw, vh, speed, q):
        rw = max(40, int(vw * q))
        rh = max(20, int(vh * q))
        self.phase += 0.055 * speed
        zoom = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(self.phase * 0.35))
        scale = (0.22 + (self.base_scale - 0.22) * zoom)
        max_iter = int(70 + (1.0 / max(scale, 0.2)) * 55)
        aspect = (rh / rw) * 1.65
        small = []
        for py in range(rh):
            y0 = self.cy + (py / rh - 0.5) * scale * aspect
            parts = []
            for px in range(rw):
                x0 = self.cx + (px / rw - 0.5) * scale
                x = 0.0
                y = 0.0
                it = 0
                while x * x + y * y <= 4.0 and it < max_iter:
                    x, y = x * x - y * y + x0, 2 * x * y + y0
                    it += 1
                if it == max_iter:
                    parts.append(' ')
                else:
                    k = it / max_iter
                    parts.append(self.ramp[int(k * (len(self.ramp) - 1))])
            small.append(''.join(parts))
        return upscale_mono(small, rw, rh, vw, vh)

    def step(self, vw, vh, t, dt, speed, theme, quality):
        if theme.is_mono() and quality < 0.999:
            return self._render_mono(vw, vh, speed, quality)
        self.phase += 0.055 * speed
        zoom = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(self.phase * 0.35))
        scale = (0.22 + (self.base_scale - 0.22) * zoom)
        max_iter = int(70 + (1.0 / max(scale, 0.2)) * 55)
        aspect = (vh / vw) * 1.65
        mono = theme.is_mono()
        lines = []
        for py in range(vh):
            y0 = self.cy + (py / vh - 0.5) * scale * aspect
            parts = []
            for px in range(vw):
                x0 = self.cx + (px / vw - 0.5) * scale
                x = 0.0
                y = 0.0
                it = 0
                while x * x + y * y <= 4.0 and it < max_iter:
                    x, y = x * x - y * y + x0, 2 * x * y + y0
                    it += 1
                if it == max_iter:
                    parts.append(' ')
                else:
                    k = it / max_iter
                    ch = self.ramp[int(k * (len(self.ramp) - 1))]
                    if mono:
                        parts.append(ch)
                    else:
                        parts.append(theme.mandel(k, t) + ch + reset())
            lines.append(''.join(parts))
        return lines


# 4) Starfield
class Starfield(Scene):
    key = '4'
    name = 'Starfield'

    def __init__(self):
        self.stars = []
        self.seed = 777

    def reset(self, vw, vh):
        random.seed(self.seed)
        n = max(500, int(vw * vh * 0.10))
        self.stars = [[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.05, 1.0)] for _ in range(n)]

    def step(self, vw, vh, t, dt, speed, theme, quality):
        if not self.stars:
            self.reset(vw, vh)
        canvas = [[(' ', '') for _ in range(vw)] for _ in range(vh)]
        cx, cy = vw / 2, vh / 2
        sp = (1.0 + 1.6 * (0.5 + 0.5 * math.sin(t * 0.6))) * speed
        for s in self.stars:
            s[2] -= dt * 0.7 * sp
            if s[2] <= 0.03:
                s[0] = random.uniform(-1, 1)
                s[1] = random.uniform(-1, 1)
                s[2] = 1.0
            x, y, z = s
            px = int(cx + (x / z) * (vw * 0.38))
            py = int(cy + (y / z) * (vh * 0.38))
            if 0 <= px < vw and 0 <= py < vh:
                bright = clamp((1.0 - z) ** 0.3)
                ch = '*' if bright > 0.8 else ('.' if bright > 0.45 else '·')
                if theme.is_mono():
                    canvas[py][px] = (ch, '')
                else:
                    canvas[py][px] = (ch, theme.fx(bright, t, base_h=0.62))
        lines = []
        for y in range(vh):
            row = canvas[y]
            if theme.is_mono():
                lines.append(''.join(ch for ch, _ in row))
            else:
                parts = []
                last = None
                for ch, col in row:
                    if col != last:
                        parts.append(col)
                        last = col
                    parts.append(ch)
                parts.append(reset())
                lines.append(''.join(parts))
        return lines


# 5) Fire
class DoomFire(Scene):
    key = '5'
    name = 'DOOM Fire'

    def __init__(self):
        self.heat = []
        self.w = 0
        self.h = 0

    def reset(self, vw, vh):
        self.w, self.h = vw, vh
        self.heat = [[0 for _ in range(vw)] for _ in range(vh)]
        for x in range(vw):
            self.heat[vh - 1][x] = 35

    def step(self, vw, vh, t, dt, speed, theme, quality):
        if self.w != vw or self.h != vh or not self.heat:
            self.reset(vw, vh)
        for x in range(vw):
            self.heat[vh - 1][x] = 30 + int(5 * random.random())
        for y in range(vh - 2, -1, -1):
            row = self.heat[y]
            below = self.heat[y + 1]
            for x in range(vw):
                src = below[x]
                dec = int(random.random() * 3)
                wind = int(random.random() * 3) - 1
                nx = max(0, min(vw - 1, x + wind))
                val = src - dec
                if val < 0:
                    val = 0
                row[nx] = val
        chars = " .,:;irsXA253hMHGS#9B&@"
        mono = theme.is_mono()
        lines = []
        for y in range(vh):
            parts = []
            for x in range(vw):
                hval = self.heat[y][x]
                ch = chars[min(len(chars) - 1, int(hval / 35 * (len(chars) - 1)))]
                if mono:
                    parts.append(ch)
                else:
                    parts.append(theme.fire(hval, t) + ch + reset())
            lines.append(''.join(parts))
        return lines


# 6) Plasma
class Plasma(Scene):
    key = '6'
    name = 'Plasma'

    def step(self, vw, vh, t, dt, speed, theme, quality):
        mono = theme.is_mono()
        ramp = " .:-=+*#%@"
        tt = t * 0.7 * speed
        lines = []
        for y in range(vh):
            ny = (y / vh - 0.5) * 2.0
            parts = []
            for x in range(vw):
                nx = (x / vw - 0.5) * 2.0
                v = 0.0
                v += math.sin((nx * 3.1 + tt) * 1.1)
                v += math.sin((ny * 4.0 - tt * 1.2) * 1.2)
                v += math.sin((nx * 2.0 + ny * 2.0 + tt) * 1.0)
                v += math.sin(math.sqrt(nx * nx + ny * ny) * 6.0 - tt * 1.7)
                v = (v / 4.0 + 1.0) / 2.0
                ch = ramp[int(v * (len(ramp) - 1))]
                if mono:
                    parts.append(ch)
                else:
                    parts.append(theme.fx(v, t, base_h=0.70) + ch + reset())
            lines.append(''.join(parts))
        return lines


# 7) Metaballs
class Metaballs(Scene):
    key = '7'
    name = 'Metaballs'
    heavy = True

    def __init__(self):
        self.balls = []

    def reset(self, vw, vh):
        self.balls = []
        n = 5
        for _ in range(n):
            self.balls.append([
                random.random() * vw,
                random.random() * vh,
                (random.random() * 0.8 + 0.4) * (min(vw, vh) * 0.12),
                random.uniform(-1.2, 1.2),
                random.uniform(-0.9, 0.9),
            ])

    def _render_mono(self, vw, vh, speed, q):
        rw = max(40, int(vw * q))
        rh = max(20, int(vh * q))
        if not self.balls:
            self.reset(rw, rh)
        for b in self.balls:
            b[0] += b[3] * speed
            b[1] += b[4] * speed
            if b[0] < 0 or b[0] > rw - 1:
                b[3] *= -1
                b[0] = max(0, min(rw - 1, b[0]))
            if b[1] < 0 or b[1] > rh - 1:
                b[4] *= -1
                b[1] = max(0, min(rh - 1, b[1]))
        ramp = " .,:;irsXA253hMHGS#9B&@"
        small = []
        for y in range(rh):
            parts = []
            for x in range(rw):
                v = 0.0
                for bx, by, r, _, _ in self.balls:
                    dx = x - bx
                    dy = y - by
                    d2 = dx * dx + dy * dy + 1e-6
                    v += (r * r) / d2
                v = v / (len(self.balls) * 1.2)
                v = clamp(v / 2.2)
                parts.append(ramp[int(v * (len(ramp) - 1))])
            small.append(''.join(parts))
        return upscale_mono(small, rw, rh, vw, vh)

    def step(self, vw, vh, t, dt, speed, theme, quality):
        if theme.is_mono() and quality < 0.999:
            return self._render_mono(vw, vh, speed, quality)
        if not self.balls:
            self.reset(vw, vh)
        for b in self.balls:
            b[0] += b[3] * speed
            b[1] += b[4] * speed
            if b[0] < 0 or b[0] > vw - 1:
                b[3] *= -1
                b[0] = max(0, min(vw - 1, b[0]))
            if b[1] < 0 or b[1] > vh - 1:
                b[4] *= -1
                b[1] = max(0, min(vh - 1, b[1]))
        ramp = " .,:;irsXA253hMHGS#9B&@"
        mono = theme.is_mono()
        lines = []
        for y in range(vh):
            parts = []
            for x in range(vw):
                v = 0.0
                for bx, by, r, _, _ in self.balls:
                    dx = x - bx
                    dy = y - by
                    d2 = dx * dx + dy * dy + 1e-6
                    v += (r * r) / d2
                v = v / (len(self.balls) * 1.2)
                v = clamp(v / 2.2)
                ch = ramp[int(v * (len(ramp) - 1))]
                if mono:
                    parts.append(ch)
                else:
                    parts.append(theme.fx(v, t, base_h=0.85) + ch + reset())
            lines.append(''.join(parts))
        return lines


# 8) Life
class Life(Scene):
    key = '8'
    name = 'Game of Life'

    def __init__(self):
        self.grid = []
        self.age = []
        self.w = 0
        self.h = 0

    def reset(self, vw, vh):
        self.w, self.h = vw, vh
        self.grid = [[1 if random.random() < 0.16 else 0 for _ in range(vw)] for _ in range(vh)]
        self.age = [[0 for _ in range(vw)] for _ in range(vh)]

    def step(self, vw, vh, t, dt, speed, theme, quality):
        if self.w != vw or self.h != vh or not self.grid:
            self.reset(vw, vh)
        steps = 1
        if speed > 1.4:
            steps = 2
        if speed > 2.2:
            steps = 3
        for _ in range(steps):
            ng = [[0] * vw for _ in range(vh)]
            nage = [[0] * vw for _ in range(vh)]
            for y in range(vh):
                ym1 = (y - 1) % vh
                yp1 = (y + 1) % vh
                row = self.grid[y]
                for x in range(vw):
                    xm1 = (x - 1) % vw
                    xp1 = (x + 1) % vw
                    n = (
                        self.grid[ym1][xm1] + self.grid[ym1][x] + self.grid[ym1][xp1]
                        + row[xm1] + row[xp1]
                        + self.grid[yp1][xm1] + self.grid[yp1][x] + self.grid[yp1][xp1]
                    )
                    alive = row[x]
                    if alive:
                        if n == 2 or n == 3:
                            ng[y][x] = 1
                            nage[y][x] = min(255, self.age[y][x] + 1)
                        else:
                            ng[y][x] = 0
                            nage[y][x] = 0
                    else:
                        if n == 3:
                            ng[y][x] = 1
                            nage[y][x] = 1
                        else:
                            ng[y][x] = 0
                            nage[y][x] = 0
            self.grid, self.age = ng, nage
        mono = theme.is_mono()
        lines = []
        for y in range(vh):
            parts = []
            for x in range(vw):
                if self.grid[y][x]:
                    a = self.age[y][x] / 255.0
                    ch = '█' if a > 0.25 else '▓'
                    if mono:
                        parts.append(ch)
                    else:
                        parts.append(theme.fx(a, t, base_h=0.33) + ch + reset())
                else:
                    parts.append(' ')
            lines.append(''.join(parts))
        return lines


# 9) Tunnel
class Tunnel(Scene):
    key = '9'
    name = 'Tunnel (Motion Blur)'

    def __init__(self):
        self.phase = 0.0
        self.prev = []
        self.w = 0
        self.h = 0

    def reset(self, vw, vh):
        self.phase = 0.0
        self.w, self.h = vw, vh
        self.prev = [[0.0 for _ in range(vw)] for _ in range(vh)]

    def step(self, vw, vh, t, dt, speed, theme, quality):
        if self.w != vw or self.h != vh or not self.prev:
            self.reset(vw, vh)
        self.phase += dt * 1.0 * speed
        decay = clamp(0.82 + 0.06 * min(1.0, (speed - 1.0) / 2.0), 0.78, 0.92)
        ramp = " .:-=+*#%@"
        mono = theme.is_mono()
        cx = (vw - 1) / 2
        cy = (vh - 1) / 2
        aspect = vw / max(1, vh)
        lines = []
        for y in range(vh):
            parts = []
            dy = (y - cy) / (vh / 2)
            prow = self.prev[y]
            for x in range(vw):
                dx = (x - cx) / (vw / 2)
                dx *= aspect
                r = math.sqrt(dx * dx + dy * dy) + 1e-6
                ang = math.atan2(dy, dx)
                z = 1.0 / r
                u = (ang / (2 * math.pi) + 0.5) * 3.0
                v = z * 0.18 + self.phase
                p = 0.5 + 0.5 * math.sin((u + v * 4.0) * 2 * math.pi)
                q = 0.5 + 0.5 * math.cos((v * 2.2 - u * 0.3) * 2 * math.pi)
                cur = clamp((p * 0.65 + q * 0.35) * (1.0 - clamp(r * 0.65)))
                val = max(cur, prow[x] * decay)
                prow[x] = val
                ch = ramp[int(val * (len(ramp) - 1))]
                if mono:
                    parts.append(ch)
                else:
                    kk = clamp((z / 6.0))
                    intensity = clamp(0.15 + 0.85 * max(val, kk * 0.9))
                    parts.append(theme.fx(intensity, t, base_h=0.62) + ch + reset())
            lines.append(''.join(parts))
        return lines


# 10) Cube
class Cube(Scene):
    key = '10'
    name = 'Wireframe Cube'

    def __init__(self):
        self.ang = 0.0

    def reset(self, vw, vh):
        self.ang = 0.0

    @staticmethod
    def _rot(v, ax, ay, az):
        x, y, z = v
        cx, sx = math.cos(ax), math.sin(ax)
        y, z = y * cx - z * sx, y * sx + z * cx
        cy, sy = math.cos(ay), math.sin(ay)
        x, z = x * cy + z * sy, -x * sy + z * cy
        cz, sz = math.cos(az), math.sin(az)
        x, y = x * cz - y * sz, x * sz + y * cz
        return (x, y, z)

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        pts = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            pts.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return pts

    def step(self, vw, vh, t, dt, speed, theme, quality):
        self.ang += dt * 1.2 * speed
        ax = self.ang * 0.9
        ay = self.ang * 1.1
        az = self.ang * 0.7
        s = 1.0
        verts = [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
                 (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        proj = []
        depth = []
        for v in verts:
            x, y, z = self._rot(v, ax, ay, az)
            z += 3.2
            f = 1.8 / z
            px = x * f
            py = y * f
            sxp = int(px * (vw * 0.35) + vw / 2)
            syp = int(-py * (vh * 0.55) + vh / 2)
            proj.append((sxp, syp))
            depth.append(clamp(1.0 - (z - 2.2) / 3.5))
        grid = [[' ' for _ in range(vw)] for _ in range(vh)]
        zbuf = [[-1e9 for _ in range(vw)] for _ in range(vh)]
        for a, b in edges:
            x0, y0 = proj[a]
            x1, y1 = proj[b]
            for i, (x, y) in enumerate(self._bresenham(x0, y0, x1, y1)):
                if 0 <= x < vw and 0 <= y < vh:
                    tt = i / max(1, len(self._bresenham(x0,y0,x1,y1)) - 1)
                    d = depth[a] * (1 - tt) + depth[b] * tt
                    if d > zbuf[y][x]:
                        zbuf[y][x] = d
                        grid[y][x] = '#' if d > 0.75 else ('*' if d > 0.45 else '.')
        for i, (x, y) in enumerate(proj):
            if 0 <= x < vw and 0 <= y < vh:
                grid[y][x] = '@'
                zbuf[y][x] = max(zbuf[y][x], depth[i])
        mono = theme.is_mono()
        lines = []
        for y in range(vh):
            if mono:
                lines.append(''.join(grid[y]))
            else:
                parts = []
                for x in range(vw):
                    ch = grid[y][x]
                    if ch == ' ':
                        parts.append(' ')
                    else:
                        parts.append(theme.fx(clamp(zbuf[y][x]), t, base_h=0.78) + ch + reset())
                lines.append(''.join(parts))
        return lines


# 11) Terrain
class Terrain(Scene):
    key = '11'
    name = 'Terrain (Normal Shading)'
    heavy = True

    def __init__(self):
        self.off = 0.0

    def reset(self, vw, vh):
        self.off = 0.0

    def height(self, x, z, t):
        return (
            math.sin(x * 0.45 + t * 0.9) * 0.55
            + math.cos(z * 0.28 + t * 0.7) * 0.65
            + math.sin((x + z) * 0.22 - t * 0.6) * 0.45
        )

    def normal_at(self, x, z, t):
        eps = 0.06
        hL = self.height(x - eps, z, t)
        hR = self.height(x + eps, z, t)
        hD = self.height(x, z - eps, t)
        hU = self.height(x, z + eps, t)
        dx = (hR - hL) / (2 * eps)
        dz = (hU - hD) / (2 * eps)
        return vnorm((-dx, 1.0, -dz))

    def step(self, vw, vh, t, dt, speed, theme, quality):
        self.off += dt * 1.2 * speed
        mono = theme.is_mono()
        grid = [[' ' for _ in range(vw)] for _ in range(vh)]
        ibuf = [[0.0 for _ in range(vw)] for _ in range(vh)]
        zbuf = [[-1e9 for _ in range(vw)] for _ in range(vh)]
        horizon = int(vh * 0.30)
        scale_y = vh * 0.18
        light = vnorm((0.35, 0.85, -0.45))
        ambient = 0.22
        for zi in range(1, 115):
            z = zi * 0.15 + self.off
            depth = zi / 115.0
            inv = 1.0 / (0.55 + depth * 2.7)
            k_depth = clamp(1.0 - depth)
            for xi in range(vw):
                x = (xi - vw / 2) * inv * 0.12
                hx = x * 10.0
                hz = z * 10.0
                h = self.height(hx, hz, t)
                sy = int(horizon + (h * scale_y) * inv)
                sy = max(0, min(vh - 1, sy))
                n = self.normal_at(hx, hz, t)
                diff = max(0.0, vdot(n, light))
                shade = ambient + (1.0 - ambient) * diff
                inten = clamp(shade * (0.30 + 0.70 * k_depth))
                if inten > 0.80:
                    ch = '█'
                elif inten > 0.60:
                    ch = '▓'
                elif inten > 0.40:
                    ch = '▒'
                else:
                    ch = '·'
                for y in range(sy, vh):
                    if k_depth > zbuf[y][xi]:
                        zbuf[y][xi] = k_depth
                        grid[y][xi] = ch
                        ibuf[y][xi] = inten
        sky_h = int(vh * 0.30)
        for y in range(0, sky_h):
            for x in range(vw):
                if grid[y][x] == ' ' and random.random() < 0.0012:
                    grid[y][x] = '·'
                    ibuf[y][x] = 0.18
        if mono:
            return [''.join(grid[y]) for y in range(vh)]
        lines = []
        for y in range(vh):
            parts = []
            for x in range(vw):
                ch = grid[y][x]
                if ch == ' ':
                    parts.append(' ')
                else:
                    base_h = 0.33 if y >= sky_h else 0.62
                    parts.append(theme.fx(ibuf[y][x], t, base_h=base_h) + ch + reset())
            lines.append(''.join(parts))
        return lines


# 12) Run Tunnel
class RunTunnel(Scene):
    key = '12'
    name = 'Run Tunnel (Infinite Run)'
    supports_mode = True

    def __init__(self):
        self.phase = 0.0
        self.prev = []
        self.w = 0
        self.h = 0
        self.mode = 0

    def reset(self, vw, vh):
        self.phase = 0.0
        self.w, self.h = vw, vh
        self.prev = [[0.0 for _ in range(vw)] for _ in range(vh)]
        self.mode = 0

    def step(self, vw, vh, t, dt, speed, theme, quality):
        if self.w != vw or self.h != vh or not self.prev:
            self.reset(vw, vh)
        run = t * (1.35 * speed)
        bob = 0.06 * math.sin(run * 2.8) + 0.025 * math.sin(run * 6.2)
        sway = 0.10 * math.sin(run * 1.3)
        roll = 0.35 * math.sin(run * 0.9)
        decay = clamp(0.84 + 0.06 * min(1.0, (speed - 1.0) / 2.0), 0.80, 0.93)
        ramp = " .:-=+*#%@"
        mono = theme.is_mono()
        cx = (vw - 1) / 2 + sway * (vw * 0.12)
        cy = (vh - 1) / 2 + bob * (vh * 0.22)
        aspect = vw / max(1, vh)
        self.phase += dt * (1.2 * speed)
        cr = math.cos(roll)
        sr = math.sin(roll)
        lines = []
        for y in range(vh):
            parts = []
            prow = self.prev[y]
            dy0 = (y - cy) / (vh / 2)
            for x in range(vw):
                dx0 = (x - cx) / (vw / 2)
                dx0 *= aspect
                dx = dx0 * cr - dy0 * sr
                dy = dx0 * sr + dy0 * cr
                r = math.sqrt(dx * dx + dy * dy) + 1e-6
                ang = math.atan2(dy, dx)
                zc = min(12.0, 1.0 / r)
                u = (ang / (2 * math.pi) + 0.5)
                v = self.phase + zc * 0.22
                rings = 0.5 + 0.5 * math.sin((v * 6.5) + 0.7 * math.sin(v * 1.3))
                ribs  = 0.5 + 0.5 * math.sin((u * 20.0 + v * 2.2) * 2 * math.pi)
                warp  = 0.5 + 0.5 * math.sin((u * 8.0 - v * 3.0) * 2 * math.pi)
                if self.mode == 0:
                    cur = rings * 0.55 + ribs * 0.45
                elif self.mode == 1:
                    chk = 0.5 + 0.5 * math.sin((u * 16.0 + v * 4.6) * 2 * math.pi)
                    cur = chk * 0.55 + warp * 0.45
                else:
                    vort = 0.5 + 0.5 * math.sin((u * 28.0 + v * 8.0 + 3.5 * math.sin(ang * 3.0)) * 2 * math.pi)
                    cur = vort * 0.6 + rings * 0.4
                cur *= (1.0 - clamp(r * 0.75))
                cur = clamp(cur)
                val = max(cur, prow[x] * decay)
                prow[x] = val
                ch = ramp[int(val * (len(ramp) - 1))]
                if mono:
                    parts.append(ch)
                else:
                    depth_boost = clamp(zc / 10.0)
                    inten = clamp(0.18 + 0.82 * max(val, depth_boost))
                    base_h = 0.62 if self.mode == 0 else (0.78 if self.mode == 1 else 0.90)
                    parts.append(theme.fx(inten, t, base_h=base_h) + ch + reset())
            lines.append(''.join(parts))
        return lines


# 13) Moiré
class MoireVortex(Scene):
    key = '13'
    name = 'Moiré Vortex (Op-Art)'
    supports_mode = True

    def __init__(self):
        self.ph = 0.0
        self.mode = 0

    def reset(self, vw, vh):
        self.ph = 0.0
        self.mode = 0

    def step(self, vw, vh, t, dt, speed, theme, quality):
        self.ph += dt * 0.8 * speed
        mono = theme.is_mono()
        ramp = " .:-=+*#%@"
        cx = (vw - 1) / 2
        cy = (vh - 1) / 2
        aspect = vw / max(1, vh)
        a1 = self.ph * 0.9
        a2 = -self.ph * 1.15
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        if self.mode == 0:
            k1, k2 = 10.0, 10.8
        elif self.mode == 1:
            k1, k2 = 14.0, 14.7
        else:
            k1, k2 = 8.5, 9.25
        lines = []
        for y in range(vh):
            parts = []
            ny = (y - cy) / (vh / 2)
            for x in range(vw):
                nx = (x - cx) / (vw / 2)
                nx *= aspect
                r = math.sqrt(nx * nx + ny * ny) + 1e-6
                ang = math.atan2(ny, nx)
                warp = 0.12 * math.sin(ang * 6.0 + self.ph * 3.0) + 0.08 * math.cos(r * 8.0 - self.ph * 2.0)
                wx = nx + nx * warp
                wy = ny + ny * warp
                g1 = math.sin((wx * c1 + wy * s1) * k1 * math.pi + self.ph * 3.0)
                g2 = math.sin((wx * c2 + wy * s2) * k2 * math.pi - self.ph * 2.2)
                cur = 0.5 + 0.5 * (g1 * g2)
                cur *= (1.0 - clamp(r * 0.55))
                cur = clamp(cur)
                ch = ramp[int(cur * (len(ramp) - 1))]
                if mono:
                    parts.append(ch)
                else:
                    base_h = 0.55 if self.mode == 0 else (0.15 if self.mode == 1 else 0.82)
                    parts.append(theme.fx(cur, t, base_h=base_h) + ch + reset())
            lines.append(''.join(parts))
        return lines


# 14) Spiral
class SpiralTrance(Scene):
    key = '14'
    name = 'Spiral Trance (Hipnótico)'
    supports_mode = True

    def __init__(self):
        self.ph = 0.0
        self.mode = 0

    def reset(self, vw, vh):
        self.ph = 0.0
        self.mode = 0

    def step(self, vw, vh, t, dt, speed, theme, quality):
        self.ph += dt * 1.1 * speed
        mono = theme.is_mono()
        ramp = " .:-=+*#%@"
        cx = (vw - 1) / 2
        cy = (vh - 1) / 2
        aspect = vw / max(1, vh)
        rot = self.ph * 0.9
        lines = []
        for y in range(vh):
            parts = []
            ny = (y - cy) / (vh / 2)
            for x in range(vw):
                nx = (x - cx) / (vw / 2)
                nx *= aspect
                r = math.sqrt(nx * nx + ny * ny) + 1e-6
                ang = math.atan2(ny, nx) + rot
                spir = ang * 5.5 + math.log(r + 1e-6) * 6.0
                wv = 0.65 + 0.35 * math.sin(self.ph * 2.2 + r * 10.0)
                if self.mode == 0:
                    cur = 0.5 + 0.5 * math.sin(spir - self.ph * 3.2)
                elif self.mode == 1:
                    cur = 0.5 + 0.5 * math.sin(spir + math.sin(ang * 3.0) * 2.0 + self.ph * 2.5)
                else:
                    cur = 0.5 + 0.5 * math.sin(spir * 1.25 + self.ph * 4.0 + 2.0 * math.sin(r * 12.0 - ang * 2.0))
                cur *= wv
                cur *= (1.0 - clamp(r * 0.72))
                cur = clamp(cur)
                ch = ramp[int(cur * (len(ramp) - 1))]
                if mono:
                    parts.append(ch)
                else:
                    base_h = 0.0 if self.mode == 2 else (0.62 if self.mode == 0 else 0.33)
                    parts.append(theme.fx(cur, t, base_h=base_h) + ch + reset())
            lines.append(''.join(parts))
        return lines


# transitions and help

def transition_wipe(prev_lines: list[str], next_lines: list[str], steps: int = 18) -> list[list[str]]:
    ph = len(prev_lines)
    nh = len(next_lines)
    h = min(ph, nh)
    prev = [strip_ansi(ln) for ln in prev_lines[:h]]
    nex = [strip_ansi(ln) for ln in next_lines[:h]]
    w = min(len(prev[0]) if prev else 0, len(nex[0]) if nex else 0)
    frames = []
    for s in range(steps + 1):
        cut = int(w * (s / steps))
        frame = []
        for y in range(h):
            frame.append(nex[y][:cut] + prev[y][cut:])
        frames.append(frame)
    return frames


def glitch_frame(lines: list[str], intensity: float = 0.12) -> list[str]:
    out = []
    for ln in lines:
        s = list(strip_ansi(ln))
        if random.random() < intensity and len(s) > 6:
            a = random.randint(0, len(s) - 4)
            b = min(len(s), a + random.randint(2, 10))
            seg = s[a:b]
            shift = random.randint(-8, 8)
            del s[a:b]
            ins = max(0, min(len(s), a + shift))
            s[ins:ins] = seg
        if random.random() < intensity and len(s) > 0:
            for _ in range(random.randint(1, 6)):
                i = random.randint(0, len(s) - 1)
                s[i] = random.choice(['#', '%', '@', '.', '*', ' '])
        out.append(''.join(s))
    return out


HELP_TEXT = [
    "Troca rápida:",
    "  1..9  | J=10 | K=11 | L=12 | M=13 | N=14",
    "",
    "Controles:",
    "  T tema | C cor | I modo | +/- vel | P pausa | R reset",
    "  A autoshow | F HUD | S screenshot | O recorder | X safe mode",
    "  H ajuda | Q/ESC menu | Ctrl+C sair",
    "",
    "Dica: Se estiver intenso, aperte X (Safe Mode).",
]


def overlay_help(content: list[str], vw: int, vh: int, theme: Theme) -> list[str]:
    box_w = min(vw - 4, max(52, max(len(s) for s in HELP_TEXT) + 4))
    box_h = min(vh - 4, len(HELP_TEXT) + 4)
    x0 = (vw - box_w) // 2
    y0 = (vh - box_h) // 2
    grid = [list(strip_ansi(line).ljust(vw)[:vw]) for line in content]
    tl, tr, bl, br = '┌', '┐', '└', '┘'
    hz, vt = '─', '│'

    def paint(x, y, s):
        if 0 <= y < vh:
            for i, ch in enumerate(s):
                xx = x + i
                if 0 <= xx < vw:
                    grid[y][xx] = ch

    paint(x0, y0, tl + hz * (box_w - 2) + tr)
    for yy in range(1, box_h - 1):
        paint(x0, y0 + yy, vt + ' ' * (box_w - 2) + vt)
    paint(x0, y0 + box_h - 1, bl + hz * (box_w - 2) + br)
    title = " AJUDA "
    paint(x0 + (box_w - len(title)) // 2, y0, title)
    for i, line in enumerate(HELP_TEXT[:box_h - 4]):
        paint(x0 + 2, y0 + 2 + i, line[:box_w - 4].ljust(box_w - 4))
    out = [''.join(r) for r in grid]
    if theme.is_mono():
        return out
    col = fg256(214) if theme.name == 'Psicodélico' else (fg256(81) if theme.name in ('Normal', 'Suave') else fg256(207))
    return [col + ln + reset() for ln in out]


def make_header(theme: Theme, scene_label: str, hud: bool,
                cols: int, lines: int, vw: int, vh: int,
                fps: float, speed: float, paused: bool,
                autoshow: bool, rec_on: bool, quality: float,
                hint: str, progress: float | None,
                mode_label: str) -> list[str]:
    title = theme.ui_title(f"Impressionador ULTRA • HyperVision 3D  |  {scene_label}{mode_label}")
    hint_line = theme.ui_subtle(
        f"Teclas: {hint} | T tema | C cor | I modo | +/- vel | P pausa | A autoshow | H ajuda | X safe | Q/ESC"
    )
    if not hud:
        return [title, hint_line, '']
    prog_txt = ""
    if progress is not None:
        bar_w = 22
        fill = int(bar_w * clamp(progress))
        prog_txt = f" | Auto {fill*'█'}{(bar_w-fill)*'·'} {int(progress*100):02d}%"
    indicators = (
        f"{theme.ui_ok('FPS')} {fps:5.1f} | {theme.ui_ok('Tela')} {cols}x{lines} | {theme.ui_ok('Viewport')} {vw}x{vh} | "
        f"{theme.ui_ok('Tema')} {theme.name} | {theme.ui_ok('Cor')} {'ON' if (theme.color_enabled and theme.name!='Mono') else 'OFF'} | "
        f"{theme.ui_ok('Vel')} {speed:0.1f} | {theme.ui_ok('Pausa')} {'SIM' if paused else 'NÃO'} | "
        f"{theme.ui_ok('Auto')} {'ON' if autoshow else 'OFF'} | {theme.ui_ok('Rec')} {'ON' if rec_on else 'OFF'} | "
        f"{theme.ui_ok('Q')} {quality:.2f}{prog_txt}"
    )
    return [title, hint_line, indicators, '']


def compute_quality(auto_quality: bool, theme: Theme, scene: Scene, vw: int, vh: int, fps: float) -> float:
    if not auto_quality:
        return 1.0
    if not theme.is_mono():
        return 1.0
    if not getattr(scene, 'heavy', False):
        return 1.0
    area = vw * vh
    q = 1.0
    if area > 120 * 40:
        q = 0.75
    if area > 180 * 55 or fps < 22:
        q = 0.60
    if area > 220 * 65 or fps < 16:
        q = 0.50
    return q


def autoshow_order() -> list[int]:
    return [0, 1, 2, 8, 9, 10, 11, 12, 13, 3, 4, 5, 6, 7]


def menu(theme: Theme) -> None:
    clear(); show_cursor()
    cols, lines = term_size()
    title = 'IMPRESSIONADOR ULTRA • HyperVision 3D'
    subtitle = 'Essência + 3D fake + ilusões hipnóticas  |  A = AUTO-SHOW'
    status = f"Tema: {theme.name} ({theme.desc}) | Cor: {'ON' if theme.color_enabled and theme.name!='Mono' else 'OFF'}"
    items = [
        f"{bold()}{title}{reset()}",
        subtitle,
        '',
        '1) Donut 3D',
        '2) Matrix Rain',
        '3) Mandelbrot',
        '4) Starfield',
        '5) DOOM Fire',
        '6) Plasma',
        '7) Metaballs',
        '8) Game of Life',
        '9) Tunnel (Motion Blur)',
        '10) Wireframe Cube',
        '11) Terrain (Normal Shading)',
        '12) Run Tunnel (Infinite Run)  🌀',
        '13) Moiré Vortex (Op-Art)     🌀',
        '14) Spiral Trance (Hipnótico) 🌀',
        '',
        'A) AUTO-SHOW (apresentação automática com transições)',
        'T) Trocar tema | C) Cor ON/OFF | X) Safe Mode | H) Ajuda',
        '0) Sair',
        '',
        status,
        '',
        'Dica: maximize o terminal (mais colunas = mais detalhe).',
    ]
    w = max(len(strip_ansi(l)) for l in items)
    left = max(0, (cols - w) // 2)
    top = max(0, (lines - len(items)) // 2)
    _w('\n' * top)
    for l in items:
        _w(' ' * left + l + '\n')
    flush()


SCENES: list[Scene] = [
    Donut(), Matrix(), Mandel(), Starfield(), DoomFire(), Plasma(), Metaballs(), Life(),
    Tunnel(), Cube(), Terrain(), RunTunnel(), MoireVortex(), SpiralTrance()
]
KEY_TO_SCENE = {s.key: i for i, s in enumerate(SCENES)}


def run_scene(scene_idx: int, theme: Theme, autoshow: bool = False) -> int:
    clear(); hide_cursor()
    paused = False
    show_help = False
    hud = True
    speed = 1.0
    auto_quality = True
    auto_list = autoshow_order()
    auto_i = auto_list.index(scene_idx) if scene_idx in auto_list else 0
    scene_start = time.perf_counter()
    auto_period = 12.0
    rec_on = False
    rec_file = None
    cols, lines = term_size()
    vw, vh, left, top = viewport(cols, lines)
    SCENES[scene_idx].reset(vw, vh)
    last_t = time.perf_counter()
    fps_t0 = last_t
    frames = 0
    fps = 0.0
    last_content: list[str] | None = None

    try:
        while True:
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            c2, l2 = term_size()
            if c2 != cols or l2 != lines:
                cols, lines = c2, l2
                vw, vh, left, top = viewport(cols, lines)
                clear()
                SCENES[scene_idx].reset(vw, vh)
                last_content = None

            k = read_key()
            if k:
                kk = k.lower()
                if kk in KEY_TO_SCENE and kk not in ('10','11','12','13','14'):
                    scene_idx = KEY_TO_SCENE[kk]; SCENES[scene_idx].reset(vw, vh)
                    scene_start = now; last_content=None
                elif kk == 'j':
                    scene_idx = KEY_TO_SCENE['10']; SCENES[scene_idx].reset(vw, vh)
                    scene_start = now; last_content=None
                elif kk == 'k':
                    scene_idx = KEY_TO_SCENE['11']; SCENES[scene_idx].reset(vw, vh)
                    scene_start = now; last_content=None
                elif kk == 'l':
                    scene_idx = KEY_TO_SCENE['12']; SCENES[scene_idx].reset(vw, vh)
                    scene_start = now; last_content=None
                elif kk == 'm':
                    scene_idx = KEY_TO_SCENE['13']; SCENES[scene_idx].reset(vw, vh)
                    scene_start = now; last_content=None
                elif kk == 'n':
                    scene_idx = KEY_TO_SCENE['14']; SCENES[scene_idx].reset(vw, vh)
                    scene_start = now; last_content=None
                elif kk == 't':
                    theme.next_theme(); clear(); last_content=None
                elif kk == 'c':
                    theme.toggle_color(); clear(); last_content=None
                elif kk in ('+','='):
                    speed = min(3.0, speed + 0.1)
                elif kk in ('-','_'):
                    speed = max(0.2, speed - 0.1)
                elif kk == 'p':
                    paused = not paused
                elif kk == 'h':
                    show_help = not show_help
                elif kk == 'f':
                    hud = not hud
                elif kk == 'a':
                    autoshow = not autoshow
                    scene_start = now
                elif kk == 'x':
                    theme.set_theme('Suave'); theme.color_enabled = True
                    speed = 1.0
                    clear(); last_content=None
                elif kk == 'i':
                    sc = SCENES[scene_idx]
                    if getattr(sc, 'supports_mode', False) and hasattr(sc, 'mode'):
                        try:
                            sc.mode = (int(sc.mode) + 1) % 3
                        except Exception:
                            pass
                elif kk == 's':
                    if last_content:
                        save_screenshot(last_content)
                elif kk == 'o':
                    if not rec_on:
                        _, rec_file = recorder_open(); rec_on = True
                    else:
                        rec_on = False
                        try:
                            if rec_file:
                                rec_file.write(reset()); rec_file.close()
                        except Exception:
                            pass
                        rec_file = None
                elif kk == 'r':
                    SCENES[scene_idx].reset(vw, vh); last_content=None
                elif kk == 'q' or k == '\x1b':
                    return scene_idx

            frames += 1
            if now - fps_t0 >= 0.5:
                fps = frames / (now - fps_t0)
                fps_t0 = now
                frames = 0

            progress = None
            if autoshow:
                elapsed = now - scene_start
                progress = clamp(elapsed / auto_period)
                if elapsed >= auto_period:
                    auto_i = (auto_i + 1) % len(auto_list)
                    new_idx = auto_list[auto_i]
                    if last_content is not None:
                        tnow = time.time()
                        next_scene = SCENES[new_idx]
                        next_scene.reset(vw, vh)
                        nxt = next_scene.step(vw, vh, tnow, 0.0, speed, theme, 1.0)
                        wipe_frames = transition_wipe(last_content, nxt, steps=18)
                        for i, fr in enumerate(wipe_frames):
                            if 5 <= i <= 12 and random.random() < 0.55:
                                fr = glitch_frame(fr, intensity=0.14)
                            hint = '1..9  J/K/L/M/N'
                            header = make_header(theme, f"AUTO-SHOW → {SCENES[new_idx].key}) {SCENES[new_idx].name}", hud,
                                                 cols, lines, vw, vh, fps, speed, paused, autoshow, rec_on, 1.0,
                                                 hint, i/len(wipe_frames), '')
                            screen = frame_to_screen(header, pad_lines(fr, left), top)
                            home(); _w(screen); flush(); time.sleep(0.018)
                    scene_idx = new_idx
                    SCENES[scene_idx].reset(vw, vh)
                    scene_start = now
                    last_content = None

            tnow = time.time()
            dt_eff = 0.0 if paused else dt
            scene = SCENES[scene_idx]
            quality = compute_quality(auto_quality, theme, scene, vw, vh, fps)
            content = scene.step(vw, vh, tnow, dt_eff, speed, theme, quality)
            if show_help:
                content = overlay_help(content, vw, vh, theme)
            if rec_on and rec_file:
                try:
                    rec_file.write(CSI + 'H'); rec_file.write('\n'.join(content) + '\n')
                except Exception:
                    pass
            content_padded = pad_lines(content, left)
            hint = '1..9  J/K/L/M/N'
            mode_label = ''
            if getattr(scene, 'supports_mode', False) and hasattr(scene, 'mode'):
                mode_label = f"  | modo {getattr(scene, 'mode', 0)}"
            header = make_header(theme, f"{scene.key}) {scene.name}", hud,
                                 cols, lines, vw, vh, fps, speed, paused, autoshow, rec_on, quality,
                                 hint, progress, mode_label)
            home(); _w(frame_to_screen(header, content_padded, top)); flush()
            last_content = content

            target = 1/45
            spare = target - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)

    except KeyboardInterrupt:
        return scene_idx
    finally:
        try:
            if rec_file:
                rec_file.write(reset()); rec_file.close()
        except Exception:
            pass
        show_cursor(); _w(reset() + '\n')


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    theme = Theme()
    scene_idx = 0
    while True:
        menu(theme)
        try:
            choice = input('Escolha (1-14), A/T/C/X/H ou 0: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if choice in KEY_TO_SCENE:
            scene_idx = KEY_TO_SCENE[choice]
            scene_idx = run_scene(scene_idx, theme, autoshow=False)
        elif choice == 'a':
            scene_idx = 0
            scene_idx = run_scene(scene_idx, theme, autoshow=True)
        elif choice == 't':
            theme.next_theme()
        elif choice == 'c':
            theme.toggle_color()
        elif choice == 'x':
            theme.set_theme('Suave'); theme.color_enabled = True
        elif choice == 'h':
            scene_idx = run_scene(scene_idx, theme, autoshow=False)
        elif choice == '0':
            break
        else:
            time.sleep(0.25)
    clear(); show_cursor(); _w(reset() + 'Até mais! 👋\n')


if __name__ == '__main__':
    main()
