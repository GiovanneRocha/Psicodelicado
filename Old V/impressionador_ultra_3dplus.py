#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMPRESSIONADOR ULTRA 3D+ (Windows-friendly) 🚀

Essência mantida (principais):
  1) Donut 3D girando
  2) Chuva Matrix
  3) Fractal Mandelbrot

Extras já inclusos:
  4) Hyperspace Starfield
  5) DOOM Fire
  6) Plasma / Nebula
  7) Metaballs
  8) Game of Life

NOVO: Cenas "3D fake" (doideira no terminal) ✅
  9) Túnel Infinito (pseudo-3D)
 10) Wireframe Rotating Cube (cubo aramado girando)
 11) Terrain Wave (terreno ondulado em perspectiva)

✅ Pedidos atendidos:
- Usa praticamente a tela toda (viewport quase full-screen + margem anti-wrap)
- T: tema de cores do NORMAL ao PSICODÉLICO (inclui MONO)
- Indicadores: FPS, resolução, viewport, cena, tema, velocidade, cor, pausa

Controles (em qualquer cena):
  1..11 -> troca de cena instantânea
  T     -> troca tema (Normal -> Suave -> Neon -> Psicodélico -> Mono)
  C     -> cor ON/OFF
  +/-   -> velocidade
  P     -> pausa
  R     -> reset da cena
  H     -> ajuda (overlay)
  Q/ESC -> menu
  Ctrl+C -> sair

Dica:
  Quanto maior o terminal (mais COLS), mais detalhes em TUDO.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
import shutil

IS_WINDOWS = (os.name == 'nt')
CSI = "\x1b["

# -----------------------------
# Habilitar ANSI/VT no Windows
# -----------------------------
if IS_WINDOWS:
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            k32.SetConsoleMode(h, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    except Exception:
        pass

# -----------------------------
# Terminal helpers
# -----------------------------

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
    s = shutil.get_terminal_size((140, 45))
    return s.columns, s.lines

# -----------------------------
# Teclado não-bloqueante (Windows)
# -----------------------------

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

# -----------------------------
# Viewport (centralizado + anti-wrap)
# -----------------------------

def viewport(cols: int, lines: int, header: int = 3, margin: int = 1):
    """Área útil quase full-screen, centralizada, com margem anti-wrap."""
    usable_h = max(12, lines - header)
    usable_w = max(30, cols - 1)  # evita última coluna

    vw = max(30, usable_w - 2*margin)
    vh = max(12, usable_h - 1)

    left = max(0, (usable_w - vw) // 2)
    top = max(0, (usable_h - vh) // 2)

    return vw, vh, left, top


def pad_lines(lines_list, left_pad: int):
    if left_pad <= 0:
        return lines_list
    pad = ' ' * left_pad
    return [pad + ln for ln in lines_list]


def frame_to_screen(header_lines: list[str], content_lines: list[str], top_pad: int):
    out = []
    out.extend(header_lines)
    out.extend([''] * top_pad)
    out.extend(content_lines)
    return '\n'.join(out)

# -----------------------------
# Temas / cores
# -----------------------------

def clampi(x: float, lo: int = 0, hi: int = 255) -> int:
    if x < lo: return lo
    if x > hi: return hi
    return int(x)

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo: return lo
    if x > hi: return hi
    return x

def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int,int,int]:
    h = (h % 1.0) * 6.0
    i = int(h)
    f = h - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if i == 0: r,g,b = v,t,p
    elif i == 1: r,g,b = q,v,p
    elif i == 2: r,g,b = p,v,t
    elif i == 3: r,g,b = p,q,v
    elif i == 4: r,g,b = t,p,v
    else: r,g,b = v,p,q
    return (clampi(r*255), clampi(g*255), clampi(b*255))

THEMES = [
    ("Normal", "cores clássicas e sóbrias"),
    ("Suave", "menos saturação, confortável"),
    ("Neon", "vibrante sem virar arco-íris"),
    ("Psicodélico", "arco-íris dinâmico total"),
    ("Mono", "sem cor (P&B)"),
]

class Theme:
    def __init__(self):
        self.idx = 0
        self.color_enabled = True

    @property
    def name(self):
        return THEMES[self.idx][0]

    @property
    def desc(self):
        return THEMES[self.idx][1]

    def toggle_color(self):
        self.color_enabled = not self.color_enabled

    def next_theme(self):
        self.idx = (self.idx + 1) % len(THEMES)

    def is_mono(self):
        return self.name == "Mono" or not self.color_enabled

    # UI colors
    def ui_title(self, s: str) -> str:
        if self.is_mono():
            return bold() + s + reset()
        if self.name == "Neon":
            return bold() + fg256(207) + s + reset()
        if self.name == "Psicodélico":
            return bold() + fg256(226) + s + reset()
        if self.name == "Suave":
            return bold() + fg256(81) + s + reset()
        return bold() + fg256(39) + s + reset()

    def ui_subtle(self, s: str) -> str:
        if self.is_mono():
            return dim() + s + reset()
        return dim() + fg256(245) + s + reset()

    def ui_ok(self, s: str) -> str:
        if self.is_mono():
            return s
        return fg256(46) + s + reset()

    # Scene palettes
    def donut(self, lum: float, t: float) -> str:
        if self.is_mono():
            return ''
        lum = clamp(lum)
        if self.name == "Normal":
            r = int(20 + 35*lum)
            g = int(120 + 95*lum)
            b = int(170 + 75*lum)
            return fg_rgb(r,g,b)
        if self.name == "Suave":
            r = int(25 + 25*lum)
            g = int(120 + 65*lum)
            b = int(155 + 55*lum)
            return fg_rgb(r,g,b)
        if self.name == "Neon":
            r = int(80 + 140*lum)
            g = int(40 + 200*lum)
            b = int(160 + 80*lum)
            return fg_rgb(r,g,b)
        # Psicodélico
        r,g,b = hsv_to_rgb((t*0.12 + lum*0.35) % 1.0, 0.95, 0.45 + 0.55*lum)
        return fg_rgb(r,g,b)

    def mandel(self, k: float, t: float) -> str:
        if self.is_mono():
            return ''
        k = clamp(k)
        if self.name == "Normal":
            if k < 0.6:
                u = k/0.6
                r = int(0 + 50*u)
                g = int(70 + 150*u)
                b = int(170 + 60*u)
            else:
                u = (k-0.6)/0.4
                r = int(50 + 180*u)
                g = int(220 + 20*u)
                b = int(230 - 170*u)
            return fg_rgb(r,g,b)
        if self.name == "Suave":
            r,g,b = hsv_to_rgb(0.60 + 0.10*k, 0.45, 0.55 + 0.35*(k**0.5))
            return fg_rgb(r,g,b)
        if self.name == "Neon":
            r,g,b = hsv_to_rgb(0.75 + 0.55*k, 0.95, 0.35 + 0.65*(k**0.45))
            return fg_rgb(r,g,b)
        r,g,b = hsv_to_rgb((t*0.08 + 2.8*k) % 1.0, 0.98, 0.25 + 0.75*(k**0.35))
        return fg_rgb(r,g,b)

    def matrix(self, intensity: float, t: float, highlight: float = 0.0) -> str:
        if self.is_mono():
            return ''
        intensity = clamp(intensity)
        if self.name in ("Normal", "Suave"):
            if highlight > 0.7:
                return fg256(120)
            if highlight > 0.35:
                return fg256(82)
            return fg256(46)
        if self.name == "Neon":
            r,g,b = hsv_to_rgb(0.40 + 0.08*math.sin(t*0.8), 0.95, 0.35 + 0.65*intensity)
            return fg_rgb(r,g,b)
        r,g,b = hsv_to_rgb((t*0.15 + intensity*0.9) % 1.0, 0.98, 0.25 + 0.75*intensity)
        return fg_rgb(r,g,b)

    def fire(self, idx: int, t: float) -> str:
        if self.is_mono():
            return ''
        idx = max(0, min(35, idx))
        if self.name == "Normal":
            palette = [16, 52, 88, 124, 160, 196, 202, 208, 214, 220, 226]
            c = palette[min(len(palette)-1, idx//4)]
            return fg256(c)
        if self.name == "Suave":
            palette = [16, 52, 88, 124, 160, 196, 202, 208, 214]
            c = palette[min(len(palette)-1, idx//5)]
            return fg256(c)
        if self.name == "Neon":
            r,g,b = hsv_to_rgb(0.02 + 0.08*(idx/35), 0.95, 0.30 + 0.70*(idx/35))
            return fg_rgb(r,g,b)
        r,g,b = hsv_to_rgb((t*0.2 + (idx/35)*0.35) % 1.0, 0.98, 0.25 + 0.75*(idx/35))
        return fg_rgb(r,g,b)

    # Generic color for "3D fake" effects
    def fx(self, k: float, t: float, base_h: float = 0.6) -> str:
        if self.is_mono():
            return ''
        k = clamp(k)
        if self.name == 'Normal':
            r,g,b = hsv_to_rgb(base_h + 0.12*k, 0.55, 0.20 + 0.80*k)
            return fg_rgb(r,g,b)
        if self.name == 'Suave':
            r,g,b = hsv_to_rgb(base_h + 0.08*k, 0.35, 0.25 + 0.75*k)
            return fg_rgb(r,g,b)
        if self.name == 'Neon':
            r,g,b = hsv_to_rgb(base_h + 0.55*k, 0.95, 0.20 + 0.80*k)
            return fg_rgb(r,g,b)
        r,g,b = hsv_to_rgb((t*0.12 + 1.8*k) % 1.0, 0.98, 0.15 + 0.85*k)
        return fg_rgb(r,g,b)

# -----------------------------
# Base de cenas
# -----------------------------

class Scene:
    key = '0'
    name = 'Scene'
    def reset(self, vw: int, vh: int):
        pass
    def step(self, vw: int, vh: int, t: float, dt: float, speed: float, theme: Theme) -> list[str]:
        raise NotImplementedError

# -----------------------------
# 1) Donut 3D
# -----------------------------

class Donut(Scene):
    key = '1'
    name = 'Donut 3D girando'

    def __init__(self):
        self.A = 0.0
        self.B = 0.0

    def reset(self, vw, vh):
        self.A = 0.0
        self.B = 0.0

    def step(self, vw, vh, t, dt, speed, theme: Theme):
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
        while theta < 2*math.pi:
            costheta = math.cos(theta)
            sintheta = math.sin(theta)
            phi = 0.0
            while phi < 2*math.pi:
                cosphi = math.cos(phi)
                sinphi = math.sin(phi)

                circlex = R2 + R1*costheta
                circley = R1*sintheta

                x = circlex * (cosB*cosphi + sinA*sinB*sinphi) - circley*cosA*sinB
                y = circlex * (sinB*cosphi - sinA*cosB*sinphi) + circley*cosA*cosB
                z = K2 + cosA*circlex*sinphi + circley*sinA
                ooz = 1.0 / z

                xp = int(vw/2 + K1*ooz*x)
                yp = int(vh/2 - K1*ooz*y)

                L = (cosphi*costheta*sinB
                     - cosA*costheta*sinphi
                     - sinA*sintheta
                     + cosB*(cosA*sintheta - costheta*sinA*sinphi))

                if L > 0 and 0 <= xp < vw and 0 <= yp < vh:
                    idx = xp + vw*yp
                    if ooz > zbuffer[idx]:
                        zbuffer[idx] = ooz
                        li = int(L * (len(chars)-1))
                        li = max(0, min(li, len(chars)-1))
                        output[idx] = chars[li]
                        lum[idx] = min(1.0, L)

                phi += 0.07
            theta += 0.03

        lines = []
        for y in range(vh):
            start = y*vw
            row_chars = output[start:start+vw]
            if theme.is_mono():
                lines.append(''.join(row_chars))
            else:
                row = []
                for i, ch in enumerate(row_chars):
                    if ch == ' ':
                        row.append(' ')
                    else:
                        l = lum[start+i]
                        row.append(theme.donut(l, t) + ch + reset())
                lines.append(''.join(row))

        self.A += 0.04 * speed
        self.B += 0.02 * speed
        return lines

# -----------------------------
# 2) Matrix Rain
# -----------------------------

class Matrix(Scene):
    key = '2'
    name = 'Chuva Matrix'

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

    def step(self, vw, vh, t, dt, speed, theme: Theme):
        if len(self.drops) != vw:
            self.reset(vw, vh)

        now = time.time()
        canvas = [list(' ' * vw) for _ in range(vh)]
        head = [0.0] * vw

        for x in range(vw):
            if now - self.last[x] >= (0.02 / (self.speeds[x] * speed)):
                self.last[x] = now
                self.drops[x] += 1
                if self.drops[x] > vh + random.randint(0, vh//2 + 1):
                    self.drops[x] = random.randint(-vh, 0)
                    self.speeds[x] = random.uniform(0.7, 2.8)

            y = self.drops[x]
            head[x] = clamp(1.0 - abs((vh/2 - y) / (vh/2 + 1e-9)))

            if 0 <= y < vh:
                canvas[y][x] = random.choice(self.charset)

            for tr in range(1, 12):
                yy = y - tr
                if 0 <= yy < vh and random.random() < 0.65:
                    canvas[yy][x] = random.choice(self.charset)

        lines = []
        if theme.is_mono():
            for y in range(vh):
                lines.append(''.join(canvas[y]))
            return lines

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

# -----------------------------
# 3) Mandelbrot
# -----------------------------

class Mandel(Scene):
    key = '3'
    name = 'Fractal Mandelbrot'

    def __init__(self):
        self.phase = 0.0
        self.cx, self.cy = -0.5, 0.0
        self.base_scale = 3.0
        self.ramp = " .:-=+*#%@"

    def reset(self, vw, vh):
        self.phase = 0.0

    def step(self, vw, vh, t, dt, speed, theme: Theme):
        self.phase += 0.055 * speed
        zoom = 0.35 + 0.65*(0.5 + 0.5*math.sin(self.phase*0.35))
        scale = (0.22 + (self.base_scale - 0.22) * zoom)
        max_iter = int(70 + (1.0 / max(scale, 0.2)) * 55)
        aspect = (vh / vw) * 1.65

        lines = []
        mono = theme.is_mono()
        for py in range(vh):
            y0 = self.cy + (py / vh - 0.5) * scale * aspect
            parts = []
            for px in range(vw):
                x0 = self.cx + (px / vw - 0.5) * scale
                x = 0.0
                y = 0.0
                it = 0
                while x*x + y*y <= 4.0 and it < max_iter:
                    x, y = x*x - y*y + x0, 2*x*y + y0
                    it += 1

                if it == max_iter:
                    parts.append(' ')
                else:
                    k = it / max_iter
                    ch = self.ramp[int(k * (len(self.ramp)-1))]
                    if mono:
                        parts.append(ch)
                    else:
                        parts.append(theme.mandel(k, t) + ch + reset())
            lines.append(''.join(parts))
        return lines

# -----------------------------
# 4) Hyperspace Starfield
# -----------------------------

class Starfield(Scene):
    key = '4'
    name = 'Hyperspace Starfield'

    def __init__(self):
        self.stars = []
        self.seed = 777

    def reset(self, vw, vh):
        random.seed(self.seed)
        n = max(500, int(vw*vh*0.10))
        self.stars = [[random.uniform(-1,1), random.uniform(-1,1), random.uniform(0.05, 1.0)] for _ in range(n)]

    def step(self, vw, vh, t, dt, speed, theme: Theme):
        if not self.stars:
            self.reset(vw, vh)
        canvas = [[(' ', '') for _ in range(vw)] for _ in range(vh)]
        cx, cy = vw/2, vh/2
        sp = (1.0 + 1.6*(0.5+0.5*math.sin(t*0.6))) * speed

        for s in self.stars:
            s[2] -= dt * 0.7 * sp
            if s[2] <= 0.03:
                s[0] = random.uniform(-1,1)
                s[1] = random.uniform(-1,1)
                s[2] = 1.0

            x,y,z = s
            px = int(cx + (x/z) * (vw*0.38))
            py = int(cy + (y/z) * (vh*0.38))
            if 0 <= px < vw and 0 <= py < vh:
                bright = clamp((1.0 - z)**0.3)
                ch = '*' if bright > 0.8 else ('.' if bright > 0.45 else '·')
                if theme.is_mono():
                    canvas[py][px] = (ch, '')
                else:
                    col = theme.fx(bright, t, base_h=0.62)
                    canvas[py][px] = (ch, col)

        lines = []
        for y in range(vh):
            row = canvas[y]
            if theme.is_mono():
                lines.append(''.join(ch for ch,_ in row))
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

# -----------------------------
# 5) DOOM Fire
# -----------------------------

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
            self.heat[vh-1][x] = 35

    def step(self, vw, vh, t, dt, speed, theme: Theme):
        if self.w != vw or self.h != vh or not self.heat:
            self.reset(vw, vh)

        for x in range(vw):
            self.heat[vh-1][x] = 30 + int(5*random.random())

        for y in range(vh-2, -1, -1):
            row = self.heat[y]
            below = self.heat[y+1]
            for x in range(vw):
                src = below[x]
                decay = int(random.random()*3)
                wind = int(random.random()*3) - 1
                nx = x + wind
                if nx < 0: nx = 0
                if nx >= vw: nx = vw-1
                val = src - decay
                if val < 0: val = 0
                row[nx] = val

        chars = " .,:;irsXA253hMHGS#9B&@"
        lines = []
        mono = theme.is_mono()
        for y in range(vh):
            parts = []
            for x in range(vw):
                hval = self.heat[y][x]
                ch = chars[min(len(chars)-1, int(hval/35*(len(chars)-1)))]
                if mono:
                    parts.append(ch)
                else:
                    parts.append(theme.fire(hval, t) + ch + reset())
            lines.append(''.join(parts))

        if speed > 1.25 and random.random() < 0.35:
            for y in range(vh-2, -1, -1):
                row = self.heat[y]
                below = self.heat[y+1]
                for x in range(vw):
                    src = below[x]
                    decay = int(random.random()*3)
                    wind = int(random.random()*3) - 1
                    nx = x + wind
                    if nx < 0: nx = 0
                    if nx >= vw: nx = vw-1
                    val = src - decay
                    if val < 0: val = 0
                    row[nx] = val

        return lines

# -----------------------------
# 6) Plasma / Nebula
# -----------------------------

class Plasma(Scene):
    key = '6'
    name = 'Plasma / Nebula'

    def step(self, vw, vh, t, dt, speed, theme: Theme):
        mono = theme.is_mono()
        ramp = " .:-=+*#%@"
        lines = []

        tt = t * 0.7 * speed
        for y in range(vh):
            ny = (y / vh - 0.5) * 2.0
            parts = []
            for x in range(vw):
                nx = (x / vw - 0.5) * 2.0
                v = 0.0
                v += math.sin((nx*3.1 + tt) * 1.1)
                v += math.sin((ny*4.0 - tt*1.2) * 1.2)
                v += math.sin((nx*2.0 + ny*2.0 + tt) * 1.0)
                v += math.sin(math.sqrt(nx*nx + ny*ny) * 6.0 - tt*1.7)
                v = (v / 4.0 + 1.0) / 2.0
                ch = ramp[int(v * (len(ramp)-1))]

                if mono:
                    parts.append(ch)
                else:
                    col = theme.fx(v, t, base_h=0.70)
                    parts.append(col + ch + reset())
            lines.append(''.join(parts))
        return lines

# -----------------------------
# 7) Metaballs 2D
# -----------------------------

class Metaballs(Scene):
    key = '7'
    name = 'Metaballs (bolhas orgânicas)'

    def __init__(self):
        self.balls = []

    def reset(self, vw, vh):
        n = 5
        self.balls = []
        for _ in range(n):
            self.balls.append([
                random.random()*vw,
                random.random()*vh,
                (random.random()*0.8+0.4) * (min(vw,vh)*0.12),
                random.uniform(-1.2, 1.2),
                random.uniform(-0.9, 0.9),
            ])

    def step(self, vw, vh, t, dt, speed, theme: Theme):
        if not self.balls:
            self.reset(vw, vh)

        for b in self.balls:
            b[0] += b[3] * speed
            b[1] += b[4] * speed
            if b[0] < 0 or b[0] > vw-1:
                b[3] *= -1
                b[0] = max(0, min(vw-1, b[0]))
            if b[1] < 0 or b[1] > vh-1:
                b[4] *= -1
                b[1] = max(0, min(vh-1, b[1]))

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
                    d2 = dx*dx + dy*dy + 1e-6
                    v += (r*r) / d2
                v = v / (len(self.balls)*1.2)
                v = clamp(v/2.2)
                ch = ramp[int(v*(len(ramp)-1))]

                if mono:
                    parts.append(ch)
                else:
                    parts.append(theme.fx(v, t, base_h=0.85) + ch + reset())
            lines.append(''.join(parts))
        return lines

# -----------------------------
# 8) Game of Life
# -----------------------------

class Life(Scene):
    key = '8'
    name = 'Game of Life'

    def __init__(self):
        self.grid = []
        self.w = 0
        self.h = 0
        self.age = []

    def reset(self, vw, vh):
        self.w, self.h = vw, vh
        self.grid = [[1 if random.random() < 0.16 else 0 for _ in range(vw)] for _ in range(vh)]
        self.age = [[0 for _ in range(vw)] for _ in range(vh)]

    def step(self, vw, vh, t, dt, speed, theme: Theme):
        if self.w != vw or self.h != vh or not self.grid:
            self.reset(vw, vh)

        steps = 1
        if speed > 1.4: steps = 2
        if speed > 2.2: steps = 3

        for _ in range(steps):
            ng = [[0]*vw for _ in range(vh)]
            nage = [[0]*vw for _ in range(vh)]
            for y in range(vh):
                ym1 = (y-1) % vh
                yp1 = (y+1) % vh
                row = self.grid[y]
                for x in range(vw):
                    xm1 = (x-1) % vw
                    xp1 = (x+1) % vw
                    n = (
                        self.grid[ym1][xm1] + self.grid[ym1][x] + self.grid[ym1][xp1] +
                        row[xm1] + row[xp1] +
                        self.grid[yp1][xm1] + self.grid[yp1][x] + self.grid[yp1][xp1]
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

# =============================
# 9) Túnel Infinito (pseudo-3D)
# =============================

class Tunnel(Scene):
    key = '9'
    name = 'Túnel Infinito (pseudo-3D)'

    def __init__(self):
        self.phase = 0.0

    def reset(self, vw, vh):
        self.phase = 0.0

    def step(self, vw, vh, t, dt, speed, theme: Theme):
        self.phase += dt * 1.0 * speed
        ramp = " .:-=+*#%@"
        mono = theme.is_mono()
        cx = (vw - 1) / 2
        cy = (vh - 1) / 2
        aspect = vw / max(1, vh)

        lines = []
        for y in range(vh):
            parts = []
            dy = (y - cy) / (vh/2)
            for x in range(vw):
                dx = (x - cx) / (vw/2)
                dx *= aspect
                r = math.sqrt(dx*dx + dy*dy) + 1e-6
                ang = math.atan2(dy, dx)

                # tunnel texture coords
                z = 1.0 / r
                u = (ang / (2*math.pi) + 0.5) * 3.0
                v = z * 0.18 + self.phase

                # pattern: rings + twist
                p = 0.5 + 0.5*math.sin((u + v*4.0) * 2*math.pi)
                q = 0.5 + 0.5*math.cos((v*2.2 - u*0.3) * 2*math.pi)
                k = clamp((p*0.65 + q*0.35) * (1.0 - clamp(r*0.65)))

                ch = ramp[int(k*(len(ramp)-1))]
                if mono:
                    parts.append(ch)
                else:
                    # color based on depth z (more depth = brighter)
                    kk = clamp((z/6.0))
                    parts.append(theme.fx(kk, t, base_h=0.62) + ch + reset())
            lines.append(''.join(parts))
        return lines

# =============================
# 10) Wireframe Rotating Cube
# =============================

class Cube(Scene):
    key = '10'
    name = 'Wireframe Rotating Cube'

    def __init__(self):
        self.ang = 0.0

    def reset(self, vw, vh):
        self.ang = 0.0

    @staticmethod
    def _rot(v, ax, ay, az):
        x,y,z = v
        # rot x
        cx,sx = math.cos(ax), math.sin(ax)
        y,z = y*cx - z*sx, y*sx + z*cx
        # rot y
        cy,sy = math.cos(ay), math.sin(ay)
        x,z = x*cy + z*sy, -x*sy + z*cy
        # rot z
        cz,sz = math.cos(az), math.sin(az)
        x,y = x*cz - y*sz, x*sz + y*cz
        return (x,y,z)

    @staticmethod
    def _bresenham(x0,y0,x1,y1):
        pts = []
        dx = abs(x1-x0)
        dy = -abs(y1-y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x,y = x0,y0
        while True:
            pts.append((x,y))
            if x == x1 and y == y1:
                break
            e2 = 2*err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return pts

    def step(self, vw, vh, t, dt, speed, theme: Theme):
        self.ang += dt * 1.2 * speed
        ax = self.ang * 0.9
        ay = self.ang * 1.1
        az = self.ang * 0.7

        # cube vertices
        s = 1.0
        verts = [
            (-s,-s,-s), ( s,-s,-s), ( s, s,-s), (-s, s,-s),
            (-s,-s, s), ( s,-s, s), ( s, s, s), (-s, s, s)
        ]
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]

        # rotate + project
        proj = []
        depth = []
        for v in verts:
            x,y,z = self._rot(v, ax, ay, az)
            z += 3.2
            # perspective
            f = 1.8 / z
            px = x * f
            py = y * f
            sxp = int((px + 0.0) * (vw*0.35) + vw/2)
            syp = int((-py + 0.0) * (vh*0.55) + vh/2)
            proj.append((sxp, syp))
            depth.append(clamp(1.0 - (z-2.2)/3.5))

        # buffer
        grid = [[' ' for _ in range(vw)] for _ in range(vh)]
        zbuf = [[-1e9 for _ in range(vw)] for _ in range(vh)]

        mono = theme.is_mono()
        # draw edges with depth interpolation
        for a,b in edges:
            x0,y0 = proj[a]
            x1,y1 = proj[b]
            pts = self._bresenham(x0,y0,x1,y1)
            for i,(x,y) in enumerate(pts):
                if 0 <= x < vw and 0 <= y < vh:
                    tt = i / max(1, len(pts)-1)
                    d = depth[a]*(1-tt) + depth[b]*tt
                    if d > zbuf[y][x]:
                        zbuf[y][x] = d
                        # char by depth
                        ch = '#' if d > 0.75 else ('*' if d > 0.45 else '.')
                        grid[y][x] = ch

        # optional: vertices
        for i,(x,y) in enumerate(proj):
            if 0 <= x < vw and 0 <= y < vh:
                grid[y][x] = '@'
                zbuf[y][x] = max(zbuf[y][x], depth[i])

        # colorize line-by-line (simple)
        lines = []
        ramp = " .:-=+*#%@"
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
                        d = clamp(zbuf[y][x])
                        parts.append(theme.fx(d, t, base_h=0.78) + ch + reset())
                lines.append(''.join(parts))
        return lines

# =============================
# 11) Terrain Wave (terreno)
# =============================

class Terrain(Scene):
    key = '11'
    name = 'Terrain Wave (perspectiva)'

    def __init__(self):
        self.off = 0.0

    def reset(self, vw, vh):
        self.off = 0.0

    def height(self, x, z, t):
        # combinação de ondas (barato, bonito)
        return (
            math.sin(x*0.45 + t*0.9) * 0.55 +
            math.cos(z*0.28 + t*0.7) * 0.65 +
            math.sin((x+z)*0.22 - t*0.6) * 0.45
        )

    def step(self, vw, vh, t, dt, speed, theme: Theme):
        self.off += dt * 1.2 * speed
        mono = theme.is_mono()

        # frame buffers
        grid = [[' ' for _ in range(vw)] for _ in range(vh)]
        zbuf = [[-1e9 for _ in range(vw)] for _ in range(vh)]

        # perspective parameters
        horizon = int(vh * 0.30)
        scale_y = vh * 0.18
        scale_x = vw * 0.55

        # render from near to far (painter-ish)
        # z grows away from camera
        for zi in range(1, 110):
            z = zi * 0.15 + self.off
            # depth factor 0..1
            depth = zi / 110.0
            # world width shrinks with distance
            inv = 1.0 / (0.55 + depth*2.6)
            for xi in range(vw):
                # map screen x to world x
                x = (xi - vw/2) * inv * 0.12
                h = self.height(x*10.0, z*10.0, t)
                # project to screen
                sy = int(horizon + (h * scale_y) * inv)
                sy = max(0, min(vh-1, sy))
                # draw vertical column downward to create "surface"
                # choose char by depth + slope hint
                k = clamp(1.0 - depth)
                ch = '█' if k > 0.75 else ('▓' if k > 0.5 else ('▒' if k > 0.28 else '·'))

                # fill from sy to bottom with decreasing density for far depths
                # but don't overdraw close surfaces
                for y in range(sy, vh):
                    if k > zbuf[y][xi]:
                        zbuf[y][xi] = k
                        grid[y][xi] = ch

        # add sky with subtle stars
        for y in range(0, int(vh*0.30)):
            for x in range(vw):
                if grid[y][x] == ' ' and random.random() < 0.0012:
                    grid[y][x] = '·'
                    zbuf[y][x] = 0.15

        lines = []
        if mono:
            for y in range(vh):
                lines.append(''.join(grid[y]))
            return lines

        for y in range(vh):
            parts = []
            for x in range(vw):
                ch = grid[y][x]
                if ch == ' ':
                    parts.append(' ')
                else:
                    k = clamp(zbuf[y][x])
                    # terrain tends to green/blue; vary base hue
                    base_h = 0.33 if y > int(vh*0.32) else 0.62
                    parts.append(theme.fx(k, t, base_h=base_h) + ch + reset())
            lines.append(''.join(parts))
        return lines

# -----------------------------
# App
# -----------------------------

SCENES = [
    Donut(), Matrix(), Mandel(), Starfield(), DoomFire(), Plasma(), Metaballs(), Life(),
    Tunnel(), Cube(), Terrain()
]
KEY_TO_SCENE = {s.key: i for i, s in enumerate(SCENES)}

HELP_TEXT = [
    "Controles:",
    "  1..11 troca de cena", 
    "  T     troca tema (Normal/Suave/Neon/Psicodélico/Mono)",
    "  C     cor ON/OFF", 
    "  +/-   velocidade", 
    "  P     pausa", 
    "  R     reset", 
    "  H     ajuda", 
    "  Q/ESC menu", 
    "  Ctrl+C sair",
    "",
    "Dica: Mais COLS = mais detalhe. No Windows: mode con: cols=200 lines=60",
]


def overlay_help(content: list[str], vw: int, vh: int, theme: Theme) -> list[str]:
    box_w = min(vw-4, max(42, max(len(s) for s in HELP_TEXT) + 4))
    box_h = min(vh-4, len(HELP_TEXT) + 4)
    x0 = (vw - box_w)//2
    y0 = (vh - box_h)//2

    grid = [list(line.ljust(vw)[:vw]) for line in content]
    tl, tr, bl, br = '┌','┐','└','┘'
    hz, vt = '─','│'

    def paint(x, y, s):
        if 0 <= y < vh:
            for i,ch in enumerate(s):
                xx = x + i
                if 0 <= xx < vw:
                    grid[y][xx] = ch

    paint(x0, y0, tl + hz*(box_w-2) + tr)
    for yy in range(1, box_h-1):
        paint(x0, y0+yy, vt + ' '*(box_w-2) + vt)
    paint(x0, y0+box_h-1, bl + hz*(box_w-2) + br)

    title = " AJUDA "
    paint(x0 + (box_w-len(title))//2, y0, title)

    for i, line in enumerate(HELP_TEXT[:box_h-4]):
        paint(x0+2, y0+2+i, line[:box_w-4].ljust(box_w-4))

    out = [''.join(r) for r in grid]
    if theme.is_mono():
        return out

    col = fg256(214) if theme.name == 'Psicodélico' else (fg256(81) if theme.name in ('Normal','Suave') else fg256(207))
    out2 = []
    for y, line in enumerate(out):
        if y0 <= y <= y0+box_h-1:
            out2.append(col + line + reset())
        else:
            out2.append(line)
    return out2


def _strip_ansi(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s[i] == '\x1b':
            j = i+1
            while j < len(s) and s[j] != 'm':
                j += 1
            i = j+1
        else:
            out.append(s[i]); i += 1
    return ''.join(out)


def menu(theme: Theme):
    clear(); show_cursor()
    cols, lines = term_size()

    title = 'PYTHON IMPRESSIONADOR ULTRA 3D+'
    subtitle = 'Essência: Donut 3D • Matrix • Mandelbrot  |  + extras + 3D fake (9..11)'
    status = f"Tema: {theme.name} ({theme.desc})  |  Cor: {'ON' if theme.color_enabled and theme.name!='Mono' else 'OFF'}"

    items = [
        f"{bold()}{title}{reset()}",
        subtitle,
        '',
        '1) Donut 3D girando',
        '2) Chuva Matrix',
        '3) Fractal Mandelbrot',
        '4) Hyperspace Starfield',
        '5) DOOM Fire',
        '6) Plasma / Nebula',
        '7) Metaballs',
        '8) Game of Life',
        '9) Túnel Infinito (pseudo-3D)',
        '10) Wireframe Rotating Cube',
        '11) Terrain Wave (perspectiva)',
        '',
        'T) Trocar tema  |  C) Cor ON/OFF  |  H) Ajuda',
        '0) Sair',
        '',
        status,
        '',
        'Dica: maximize o terminal (mais colunas = mais detalhe).',
    ]

    w = max(len(_strip_ansi(l)) for l in items)
    left = max(0, (cols - w)//2)
    top = max(0, (lines - len(items))//2)

    _w('\n'*top)
    for l in items:
        _w(' '*left + l + '\n')
    flush()


def run_scene(scene_idx: int, theme: Theme) -> int:
    clear(); hide_cursor()

    paused = False
    show_help = False
    speed = 1.0

    last_cols, last_lines = term_size()
    vw, vh, left, top = viewport(last_cols, last_lines, header=3, margin=1)
    SCENES[scene_idx].reset(vw, vh)

    last_t = time.perf_counter()
    fps_t0 = last_t
    frames = 0
    fps = 0.0

    try:
        while True:
            now = time.perf_counter()
            dt = now - last_t
            last_t = now

            cols, lines = term_size()
            if cols != last_cols or lines != last_lines:
                clear()
                last_cols, last_lines = cols, lines
                vw, vh, left, top = viewport(cols, lines, header=3, margin=1)
                SCENES[scene_idx].reset(vw, vh)

            k = read_key()
            if k:
                kk = k.lower()
                # map: allow '0'.. '9' and '10'/'11' from menu only; in-scene use F-keys not feasible.
                # We keep instant swap for 1..9 via single key, and add J/K for 10/11.
                if kk in KEY_TO_SCENE and kk != '10' and kk != '11':
                    scene_idx = KEY_TO_SCENE[kk]
                    SCENES[scene_idx].reset(vw, vh)
                elif kk == 'j':
                    scene_idx = KEY_TO_SCENE['10']; SCENES[scene_idx].reset(vw, vh)
                elif kk == 'k':
                    scene_idx = KEY_TO_SCENE['11']; SCENES[scene_idx].reset(vw, vh)
                elif kk == 't':
                    theme.next_theme(); clear()
                elif kk == 'c':
                    theme.toggle_color(); clear()
                elif kk in ('+','='):
                    speed = min(3.0, speed + 0.1)
                elif kk in ('-','_'):
                    speed = max(0.2, speed - 0.1)
                elif kk == 'p':
                    paused = not paused
                elif kk == 'r':
                    SCENES[scene_idx].reset(vw, vh)
                elif kk == 'h':
                    show_help = not show_help
                elif kk == 'q' or k == '\x1b':
                    return scene_idx

            t = time.time()
            dt_eff = 0.0 if paused else dt

            content = SCENES[scene_idx].step(vw, vh, t, dt_eff, speed, theme)
            if show_help:
                content = overlay_help(content, vw, vh, theme)

            content = pad_lines(content, left)

            frames += 1
            if now - fps_t0 >= 0.5:
                fps = frames / (now - fps_t0)
                fps_t0 = now
                frames = 0

            # Note: for scenes 10/11, we show shortcut hints (J/K) because multi-digit keys aren't a thing.
            swap_hint = "1..9" if scene_idx < 9 else ("J=10 K=11")
            info = theme.ui_subtle(f"Teclas: {swap_hint} | T tema | C cor | +/- vel | P pausa | H ajuda | Q/ESC menu")

            title = theme.ui_title(f"{SCENES[scene_idx].key}) {SCENES[scene_idx].name}")
            indicators = (
                f"{theme.ui_ok('FPS')} {fps:5.1f}  | "
                f"{theme.ui_ok('Tela')} {cols}x{lines}  | "
                f"{theme.ui_ok('Viewport')} {vw}x{vh}  | "
                f"{theme.ui_ok('Tema')} {theme.name}  | "
                f"{theme.ui_ok('Cor')} {'ON' if theme.color_enabled and theme.name!='Mono' else 'OFF'}  | "
                f"{theme.ui_ok('Vel')} {speed:0.1f}  | "
                f"{theme.ui_ok('Pausa')} {'SIM' if paused else 'NÃO'}"
            )

            header_lines = [title, info, indicators]
            home()
            _w(frame_to_screen(header_lines, content, top))
            flush()

            target = 1/45
            spare = target - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)

    except KeyboardInterrupt:
        return scene_idx
    finally:
        show_cursor()
        _w(reset() + '\n')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    theme = Theme()
    scene_idx = 0

    while True:
        menu(theme)
        try:
            choice = input('Escolha (1-11), T/C/H ou 0: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice in KEY_TO_SCENE:
            scene_idx = KEY_TO_SCENE[choice]
            scene_idx = run_scene(scene_idx, theme)
        elif choice == 't':
            theme.next_theme()
        elif choice == 'c':
            theme.toggle_color()
        elif choice == 'h':
            scene_idx = run_scene(scene_idx, theme)
        elif choice == '0':
            break
        else:
            time.sleep(0.25)

    clear(); show_cursor(); _w(reset() + 'Até mais! 👋\n')


if __name__ == '__main__':
    main()
