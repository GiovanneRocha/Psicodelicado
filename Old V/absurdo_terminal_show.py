#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ABSURDO Terminal Show (Windows-friendly)

Um show de efeitos REALMENTE impressionantes no terminal, com TrueColor (24-bit),
packing por "half-block" (▀) para dobrar a resolução vertical e suporte a resize.

✅ Requisitos: Python 3.10+ (recomendado) em Windows
✅ Terminal recomendado: Windows Terminal / PowerShell / VS Code terminal

Controles:
  1..5  -> troca de cena
  Espaco -> pausa/continua
  R -> reset da cena atual
  +/- -> altera velocidade
  Q ou ESC -> sair
  Ctrl+C -> sair

Dica ABSURDA:
  Quanto maior o terminal (mais COLS), mais detalhes (todas as cenas escalam).

Autor: Copilot (M365)
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
import shutil
from dataclasses import dataclass

IS_WINDOWS = (os.name == "nt")

# =============================
# Windows: habilitar ANSI/VT
# =============================
if IS_WINDOWS:
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(h, new_mode)
    except Exception:
        pass

# =============================
# Terminal helpers
# =============================
CSI = "\x1b["

def _w(s: str) -> None:
    sys.stdout.write(s)

def flush() -> None:
    sys.stdout.flush()

def clear() -> None:
    _w(CSI + "2J" + CSI + "H")

def home() -> None:
    _w(CSI + "H")

def hide_cursor() -> None:
    _w(CSI + "?25l")

def show_cursor() -> None:
    _w(CSI + "?25h")

def reset() -> str:
    return CSI + "0m"

def fg(r: int, g: int, b: int) -> str:
    return f"{CSI}38;2;{r};{g};{b}m"

def bg(r: int, g: int, b: int) -> str:
    return f"{CSI}48;2;{r};{g};{b}m"

def term_size() -> tuple[int, int]:
    s = shutil.get_terminal_size((120, 40))
    return s.columns, s.lines

# =============================
# Non-blocking keyboard (Windows)
# =============================
KEY_NONE = ""

def read_key() -> str:
    if IS_WINDOWS:
        try:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                # Teclas especiais vêm como '\x00' ou '\xe0' seguido de outro código
                if ch in ("\x00", "\xe0") and msvcrt.kbhit():
                    ch2 = msvcrt.getwch()
                    return ch + ch2
                return ch
        except Exception:
            return KEY_NONE
        return KEY_NONE

    # Fallback simples (não-windows): não bloqueia, mas não captura teclas
    return KEY_NONE

# =============================
# Render: pixel packing via '▀'
# - Cada caractere representa 2 pixels verticais:
#   topo = foreground, baixo = background
# =============================

@dataclass
class FrameBuffer:
    cols: int
    lines: int
    header_lines: int = 2

    def __post_init__(self):
        # Área útil em caracteres
        self.w = max(20, self.cols)
        self.h_chars = max(10, self.lines - self.header_lines)
        # Resolução em pixels (2x vertical)
        self.hp = self.h_chars * 2
        self.wp = self.w

    def rebuild(self, cols: int, lines: int) -> None:
        self.cols = cols
        self.lines = lines
        self.__post_init__()

    def present(self, top_pixels, bot_pixels, title: str, info: str) -> None:
        # top_pixels e bot_pixels são listas de (r,g,b) por linha-pixel
        # Estrutura esperada: top_pixels[y][x], bot_pixels[y][x]
        home()
        _w(reset())
        _w(title + "\n")
        _w(info + "\n")

        # Renderiza em linhas de caracteres
        # Para performance, montamos em lista e depois join
        out_lines = []
        for y in range(self.h_chars):
            y2 = y * 2
            tp = top_pixels[y2]
            bp = bot_pixels[y2 + 1]
            # constrói linha: bg+fg por caractere
            parts = []
            last = None
            for x in range(self.wp):
                tr,tg,tb = tp[x]
                br,bg_,bb = bp[x]
                key = (tr,tg,tb, br,bg_,bb)
                if key != last:
                    parts.append(fg(tr,tg,tb) + bg(br,bg_,bb))
                    last = key
                parts.append("▀")
            parts.append(reset())
            out_lines.append("".join(parts))
        _w("\n".join(out_lines))


# =============================
# Color palettes / utilities
# =============================

def clampi(v: float, a: int=0, b: int=255) -> int:
    if v < a: return a
    if v > b: return b
    return int(v)

def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int,int,int]:
    # h in [0,1)
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

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

# =============================
# Scenes
# =============================

class Scene:
    name = "Scene"
    def reset(self):
        pass
    def update(self, fb: FrameBuffer, t: float, dt: float, speed: float):
        raise NotImplementedError

class Plasma(Scene):
    name = "PLASMA TrueColor (high-res)"

    def update(self, fb: FrameBuffer, t: float, dt: float, speed: float):
        wp, hp = fb.wp, fb.hp
        # Prepare 2 pixel rows arrays
        top = [[(0,0,0)]*wp for _ in range(hp)]
        bot = top  # we'll still index even/odd accordingly (same ref ok)
        # Compute color per pixel
        # Use sin/cos combinations for a fluid plasma
        tt = t * 0.6 * speed
        for y in range(hp):
            ny = (y / hp - 0.5) * 2.0
            row = top[y]
            for x in range(wp):
                nx = (x / wp - 0.5) * 2.0
                v = 0.0
                v += math.sin((nx*3.0 + tt) * 1.1)
                v += math.sin((ny*4.0 - tt) * 1.3)
                v += math.sin((nx*2.0 + ny*2.0 + tt) * 1.2)
                v += math.sin(math.sqrt(nx*nx + ny*ny) * 6.0 - tt*1.8)
                v = (v / 4.0 + 1.0) / 2.0
                h = (v + 0.15*math.sin(tt*0.7)) % 1.0
                s = 0.95
                val = 0.85
                row[x] = hsv_to_rgb(h, s, val)
        # present expects top_pixels[y2] and bot_pixels[y2+1]; we pass same list
        return top, top

class Mandelbrot(Scene):
    name = "MANDELBROT TrueColor (zoom)"
    def __init__(self):
        self.cx = -0.5
        self.cy = 0.0
        self.scale = 3.0
        self.base_scale = 3.0
        self.phase = 0.0

    def reset(self):
        self.cx, self.cy = -0.5, 0.0
        self.scale = self.base_scale
        self.phase = 0.0

    def update(self, fb: FrameBuffer, t: float, dt: float, speed: float):
        wp, hp = fb.wp, fb.hp
        # Auto-zoom loop with slight orbit
        self.phase += dt * 0.35 * speed
        orbit = 0.08
        cx = self.cx + math.cos(self.phase*0.9) * orbit
        cy = self.cy + math.sin(self.phase*1.1) * orbit * 0.6
        # Zoom oscillation
        zoom = 0.18 + 0.82 * (0.5 + 0.5*math.sin(self.phase*0.6))
        scale = lerp(0.25, self.base_scale, zoom)

        # Iterations scale with zoom
        max_iter = int(90 + (1.0 / max(scale, 0.12)) * 55)
        aspect = (hp / wp) * 1.15

        top = [[(0,0,0)]*wp for _ in range(hp)]
        for y in range(hp):
            yy = cy + (y / hp - 0.5) * scale * aspect
            row = top[y]
            for x in range(wp):
                xx = cx + (x / wp - 0.5) * scale
                zx, zy = 0.0, 0.0
                it = 0
                while zx*zx + zy*zy <= 4.0 and it < max_iter:
                    zx, zy = zx*zx - zy*zy + xx, 2.0*zx*zy + yy
                    it += 1
                if it == max_iter:
                    row[x] = (0,0,0)
                else:
                    # smooth coloring
                    mod = math.sqrt(zx*zx + zy*zy)
                    mu = it - math.log(max(1e-9, math.log(max(1e-9, mod), 2.0)), 2.0)
                    k = mu / max_iter
                    # palette by HSV
                    h = (0.66 + 2.2*k) % 1.0
                    s = 0.95
                    v = 0.25 + 0.85*(k**0.35)
                    row[x] = hsv_to_rgb(h, s, min(1.0, v))
        return top, top

class Starfield(Scene):
    name = "HYPERSPACE Starfield"
    def __init__(self):
        self.stars = []
        self.seed = 123
        self.reset()

    def reset(self):
        random.seed(self.seed)
        self.stars = []
        for _ in range(1400):
            # x,y in [-1,1], z in (0,1]
            self.stars.append([random.uniform(-1,1), random.uniform(-1,1), random.uniform(0.05, 1.0)])

    def update(self, fb: FrameBuffer, t: float, dt: float, speed: float):
        wp, hp = fb.wp, fb.hp
        top = [[(0,0,0)]*wp for _ in range(hp)]
        cx, cy = wp/2, hp/2
        sp = 1.5 * speed

        # draw background subtle gradient
        for y in range(hp):
            row = top[y]
            gy = y / hp
            base = 6 + int(14*gy)
            for x in range(wp):
                row[x] = (0, 0, base)

        # update stars
        for s in self.stars:
            s[2] -= dt * 0.55 * sp
            if s[2] <= 0.03:
                s[0] = random.uniform(-1,1)
                s[1] = random.uniform(-1,1)
                s[2] = 1.0
            x, y, z = s
            # perspective
            px = int(cx + (x / z) * (wp * 0.35))
            py = int(cy + (y / z) * (hp * 0.35))
            if 0 <= px < wp and 0 <= py < hp:
                # brightness increases as z decreases
                b = int(255 * (1.0 - z) ** 0.35)
                # color shift: bluish to white
                row = top[py]
                row[px] = (b, b, min(255, b+60))
                # streaks
                if z < 0.22:
                    for k in range(1, 4):
                        qx = px - int(x*3*k)
                        qy = py - int(y*3*k)
                        if 0 <= qx < wp and 0 <= qy < hp:
                            bb = max(0, b - 70*k)
                            top[qy][qx] = (bb, bb, min(255, bb+80))

        return top, top

class Donut3D(Scene):
    name = "DONUT 3D (color + shading)"

    def update(self, fb: FrameBuffer, t: float, dt: float, speed: float):
        # Render donut in a pixel buffer using point plotting
        wp, hp = fb.wp, fb.hp
        top = [[(0,0,0)]*wp for _ in range(hp)]

        A = t * 0.9 * speed
        B = t * 0.6 * speed
        cosA, sinA = math.cos(A), math.sin(A)
        cosB, sinB = math.cos(B), math.sin(B)

        R1, R2 = 1.0, 2.0
        K2 = 5.0
        K1 = wp * K2 * 3 / (8 * (R1 + R2))

        zbuf = [[0.0]*wp for _ in range(hp)]
        for theta_i in range(0, 628, 10):
            theta = theta_i / 100
            costheta = math.cos(theta)
            sintheta = math.sin(theta)
            for phi_i in range(0, 628, 4):
                phi = phi_i / 100
                cosphi = math.cos(phi)
                sinphi = math.sin(phi)

                circlex = R2 + R1*costheta
                circley = R1*sintheta

                x = circlex * (cosB*cosphi + sinA*sinB*sinphi) - circley*cosA*sinB
                y = circlex * (sinB*cosphi - sinA*cosB*sinphi) + circley*cosA*cosB
                z = K2 + cosA*circlex*sinphi + circley*sinA
                ooz = 1 / z

                xp = int(wp/2 + K1*ooz*x)
                yp = int(hp/2 - K1*ooz*y)

                # Lighting
                L = (cosphi*costheta*sinB
                     - cosA*costheta*sinphi
                     - sinA*sintheta
                     + cosB*(cosA*sintheta - costheta*sinA*sinphi))

                if 0 <= xp < wp and 0 <= yp < hp and L > 0:
                    if ooz > zbuf[yp][xp]:
                        zbuf[yp][xp] = ooz
                        # color from angle + luminance
                        h = (phi / (2*math.pi) + t*0.02) % 1.0
                        v = 0.2 + 0.8 * min(1.0, L)
                        s = 0.95
                        top[yp][xp] = hsv_to_rgb(h, s, v)

        # Add subtle background
        for y in range(hp):
            row = top[y]
            gy = y / hp
            for x in range(wp):
                if row[x] == (0,0,0):
                    row[x] = (0, 0, int(10 + 30*gy))
        return top, top

class Particles(Scene):
    name = "PARTICLE VORTEX (flow field)"
    def __init__(self):
        self.p = []
        self.reset()

    def reset(self):
        self.p = []
        for _ in range(2200):
            self.p.append([random.random(), random.random(), random.random()])  # x,y,hue

    def update(self, fb: FrameBuffer, t: float, dt: float, speed: float):
        wp, hp = fb.wp, fb.hp
        top = [[(0,0,0)]*wp for _ in range(hp)]

        # background
        for y in range(hp):
            row = top[y]
            base = int(8 + 12*(y/hp))
            for x in range(wp):
                row[x] = (0, 0, base)

        # flow field defined by curls of sin/cos
        cx, cy = 0.5 + 0.1*math.cos(t*0.2), 0.5 + 0.1*math.sin(t*0.17)
        sp = 0.22 * speed

        for p in self.p:
            x, y, h = p
            # vector field
            dx = x - cx
            dy = y - cy
            r2 = dx*dx + dy*dy + 1e-6
            # swirl + waves
            vx = (-dy / r2) + 0.35*math.sin((y*12 + t*0.9))
            vy = ( dx / r2) + 0.35*math.cos((x*12 - t*0.8))
            x = (x + vx*dt*sp) % 1.0
            y = (y + vy*dt*sp) % 1.0
            h = (h + dt*0.05*speed) % 1.0
            p[0], p[1], p[2] = x, y, h

            px = int(x * (wp-1))
            py = int(y * (hp-1))
            # draw point + tiny glow
            col = hsv_to_rgb(h, 0.95, 1.0)
            top[py][px] = col
            if px+1 < wp: top[py][px+1] = tuple(min(255, c//2 + 40) for c in col)
            if py+1 < hp: top[py+1][px] = tuple(min(255, c//2 + 20) for c in col)

        return top, top


SCENES = [Plasma(), Mandelbrot(), Starfield(), Donut3D(), Particles()]

# =============================
# Main Loop
# =============================

def main():
    # Improve Unicode output on Windows (helps braille/block chars)
    try:
        # Python 3.7+ allows reconfigure
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    clear()
    hide_cursor()

    cols, lines = term_size()
    fb = FrameBuffer(cols, lines, header_lines=2)

    scene_idx = 0
    paused = False
    speed = 1.0

    # FPS accounting
    last = time.perf_counter()
    fps_t0 = last
    frames = 0
    fps = 0.0

    try:
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now

            # resize aware
            c, l = term_size()
            if c != fb.cols or l != fb.lines:
                fb.rebuild(c, l)

            # input
            k = read_key()
            if k:
                kk = k.lower()
                if kk in ("q",):
                    break
                if k == "\x1b":  # ESC
                    break
                if kk == " ":
                    paused = not paused
                if kk == "r":
                    try:
                        SCENES[scene_idx].reset()
                    except Exception:
                        pass
                if kk == "+" or kk == "=":
                    speed = min(3.0, speed + 0.1)
                if kk == "-" or kk == "_":
                    speed = max(0.2, speed - 0.1)
                # number keys 1..5
                if kk in ("1","2","3","4","5"):
                    scene_idx = int(kk) - 1

            # time for scene
            t = now
            if paused:
                dt_eff = 0.0
            else:
                dt_eff = dt

            # render
            scene = SCENES[scene_idx]
            title = (f"{reset()}\x1b[1mABSURDO TERMINAL SHOW\x1b[0m  "
                     f"| Cena: \x1b[1m{scene_idx+1}\x1b[0m - {scene.name}  "
                     f"| {fb.cols}x{fb.lines} chars (pixels: {fb.wp}x{fb.hp})")
            info = (f"Teclas: 1..5 troca | ESP pausar | R reset | +/- velocidade ({speed:.1f}) | Q/ESC sair  "
                    f"| FPS: {fps:5.1f}")

            top, bot = scene.update(fb, t, dt_eff, speed)
            # 'bot' is same ref; present expects top_pixels[y2] and bot_pixels[y2+1]
            fb.present(top, bot, title, info)
            flush()

            # FPS
            frames += 1
            if now - fps_t0 >= 0.5:
                fps = frames / (now - fps_t0)
                fps_t0 = now
                frames = 0

            # frame cap
            # A cena do Mandelbrot é pesada: cap levemente menor
            target = 1/35 if scene_idx == 1 else 1/45
            spare = target - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)

    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        _w(reset() + "\n")


if __name__ == "__main__":
    main()
