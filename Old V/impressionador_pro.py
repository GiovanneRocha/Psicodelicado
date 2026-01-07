#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMPRESSIONADOR PRO (Windows-friendly)

🎯 Essência original + melhorias:
- Donut 3D girando (principal)
- Chuva Matrix (principal)
- Fractal de Mandelbrot (principal)

✅ Melhorias pedidas:
- Opção COM cor / SEM cor (toggle)
- Tudo sempre CENTRALIZADO no terminal
- Evita "torto" / "fora da tela" (viewports com margem + clamp)
- Resize-aware (se redimensionar o terminal, recalcula)

Controles (em qualquer animação):
- C -> alterna cor (ON/OFF)
- R -> reset (quando aplicável)
- Q / ESC -> volta para o menu
- Ctrl+C -> sair

Dica:
- Quanto maior o terminal (mais COLS), mais detalhe em todas as cenas.
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
    s = shutil.get_terminal_size((120, 40))
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
# Render helpers (centralização)
# -----------------------------

def viewport(cols: int, lines: int, max_w: int, max_h: int, header: int = 2, margin: int = 1):
    """Calcula uma área de renderização centralizada, evitando wrap.

    - Respeita header (linhas usadas por título/info).
    - Usa margem para evitar quebras (wrap) na última coluna.
    """
    usable_h = max(10, lines - header)
    # evitar a última coluna (wrap fácil)
    usable_w = max(20, cols - 1)

    vw = min(usable_w - 2*margin, max_w)
    vh = min(usable_h - 1, max_h)

    vw = max(20, vw)
    vh = max(10, vh)

    left = max(0, (usable_w - vw) // 2)
    top = max(0, (usable_h - vh) // 2)

    return vw, vh, left, top


def pad_lines(lines_list, left_pad: int):
    if left_pad <= 0:
        return lines_list
    pad = ' ' * left_pad
    return [pad + ln for ln in lines_list]


def frame_to_screen(header_lines: list[str], content_lines: list[str], top_pad: int):
    # Monta frame completo com top padding (abaixo do header)
    out = []
    out.extend(header_lines)
    out.extend([''] * top_pad)
    out.extend(content_lines)
    return '\n'.join(out)

# -----------------------------
# Modo cor ON/OFF (menos psicodélico)
# -----------------------------

class ColorMode:
    def __init__(self):
        self.enabled = True

    def toggle(self):
        self.enabled = not self.enabled

    def green(self, s: str) -> str:
        if not self.enabled:
            return s
        return fg256(46) + s + reset()

    def title(self, s: str) -> str:
        if not self.enabled:
            return bold() + s + reset()
        return bold() + fg256(81) + s + reset()

    def subtle(self, s: str) -> str:
        if not self.enabled:
            return s
        return dim() + fg256(245) + s + reset()

    def mandel_color(self, t: float) -> str:
        """Paleta suave (azul -> ciano -> amarelo)."""
        if not self.enabled:
            return ''
        # t in [0,1]
        # segment 0..0.6 blue->cyan, 0.6..1 cyan->yellow
        if t < 0.6:
            k = t/0.6
            r = int(0 + 40*k)
            g = int(80 + 140*k)
            b = int(160 + 80*k)
        else:
            k = (t-0.6)/0.4
            r = int(40 + 200*k)
            g = int(220 + 20*k)
            b = int(240 - 180*k)
        return fg_rgb(r, g, b)

    def donut_color(self, t: float) -> str:
        """Cor suave pro donut (ciano/azul), sem arco-íris."""
        if not self.enabled:
            return ''
        # t ~ luminância 0..1
        r = int(20 + 30*t)
        g = int(120 + 100*t)
        b = int(170 + 80*t)
        return fg_rgb(r, g, b)


# -----------------------------
# 1) Donut 3D (ASCII) centralizado
# -----------------------------

def donut_3d(colors: ColorMode):
    clear(); hide_cursor()

    chars = '.,-~:;=!*#$@'
    A = 0.0
    B = 0.0

    # limites para evitar "torto" / wrap
    MAX_W = 120
    MAX_H = 40

    # keep running
    last_cols, last_lines = term_size()

    try:
        while True:
            cols, lines = term_size()
            if cols != last_cols or lines != last_lines:
                clear()
                last_cols, last_lines = cols, lines

            vw, vh, left, top = viewport(cols, lines, MAX_W, MAX_H, header=2, margin=1)

            # buffers
            zbuffer = [0.0] * (vw * vh)
            output = [' '] * (vw * vh)
            lum = [0.0] * (vw * vh)

            cosA, sinA = math.cos(A), math.sin(A)
            cosB, sinB = math.cos(B), math.sin(B)

            # donut params
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

            # Build lines
            content = []
            for y in range(vh):
                start = y*vw
                row_chars = output[start:start+vw]
                if colors.enabled:
                    # colorize by luminance, but keep subtle
                    row = []
                    for i, ch in enumerate(row_chars):
                        if ch == ' ':
                            row.append(' ')
                        else:
                            t = lum[start+i]
                            row.append(colors.donut_color(t) + ch + reset())
                    content.append(''.join(row))
                else:
                    content.append(''.join(row_chars))

            content = pad_lines(content, left)

            title = colors.title('Donut 3D girando')
            info = colors.subtle('Teclas: C cor | Q/ESC menu | Ctrl+C sair')
            header_lines = [title, info]

            home()
            _w(frame_to_screen(header_lines, content, top))
            flush()

            # input
            k = read_key()
            if k:
                kk = k.lower()
                if kk == 'c':
                    colors.toggle(); clear()
                elif kk == 'q' or k == '\x1b':
                    break

            A += 0.04
            B += 0.02
            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        _w(reset() + '\n')


# -----------------------------
# 2) Matrix Rain centralizado
# -----------------------------

def matrix_rain(colors: ColorMode):
    clear(); hide_cursor()

    MAX_W = 180
    MAX_H = 55

    cols, lines = term_size()
    vw, vh, left, top = viewport(cols, lines, MAX_W, MAX_H, header=2, margin=1)

    # init columns for this viewport width
    drops = [random.randint(-vh, vh) for _ in range(vw)]
    speeds = [random.uniform(0.7, 2.6) for _ in range(vw)]
    last = [time.time()] * vw

    charset = "アイウエオカキクケコサシスセソタチツテトナニヌネノ" \
              "ハヒフヘホマミムメモヤユヨラリルレロワヲン" \
              "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    last_cols, last_lines = cols, lines

    try:
        while True:
            cols, lines = term_size()
            if cols != last_cols or lines != last_lines:
                clear()
                vw, vh, left, top = viewport(cols, lines, MAX_W, MAX_H, header=2, margin=1)
                drops = [random.randint(-vh, vh) for _ in range(vw)]
                speeds = [random.uniform(0.7, 2.6) for _ in range(vw)]
                last = [time.time()] * vw
                last_cols, last_lines = cols, lines

            now = time.time()

            # canvas
            canvas = [list(' ' * vw) for _ in range(vh)]

            for x in range(vw):
                # update column
                if now - last[x] >= (0.02 / speeds[x]):
                    last[x] = now
                    drops[x] += 1
                    if drops[x] > vh + random.randint(0, vh//2 + 1):
                        drops[x] = random.randint(-vh, 0)
                        speeds[x] = random.uniform(0.7, 2.6)

                y = drops[x]
                if 0 <= y < vh:
                    canvas[y][x] = random.choice(charset)

                # trail
                for t in range(1, 12):
                    yy = y - t
                    if 0 <= yy < vh and random.random() < 0.65:
                        canvas[yy][x] = random.choice(charset)

            # Convert to lines (subtle green or monochrome)
            content = []
            for y in range(vh):
                row = canvas[y]
                if colors.enabled:
                    # green only (less psychedelic)
                    line = []
                    for ch in row:
                        if ch == ' ':
                            line.append(' ')
                        else:
                            # small highlight chance
                            r = random.random()
                            if r < 0.03:
                                line.append(fg256(120) + ch + reset())
                            elif r < 0.08:
                                line.append(fg256(82) + ch + reset())
                            else:
                                line.append(fg256(46) + ch + reset())
                    content.append(''.join(line))
                else:
                    content.append(''.join(row))

            content = pad_lines(content, left)

            title = colors.title('Chuva Matrix')
            info = colors.subtle('Teclas: C cor | Q/ESC menu | Ctrl+C sair')
            header_lines = [title, info]

            home()
            _w(frame_to_screen(header_lines, content, top))
            flush()

            k = read_key()
            if k:
                kk = k.lower()
                if kk == 'c':
                    colors.toggle(); clear()
                elif kk == 'q' or k == '\x1b':
                    break

            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        _w(reset() + '\n')


# -----------------------------
# 3) Mandelbrot centralizado (cor suave / mono)
# -----------------------------

def mandelbrot(colors: ColorMode):
    clear(); hide_cursor()

    MAX_W = 160
    MAX_H = 50

    # Regiões e controle
    center_x, center_y = -0.5, 0.0
    base_scale = 3.0
    phase = 0.0

    ramp = " .:-=+*#%@"  # mono ramp

    last_cols, last_lines = term_size()

    try:
        while True:
            cols, lines = term_size()
            if cols != last_cols or lines != last_lines:
                clear()
                last_cols, last_lines = cols, lines

            vw, vh, left, top = viewport(cols, lines, MAX_W, MAX_H, header=2, margin=1)

            # zoom oscilante (não tão agressivo)
            phase += 0.055
            zoom = 0.35 + 0.65*(0.5 + 0.5*math.sin(phase*0.35))
            scale = (0.22 + (base_scale - 0.22) * zoom)

            max_iter = int(70 + (1.0 / max(scale, 0.2)) * 55)

            # Aspect fix para não "tortar"
            aspect = (vh / vw) * 1.65

            content = []
            for py in range(vh):
                y0 = center_y + (py / vh - 0.5) * scale * aspect
                row_parts = []
                for px in range(vw):
                    x0 = center_x + (px / vw - 0.5) * scale

                    x = 0.0
                    y = 0.0
                    it = 0
                    while x*x + y*y <= 4.0 and it < max_iter:
                        x, y = x*x - y*y + x0, 2*x*y + y0
                        it += 1

                    if it == max_iter:
                        # inside set
                        row_parts.append(' ')
                    else:
                        t = it / max_iter
                        # character density
                        ch = ramp[int(t * (len(ramp)-1))]

                        if colors.enabled:
                            # cor suave (não psicodélica)
                            row_parts.append(colors.mandel_color(t) + ch + reset())
                        else:
                            row_parts.append(ch)

                content.append(''.join(row_parts))

            content = pad_lines(content, left)

            title = colors.title('Fractal Mandelbrot')
            info = colors.subtle('Teclas: C cor | Q/ESC menu | Ctrl+C sair')
            header_lines = [title, info]

            home()
            _w(frame_to_screen(header_lines, content, top))
            flush()

            k = read_key()
            if k:
                kk = k.lower()
                if kk == 'c':
                    colors.toggle(); clear()
                elif kk == 'q' or k == '\x1b':
                    break

            time.sleep(0.06)

    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        _w(reset() + '\n')


# -----------------------------
# Menu
# -----------------------------

def menu(colors: ColorMode):
    clear(); show_cursor()
    cols, lines = term_size()

    title = 'PYTHON IMPRESSIONADOR PRO'
    subtitle = 'Donut 3D • Matrix Rain • Mandelbrot  |  C = cor ON/OFF em tempo real'
    status = f"Cor: {'ON' if colors.enabled else 'OFF'}"

    box = [
        f"{bold()}{title}{reset()}",
        subtitle,
        '',
        '1) Donut 3D girando',
        '2) Chuva Matrix',
        '3) Fractal de Mandelbrot',
        '',
        'C) Alternar cor (menu)',
        '0) Sair',
        '',
        f"{status}",
        '',
        'Dica: maximize o terminal (mais colunas = mais detalhe).',
    ]

    # centraliza aproximadamente
    w = max(len(l) for l in box)
    left = max(0, (cols - w)//2)
    top = max(0, (lines - len(box))//2)

    _w('\n'*top)
    for l in box:
        _w(' '*left + l + '\n')

    flush()


def main():
    # UTF-8 ajuda caracteres do Matrix
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    colors = ColorMode()

    while True:
        menu(colors)
        try:
            choice = input('Escolha (0-3) ou C: ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == '1':
            donut_3d(colors)
        elif choice == '2':
            matrix_rain(colors)
        elif choice == '3':
            mandelbrot(colors)
        elif choice == 'c':
            colors.toggle()
        elif choice == '0':
            break
        else:
            time.sleep(0.4)

    clear()
    show_cursor()
    _w(reset() + 'Até mais! 👋\n')


if __name__ == '__main__':
    main()
