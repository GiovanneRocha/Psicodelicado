#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ABSURDO AV: Raymarching (pseudo-3D) + Áudio Reativo (microfone)

✅ Terminal: Windows Terminal (ideal), mas roda também em Linux/macOS com terminais ANSI.
✅ TrueColor (24-bit) + técnica half-block '▀' (2 pixels por caractere).
✅ Raymarching com sombras suaves (soft shadows), AO simples e névoa.
✅ Áudio reativo: captura do microfone e detecta "batida" (energia de graves).

Instalação (recomendado):
  pip install -r requirements.txt

Controles:
  1..5  -> troca de cena
  ESPAÇO -> pausa
  R -> reset da cena
  D -> alterna qualidade (útil em terminais enormes)
  +/- -> velocidade
  M -> mutar áudio (usa sinal simulado)
  Q ou ESC -> sair

Dica:
  Quanto maior o terminal (mais COLS), mais detalhe. Raymarch é pesado: use 'D' se cair FPS.
"""

from __future__ import annotations

import os
import sys
import time
import math
import random
import shutil
from dataclasses import dataclass

# -----------------------------
# ANSI / VT
# -----------------------------
IS_WINDOWS = (os.name == "nt")
CSI = "\x1b["

if IS_WINDOWS:
    # Enable VT processing on Windows
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(h, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
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
    _w(CSI + "2J" + CSI + "H")

def home() -> None:
    _w(CSI + "H")

def hide_cursor() -> None:
    _w(CSI + "?25l")

def show_cursor() -> None:
    _w(CSI + "?25h")

def reset() -> str:
    return CSI + "0m"

def bold(s: str) -> str:
    return CSI + "1m" + s + CSI + "22m"

def fg(r: int, g: int, b: int) -> str:
    return f"{CSI}38;2;{r};{g};{b}m"

def bg(r: int, g: int, b: int) -> str:
    return f"{CSI}48;2;{r};{g};{b}m"

def term_size() -> tuple[int, int]:
    s = shutil.get_terminal_size((140, 45))
    return s.columns, s.lines

# -----------------------------
# Non-blocking keyboard
# -----------------------------
KEY_NONE = ""

if IS_WINDOWS:
    def read_key() -> str:
        try:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0") and msvcrt.kbhit():
                    return ch + msvcrt.getwch()
                return ch
        except Exception:
            return KEY_NONE
        return KEY_NONE
else:
    # POSIX: raw mode + select
    import termios
    import tty
    import select

    _orig_attrs = None

    def _enter_raw() -> None:
        global _orig_attrs
        fd = sys.stdin.fileno()
        _orig_attrs = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    def _exit_raw() -> None:
        global _orig_attrs
        if _orig_attrs is None:
            return
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _orig_attrs)
        _orig_attrs = None

    def read_key() -> str:
        try:
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if dr:
                return sys.stdin.read(1)
        except Exception:
            return KEY_NONE
        return KEY_NONE

# -----------------------------
# FrameBuffer: '▀' packing
# -----------------------------

@dataclass
class FrameBuffer:
    cols: int
    lines: int
    header_lines: int = 3

    def __post_init__(self):
        self.w = max(30, self.cols)
        self.h_chars = max(12, self.lines - self.header_lines)
        self.wp = self.w
        self.hp = self.h_chars * 2

    def rebuild(self, cols: int, lines: int) -> None:
        self.cols = cols
        self.lines = lines
        self.__post_init__()

    def present(self, pixels, title: str, info: str, audioline: str) -> None:
        # pixels: list[list[(r,g,b)]], size hp x wp
        home()
        _w(reset())
        _w(title + "\n")
        _w(info + "\n")
        _w(audioline + "\n")

        out_lines = []
        for y in range(self.h_chars):
            y2 = y * 2
            top = pixels[y2]
            bot = pixels[y2 + 1]
            parts = []
            last = None
            for x in range(self.wp):
                tr,tg,tb = top[x]
                br,bg_,bb = bot[x]
                key = (tr,tg,tb, br,bg_,bb)
                if key != last:
                    parts.append(fg(tr,tg,tb) + bg(br,bg_,bb))
                    last = key
                parts.append("▀")
            parts.append(reset())
            out_lines.append("".join(parts))
        _w("\n".join(out_lines))

# -----------------------------
# Math / Color utils
# -----------------------------

def clampi(x: float, lo: int = 0, hi: int = 255) -> int:
    if x < lo: return lo
    if x > hi: return hi
    return int(x)

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo: return lo
    if x > hi: return hi
    return x

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def smoothstep(a: float, b: float, x: float) -> float:
    t = clamp((x - a) / (b - a + 1e-9))
    return t * t * (3 - 2 * t)

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

# -----------------------------
# Audio Analyzer (microfone)
# -----------------------------

class AudioAnalyzer:
    """Captura áudio do microfone e estima nível + 'batida' (graves).

    Depende de numpy e sounddevice. Se não disponíveis, cai em modo simulado.
    """

    def __init__(self, sample_rate: int = 44100, block: int = 1024):
        self.sample_rate = sample_rate
        self.block = block
        self.enabled = True
        self.simulated = False

        self.level = 0.0
        self.bass = 0.0
        self.beat = 0.0

        self._bass_avg = 0.0
        self._last_beat_t = 0.0

        self._np = None
        self._sd = None
        self._stream = None
        self._buffer = None
        self._buf_i = 0
        self._buf_n = sample_rate  # 1 segundo

        try:
            import numpy as np
            import sounddevice as sd
            self._np = np
            self._sd = sd
            self._buffer = np.zeros(self._buf_n, dtype=np.float32)

            def callback(indata, frames, time_info, status):
                if not self.enabled:
                    return
                x = indata[:, 0].astype(np.float32)
                n = len(x)
                i = self._buf_i
                end = i + n
                if end < self._buf_n:
                    self._buffer[i:end] = x
                else:
                    k = self._buf_n - i
                    self._buffer[i:] = x[:k]
                    self._buffer[:end - self._buf_n] = x[k:]
                self._buf_i = end % self._buf_n

            self._stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block,
                callback=callback,
                dtype='float32'
            )
            self._stream.start()

        except Exception:
            self.simulated = True

    def toggle_mute(self):
        self.enabled = not self.enabled

    def close(self):
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass

    def update(self, t: float):
        # simulated signal (beats + noise)
        if self.simulated or not self.enabled:
            # generate pseudo beat
            beat_wave = max(0.0, math.sin(t*2.2) + 0.2*math.sin(t*6.4))
            beat_wave = beat_wave**3
            self.bass = 0.25 + 0.75*beat_wave
            self.level = 0.2 + 0.6*beat_wave
            # beat pulse
            self.beat = smoothstep(0.65, 0.95, beat_wave)
            return

        np = self._np
        # pull most recent window
        # use 2048 samples from ring buffer
        win = 2048
        if self._buffer is None:
            return
        i = self._buf_i
        if i - win >= 0:
            x = self._buffer[i-win:i].copy()
        else:
            x = np.concatenate([self._buffer[i-win:], self._buffer[:i]]).copy()

        # remove DC
        x -= x.mean()
        # level (RMS)
        rms = float(np.sqrt(np.mean(x*x)) + 1e-9)
        self.level = clamp(rms * 6.0)

        # FFT for bass energy
        w = np.hanning(len(x)).astype(np.float32)
        xf = np.fft.rfft(x * w)
        mag = np.abs(xf)
        freqs = np.fft.rfftfreq(len(x), 1.0/self.sample_rate)

        # bass band 35..180 Hz
        mask = (freqs >= 35.0) & (freqs <= 180.0)
        bass_energy = float(mag[mask].mean() + 1e-9)
        bass = clamp((math.log10(bass_energy) + 2.5) / 2.5)  # normalize by log
        self.bass = bass

        # beat detection via adaptive average
        # Exponential moving average of bass
        self._bass_avg = lerp(self._bass_avg, bass, 0.08)
        threshold = max(0.12, self._bass_avg * 1.55)
        is_beat = (bass > threshold) and (t - self._last_beat_t > 0.12)
        if is_beat:
            self._last_beat_t = t
            self.beat = 1.0
        else:
            # decay pulse
            self.beat = max(0.0, self.beat - 0.07)

# -----------------------------
# Scenes base
# -----------------------------

class Scene:
    name = "Scene"
    def reset(self):
        pass
    def update(self, fb: FrameBuffer, t: float, dt: float, speed: float, audio: AudioAnalyzer, quality: float):
        raise NotImplementedError

# -----------------------------
# Raymarching scene (pseudo-3D)
# -----------------------------

class Raymarch(Scene):
    name = "RAYMARCH: soft shadows + AO (audio reactive)"

    def reset(self):
        pass

    # Vector helpers (tuples)
    @staticmethod
    def add(a,b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    @staticmethod
    def sub(a,b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    @staticmethod
    def mul(a,s): return (a[0]*s, a[1]*s, a[2]*s)
    @staticmethod
    def dot(a,b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    @staticmethod
    def length(a): return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
    @staticmethod
    def norm(a):
        l = Raymarch.length(a) + 1e-9
        return (a[0]/l, a[1]/l, a[2]/l)

    @staticmethod
    def rot_y(p, a):
        c, s = math.cos(a), math.sin(a)
        return (c*p[0] + s*p[2], p[1], -s*p[0] + c*p[2])

    @staticmethod
    def rot_x(p, a):
        c, s = math.cos(a), math.sin(a)
        return (p[0], c*p[1] - s*p[2], s*p[1] + c*p[2])

    @staticmethod
    def sdf_sphere(p, r):
        return Raymarch.length(p) - r

    @staticmethod
    def sdf_box(p, b):
        # signed distance to axis-aligned box of half-size b
        qx = abs(p[0]) - b[0]
        qy = abs(p[1]) - b[1]
        qz = abs(p[2]) - b[2]
        ax = max(qx, 0.0)
        ay = max(qy, 0.0)
        az = max(qz, 0.0)
        outside = math.sqrt(ax*ax + ay*ay + az*az)
        inside = min(max(qx, max(qy, qz)), 0.0)
        return outside + inside

    @staticmethod
    def sdf_plane(p, n=(0.0,1.0,0.0), h=0.0):
        return Raymarch.dot(p, n) + h

    def map(self, p, t, audio: AudioAnalyzer):
        # Scene: plane + sphere + rotating box
        # Audio drives a subtle pulsation and twist
        beat = audio.beat
        bass = audio.bass

        # ground
        d_plane = self.sdf_plane(p, (0,1,0), 1.15)
        mat_plane = 0

        # moving sphere (audio radius)
        sp_center = (0.75*math.sin(t*0.6), -0.15 + 0.25*math.sin(t*0.4), 0.6*math.cos(t*0.5))
        pp = self.sub(p, sp_center)
        r = 0.55 + 0.15*bass + 0.07*beat
        d_sphere = self.sdf_sphere(pp, r)
        mat_sphere = 1

        # rotating box
        bp = self.sub(p, (-0.9, -0.1, 0.2))
        bp = self.rot_y(bp, t*0.8 + 0.6*beat)
        bp = self.rot_x(bp, t*0.6)
        # twist by bass
        tw = 0.6*bass
        bp = (bp[0]*math.cos(bp[1]*tw) - bp[2]*math.sin(bp[1]*tw), bp[1], bp[0]*math.sin(bp[1]*tw) + bp[2]*math.cos(bp[1]*tw))
        d_box = self.sdf_box(bp, (0.45, 0.45, 0.45))
        mat_box = 2

        # union of objects
        d_obj = d_sphere
        mat = mat_sphere
        if d_box < d_obj:
            d_obj = d_box
            mat = mat_box

        # union with plane
        if d_plane < d_obj:
            return d_plane, mat_plane
        return d_obj, mat

    def normal(self, p, t, audio: AudioAnalyzer):
        e = 0.0018
        dx = self.map((p[0]+e, p[1], p[2]), t, audio)[0] - self.map((p[0]-e, p[1], p[2]), t, audio)[0]
        dy = self.map((p[0], p[1]+e, p[2]), t, audio)[0] - self.map((p[0], p[1]-e, p[2]), t, audio)[0]
        dz = self.map((p[0], p[1], p[2]+e), t, audio)[0] - self.map((p[0], p[1], p[2]-e), t, audio)[0]
        return self.norm((dx, dy, dz))

    def soft_shadow(self, ro, rd, t, audio: AudioAnalyzer, mint=0.02, maxt=6.0, k=18.0):
        # IQ-style soft shadow approximation
        res = 1.0
        tt = mint
        for _ in range(32):
            h, _ = self.map(self.add(ro, self.mul(rd, tt)), t, audio)
            if h < 0.001:
                return 0.0
            res = min(res, k * h / tt)
            tt += clamp(h, 0.02, 0.35)
            if tt > maxt:
                break
        return clamp(res)

    def ao(self, p, n, t, audio: AudioAnalyzer):
        # cheap ambient occlusion
        occ = 0.0
        sca = 1.0
        for i in range(1, 6):
            h = 0.05 * i
            d, _ = self.map(self.add(p, self.mul(n, h)), t, audio)
            occ += (h - d) * sca
            sca *= 0.55
        return clamp(1.0 - occ)

    def shade(self, p, n, mat, ro, t, audio: AudioAnalyzer):
        # Lights
        beat = audio.beat
        bass = audio.bass
        light_dir = self.norm((0.55, 0.9, -0.35))
        view = self.norm(self.sub(ro, p))
        halfv = self.norm(self.add(light_dir, view))

        # Base colors by material
        if mat == 0:
            # plane: checker
            chk = (int(math.floor(p[0]*2.2) + math.floor(p[2]*2.2)) & 1)
            base = (0.10, 0.12, 0.16) if chk else (0.05, 0.06, 0.08)
            base = tuple(b*(1.0 + 0.8*beat) for b in base)
            rough = 0.9
        elif mat == 1:
            # sphere: neon cyan/pink shift with bass
            h = (0.52 + 0.18*math.sin(t*0.35) + 0.25*bass) % 1.0
            r,g,b = hsv_to_rgb(h, 0.95, 1.0)
            base = (r/255, g/255, b/255)
            rough = 0.35
        else:
            # box: warm gold
            h = (0.10 + 0.06*math.sin(t*0.7 + bass*2.0)) % 1.0
            r,g,b = hsv_to_rgb(h, 0.85, 1.0)
            base = (r/255, g/255, b/255)
            rough = 0.22

        # Diffuse
        diff = max(0.0, self.dot(n, light_dir))

        # Soft shadow
        sh = self.soft_shadow(self.add(p, self.mul(n, 0.02)), light_dir, t, audio)

        # Specular
        spec = max(0.0, self.dot(n, halfv))
        spec = spec ** (lerp(20, 80, 1.0-rough))

        # Ambient + AO
        ao = self.ao(p, n, t, audio)
        amb = 0.10 + 0.18*ao

        # Rim light
        rim = (1.0 - max(0.0, self.dot(n, view)))
        rim = rim**2.2

        # Combine
        # Beat makes light pulse
        pulse = 1.0 + 0.55*beat
        col = [0.0, 0.0, 0.0]
        for i in range(3):
            col[i] = base[i] * (amb + diff*1.15*sh*pulse) + spec*0.85*sh + rim*0.18

        # Fog
        dist = self.length(self.sub(p, ro))
        fog = smoothstep(2.0, 7.0, dist)
        fog_col = (0.02, 0.04, 0.09)
        col = [lerp(col[i], fog_col[i], fog) for i in range(3)]

        return (clamp(col[0]), clamp(col[1]), clamp(col[2]))

    def update(self, fb: FrameBuffer, t: float, dt: float, speed: float, audio: AudioAnalyzer, quality: float):
        wp, hp = fb.wp, fb.hp

        # Dynamic resolution scaling for raymarching
        # quality in {1.0, 0.75, 0.55}
        rw = max(40, int(wp * quality))
        rh = max(24, int(hp * quality))

        # pixel buffer (full size)
        pixels = [[(0,0,0)]*wp for _ in range(hp)]

        # Camera
        tt = t * 0.75 * speed
        cam_pos = (2.8*math.sin(tt*0.35), 0.25 + 0.25*math.sin(tt*0.22), 2.8*math.cos(tt*0.35))
        target = (0.0, -0.25, 0.0)

        # Camera basis
        forward = self.norm(self.sub(target, cam_pos))
        right = self.norm((forward[2], 0.0, -forward[0]))
        up = self.norm((
            right[1]*forward[2] - right[2]*forward[1],
            right[2]*forward[0] - right[0]*forward[2],
            right[0]*forward[1] - right[1]*forward[0]
        ))

        fov = 1.1
        aspect = rh / rw

        # Audio drives subtle camera shake
        shake = 0.010 * audio.beat

        # Raymarch settings
        max_steps = 70
        max_dist = 10.0
        surf_eps = 0.0025

        for j in range(rh):
            v = (j / (rh-1) - 0.5) * fov * aspect
            for i in range(rw):
                u = (i / (rw-1) - 0.5) * fov

                # Ray direction
                rd = self.add(
                    forward,
                    self.add(self.mul(right, u), self.mul(up, -v))
                )
                rd = self.norm(rd)

                ro = cam_pos
                # small audio shake
                ro = (ro[0] + (random.random()-0.5)*shake, ro[1] + (random.random()-0.5)*shake, ro[2])

                dsum = 0.0
                mat = 0
                hit = False

                for _ in range(max_steps):
                    p = self.add(ro, self.mul(rd, dsum))
                    d, mat = self.map(p, tt, audio)
                    if d < surf_eps:
                        hit = True
                        break
                    dsum += d
                    if dsum > max_dist:
                        break

                if hit:
                    p = self.add(ro, self.mul(rd, dsum))
                    n = self.normal(p, tt, audio)
                    r,g,b = self.shade(p, n, mat, ro, tt, audio)
                else:
                    # background gradient + subtle stars
                    sky = 0.5 + 0.5*rd[1]
                    sky = clamp(sky)
                    # beat tint
                    h = (0.62 + 0.08*audio.bass) % 1.0
                    cr,cg,cb = hsv_to_rgb(h, 0.55, 0.55)
                    r = lerp(0.01, cr/255, sky)
                    g = lerp(0.02, cg/255, sky)
                    b = lerp(0.05, cb/255, sky)
                    # stars
                    if random.random() < 0.0018:
                        r,g,b = 1.0, 1.0, 1.0

                # write to full-res buffer with nearest upsample
                x0 = int(i / max(1, rw-1) * (wp-1))
                y0 = int(j / max(1, rh-1) * (hp-1))

                col = (clampi(r*255), clampi(g*255), clampi(b*255))
                pixels[y0][x0] = col
                # fill small blocks for upscaling when quality < 1
                if quality < 0.99:
                    # approximate coverage
                    if x0+1 < wp:
                        pixels[y0][x0+1] = col
                    if y0+1 < hp:
                        pixels[y0+1][x0] = col
                    if x0+1 < wp and y0+1 < hp:
                        pixels[y0+1][x0+1] = col

        return pixels

# -----------------------------
# A few lighter scenes (still audio-reactive)
# -----------------------------

class Plasma(Scene):
    name = "PLASMA TrueColor (audio tint)"
    def update(self, fb, t, dt, speed, audio, quality):
        wp, hp = fb.wp, fb.hp
        pixels = [[(0,0,0)]*wp for _ in range(hp)]
        tt = t*0.7*speed
        for y in range(hp):
            ny = (y/hp - 0.5) * 2
            row = pixels[y]
            for x in range(wp):
                nx = (x/wp - 0.5) * 2
                v = 0.0
                v += math.sin(nx*3.2 + tt)
                v += math.sin(ny*4.1 - tt*1.2)
                v += math.sin((nx+ny)*3.0 + tt*0.8)
                v += math.sin(math.sqrt(nx*nx + ny*ny)*6.0 - tt*1.9)
                v = (v/4.0 + 1)/2
                h = (v + 0.15*audio.bass + 0.08*math.sin(tt*0.4)) % 1.0
                s = 0.95
                val = 0.55 + 0.35*v + 0.10*audio.beat
                row[x] = hsv_to_rgb(h, s, clamp(val))
        return pixels

class Starfield(Scene):
    name = "HYPERSPACE (audio warp)"
    def __init__(self):
        self.stars = []
        self.reset()
    def reset(self):
        self.stars = [[random.uniform(-1,1), random.uniform(-1,1), random.uniform(0.1,1.0)] for _ in range(1500)]
    def update(self, fb, t, dt, speed, audio, quality):
        wp, hp = fb.wp, fb.hp
        pixels = [[(0,0,0)]*wp for _ in range(hp)]
        cx, cy = wp/2, hp/2
        sp = (1.2 + 1.8*audio.bass) * speed
        for s in self.stars:
            s[2] -= dt * 0.55 * sp
            if s[2] <= 0.05:
                s[0] = random.uniform(-1,1)
                s[1] = random.uniform(-1,1)
                s[2] = 1.0
            x,y,z = s
            px = int(cx + (x/z) * (wp*0.35))
            py = int(cy + (y/z) * (hp*0.35))
            if 0 <= px < wp and 0 <= py < hp:
                b = int(255 * (1.0 - z) ** 0.35)
                # beat makes stars flash
                b = min(255, int(b * (1.0 + 0.65*audio.beat)))
                pixels[py][px] = (b, b, min(255, b+80))
        return pixels

class AudioBars(Scene):
    name = "AUDIO VISUALIZER (bars)"
    def update(self, fb, t, dt, speed, audio, quality):
        wp, hp = fb.wp, fb.hp
        pixels = [[(0,0,0)]*wp for _ in range(hp)]
        # background
        for y in range(hp):
            base = int(5 + 18*(y/hp))
            row = pixels[y]
            for x in range(wp):
                row[x] = (0, 0, base)
        # draw central bass pulse + moving bars
        bars = min(120, wp)
        start = (wp - bars)//2
        for i in range(bars):
            u = i/(bars-1)
            # fake spectrum derived from bass + noise (works even without full FFT)
            amp = clamp(audio.level * (1.2 - u*0.9) + audio.bass*(1.4 - u) + 0.12*math.sin(t*2.0 + i*0.15))
            height = int(amp * (hp*0.85))
            hue = (0.35 + 0.55*u + 0.18*audio.beat) % 1.0
            col = hsv_to_rgb(hue, 0.95, 1.0)
            for y in range(hp-1, hp-1-height, -1):
                if 0 <= y < hp:
                    pixels[y][start+i] = col
                    if y-1 >= 0:
                        pixels[y-1][start+i] = tuple(min(255, c) for c in col)
        # center beat circle
        cx, cy = wp//2, hp//2
        rad = int(4 + audio.bass*min(wp,hp)*0.18)
        for y in range(max(0, cy-rad), min(hp, cy+rad)):
            for x in range(max(0, cx-rad), min(wp, cx+rad)):
                dx = x - cx
                dy = y - cy
                d = math.sqrt(dx*dx + dy*dy)
                if abs(d - rad) < 1.5:
                    pixels[y][x] = (255, 255, 255)
        return pixels

class Mandelbrot(Scene):
    name = "MANDELBROT (audio palette)"
    def update(self, fb, t, dt, speed, audio, quality):
        wp, hp = fb.wp, fb.hp
        # downscale for performance
        rw = max(60, int(wp*0.75))
        rh = max(36, int(hp*0.75))
        pixels = [[(0,0,0)]*wp for _ in range(hp)]

        phase = t*0.2*speed
        cx = -0.6 + 0.14*math.cos(phase*1.3)
        cy =  0.0 + 0.10*math.sin(phase*1.1)
        zoom = 0.25 + 0.75*(0.5+0.5*math.sin(phase*0.7))
        scale = lerp(0.22, 3.0, zoom)
        max_iter = int(90 + (1.0/max(scale,0.12))*60 + audio.bass*30)
        aspect = (rh/rw) * 1.15

        for j in range(rh):
            y0 = cy + (j/rh - 0.5) * scale * aspect
            for i in range(rw):
                x0 = cx + (i/rw - 0.5) * scale
                x = 0.0
                y = 0.0
                it = 0
                while x*x + y*y <= 4.0 and it < max_iter:
                    x, y = x*x - y*y + x0, 2*x*y + y0
                    it += 1
                if it == max_iter:
                    r,g,b = 0,0,0
                else:
                    k = it/max_iter
                    h = (0.66 + 1.9*k + 0.12*audio.bass) % 1.0
                    val = 0.25 + 0.85*(k**0.35) + 0.10*audio.beat
                    r,g,b = hsv_to_rgb(h, 0.95, clamp(val))

                x0p = int(i/(rw-1)*(wp-1))
                y0p = int(j/(rh-1)*(hp-1))
                pixels[y0p][x0p] = (r,g,b)
                if x0p+1 < wp: pixels[y0p][x0p+1] = (r,g,b)
                if y0p+1 < hp: pixels[y0p+1][x0p] = (r,g,b)
                if x0p+1 < wp and y0p+1 < hp: pixels[y0p+1][x0p+1] = (r,g,b)

        return pixels

# -----------------------------
# Main
# -----------------------------

SCENES = [Raymarch(), Plasma(), Starfield(), AudioBars(), Mandelbrot()]


def audio_line(audio: AudioAnalyzer, width: int) -> str:
    # Build a compact bar meter + status
    lvl = audio.level
    bass = audio.bass
    beat = audio.beat

    bar_w = max(10, min(60, width - 55))
    fill = int(bar_w * clamp(lvl))
    bfill = int(bar_w * clamp(bass))

    # Colors
    def c(h, s, v):
        r,g,b = hsv_to_rgb(h,s,v)
        return fg(r,g,b)

    meter = "█"*fill + "·"*(bar_w-fill)
    bmeter = "█"*bfill + "·"*(bar_w-bfill)

    beat_txt = "BEAT!" if beat > 0.6 else "     "
    status = "MIC" if (not audio.simulated and audio.enabled) else ("MUT" if (not audio.simulated and not audio.enabled) else "SIM")

    return (f"Áudio[{status}]  Lvl {lvl:0.2f}  Bass {bass:0.2f}  {beat_txt}  "
            f"{c(0.32,0.85,1.0)}{meter}{reset()}  "
            f"{c(0.10,0.85,1.0)}{bmeter}{reset()}")


def main():
    # UTF-8 output
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    clear()
    hide_cursor()

    if not IS_WINDOWS:
        _enter_raw()

    cols, lines = term_size()
    fb = FrameBuffer(cols, lines, header_lines=3)

    audio = AudioAnalyzer()

    scene_idx = 0
    paused = False
    speed = 1.0
    quality_levels = [1.0, 0.75, 0.55]
    q_idx = 0

    # FPS
    last = time.perf_counter()
    fps_t0 = last
    frames = 0
    fps = 0.0

    try:
        while True:
            now = time.perf_counter()
            dt = now - last
            last = now

            # resize
            c, l = term_size()
            if c != fb.cols or l != fb.lines:
                fb.rebuild(c, l)

            # input
            k = read_key()
            if k:
                kk = k.lower()
                if kk == 'q' or k == "\x1b":
                    break
                if kk == ' ':
                    paused = not paused
                if kk == 'r':
                    try:
                        SCENES[scene_idx].reset()
                    except Exception:
                        pass
                if kk in ('+','='):
                    speed = min(3.0, speed + 0.1)
                if kk in ('-','_'):
                    speed = max(0.2, speed - 0.1)
                if kk == 'd':
                    q_idx = (q_idx + 1) % len(quality_levels)
                if kk == 'm':
                    audio.toggle_mute()
                if kk in ('1','2','3','4','5'):
                    scene_idx = int(kk) - 1

            # update audio
            audio.update(now)

            # time step
            if paused:
                dt_eff = 0.0
            else:
                dt_eff = dt

            q = quality_levels[q_idx]
            scene = SCENES[scene_idx]

            title = f"{bold('ABSURDO AV')}  | Cena {scene_idx+1}/5: {scene.name}  | {fb.cols}x{fb.lines} chars → pixels {fb.wp}x{fb.hp}"
            info = f"Teclas: 1..5 | ESP pausa | R reset | D qualidade({q:.2f}) | +/- vel({speed:.1f}) | M mute | Q/ESC sair | FPS {fps:5.1f}"
            aline = audio_line(audio, fb.cols)

            pixels = scene.update(fb, now, dt_eff, speed, audio, q)
            fb.present(pixels, title, info, aline)
            flush()

            # fps
            frames += 1
            if now - fps_t0 >= 0.5:
                fps = frames / (now - fps_t0)
                fps_t0 = now
                frames = 0

            # cap
            target = 1/40
            spare = target - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)

    except KeyboardInterrupt:
        pass
    finally:
        audio.close()
        if not IS_WINDOWS:
            _exit_raw()
        show_cursor()
        _w(reset() + "\n")


if __name__ == '__main__':
    main()
