
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IMPRESSIONADOR.PY
Um "showcase" de efeitos no terminal, 100% Python (sem libs externas):
1) Donut 3D girando
2) Chuva Matrix
3) Fractal de Mandelbrot colorido

Saída: Ctrl+C em qualquer modo.
"""

import os
import sys
import time
import math
import random
import shutil

# -----------------------------
# Utilitários de Terminal
# -----------------------------

def supports_ansi():
    """Detecta se o terminal provavelmente suporta ANSI."""
    if os.name != "nt":
        return True
    # Windows 10+ geralmente suporta ANSI no terminal moderno.
    # Se estiver no CMD antigo, pode não funcionar perfeito.
    return True

ANSI_OK = supports_ansi()

def clear():
    if ANSI_OK:
        sys.stdout.write("\x1b[2J\x1b[H")
    else:
        os.system("cls" if os.name == "nt" else "clear")

def hide_cursor():
    if ANSI_OK:
        sys.stdout.write("\x1b[?25l")

def show_cursor():
    if ANSI_OK:
        sys.stdout.write("\x1b[?25h")

def move_home():
    if ANSI_OK:
        sys.stdout.write("\x1b[H")

def flush():
    sys.stdout.flush()

def term_size():
    s = shutil.get_terminal_size((80, 24))
    return s.columns, s.lines

def color256(n):
    """Retorna escape para cor 256 (foreground)."""
    if not ANSI_OK:
        return ""
    return f"\x1b[38;5;{n}m"

def reset():
    return "\x1b[0m" if ANSI_OK else ""

def dim():
    return "\x1b[2m" if ANSI_OK else ""

def bold():
    return "\x1b[1m" if ANSI_OK else ""


# -----------------------------
# 1) Donut 3D (ASCII)
# -----------------------------

def donut_3d():
    clear()
    hide_cursor()

    # Parâmetros visuais
    chars = ".,-~:;=!*#$@"  # densidade do "sombreamento"
    A = 0.0
    B = 0.0

    # Ajuste conforme terminal
    w, h = term_size()
    h = max(24, h)
    w = max(80, w)

    # "Tela"
    # Vamos usar uma área central
    out_w = min(w, 120)
    out_h = min(h, 40)

    # Constantes do donut
    R1 = 1.0
    R2 = 2.0
    K2 = 5.0

    # K1 depende do tamanho da tela
    K1 = out_w * K2 * 3 / (8 * (R1 + R2))

    try:
        while True:
            zbuffer = [0.0] * (out_w * out_h)
            output = [" "] * (out_w * out_h)

            # Rotação
            cosA, sinA = math.cos(A), math.sin(A)
            cosB, sinB = math.cos(B), math.sin(B)

            # Varredura angular
            theta = 0.0
            while theta < 2 * math.pi:
                costheta = math.cos(theta)
                sintheta = math.sin(theta)

                phi = 0.0
                while phi < 2 * math.pi:
                    cosphi = math.cos(phi)
                    sinphi = math.sin(phi)

                    # Círculo no tubo e projeção
                    circlex = R2 + R1 * costheta
                    circley = R1 * sintheta

                    # Coordenadas 3D rotacionadas
                    x = circlex * (cosB * cosphi + sinA * sinB * sinphi) - circley * cosA * sinB
                    y = circlex * (sinB * cosphi - sinA * cosB * sinphi) + circley * cosA * cosB
                    z = K2 + cosA * circlex * sinphi + circley * sinA
                    ooz = 1 / z

                    # Projeção em 2D
                    xp = int(out_w / 2 + K1 * ooz * x)
                    yp = int(out_h / 2 - K1 * ooz * y)

                    # Iluminação (dot product)
                    L = (cosphi * costheta * sinB
                         - cosA * costheta * sinphi
                         - sinA * sintheta
                         + cosB * (cosA * sintheta - costheta * sinA * sinphi))

                    if L > 0:
                        idx = xp + out_w * yp
                        if 0 <= xp < out_w and 0 <= yp < out_h:
                            if ooz > zbuffer[idx]:
                                zbuffer[idx] = ooz
                                luminance_index = int(L * 8)  # 0..~11
                                luminance_index = max(0, min(luminance_index, len(chars) - 1))
                                output[idx] = chars[luminance_index]

                    phi += 0.07
                theta += 0.03

            # Render
            move_home()
            title = f"{bold()}Donut 3D | Ctrl+C para sair{reset()}\n"
            sys.stdout.write(title)

            # Monta linhas
            for y in range(out_h):
                start = y * out_w
                line = "".join(output[start:start + out_w])
                sys.stdout.write(line + "\n")

            flush()

            # Atualiza ângulos
            A += 0.04
            B += 0.02

            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        sys.stdout.write(reset() + "\n")


# -----------------------------
# 2) Chuva Matrix
# -----------------------------

def matrix_rain():
    clear()
    hide_cursor()

    w, h = term_size()
    # Colunas: cada uma cai em velocidade e posição diferentes
    drops = [random.randint(0, h) for _ in range(w)]
    speeds = [random.uniform(0.5, 2.5) for _ in range(w)]
    last = [time.time()] * w

    charset = "アイウエオカキクケコサシスセソタチツテトナニヌネノ" \
              "ハヒフヘホマミムメモヤユヨラリルレロワヲン" \
              "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    try:
        while True:
            now = time.time()
            move_home()

            header = f"{bold()}{color256(46)}MATRIX RAIN{reset()}  {dim()}(Ctrl+C para sair){reset()}\n"
            sys.stdout.write(header)

            # Criamos frame usando buffer de linhas (mais rápido)
            frame_lines = [""] * (h - 1)
            # Inicializa com espaços
            canvas = [list(" " * w) for _ in range(h - 1)]

            for x in range(w):
                if now - last[x] >= (0.02 * (1.0 / speeds[x])):
                    last[x] = now
                    drops[x] += 1
                    if drops[x] > h + random.randint(0, h // 2):
                        drops[x] = random.randint(-h, 0)
                        speeds[x] = random.uniform(0.6, 2.8)

                y = drops[x]
                # Cabeça brilhante
                if 0 <= y < h - 1:
                    canvas[y][x] = random.choice(charset)
                # Trilhas mais fracas
                for t in range(1, 12):
                    yy = y - t
                    if 0 <= yy < h - 1:
                        # Quanto mais distante, mais chance de apagar
                        if random.random() < 0.65:
                            canvas[yy][x] = random.choice(charset)

            # Converte canvas em texto com cores
            for row in range(h - 1):
                line_chars = canvas[row]
                # Colorização simples: maioria verde, algumas “mais claras”
                # (evita custo de colorir cada caractere individualmente demais)
                # Vamos dar um toque: random highlights
                if ANSI_OK:
                    line = []
                    for ch in line_chars:
                        if ch == " ":
                            line.append(" ")
                        else:
                            r = random.random()
                            if r < 0.03:
                                line.append(color256(120) + ch + reset())
                            elif r < 0.08:
                                line.append(color256(82) + ch + reset())
                            else:
                                line.append(color256(46) + ch + reset())
                    frame_lines[row] = "".join(line)
                else:
                    frame_lines[row] = "".join(line_chars)

            sys.stdout.write("\n".join(frame_lines))
            flush()
            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        sys.stdout.write(reset() + "\n")


# -----------------------------
# 3) Mandelbrot Colorido
# -----------------------------

def mandelbrot():
    clear()
    hide_cursor()

    w, h = term_size()
    # Reservar uma linha para cabeçalho
    h = max(24, h) - 1
    w = max(80, w)

    # Região inicial do plano complexo (boa pra ficar bonito)
    center_x, center_y = -0.5, 0.0
    scale = 3.0  # quanto maior, mais "zoom out"
    max_iter = 80

    # Paleta 256 cores (gradiente)
    palette = [16, 17, 18, 19, 20, 21, 27, 33, 39, 45, 51, 50, 49, 48, 47, 46,
               82, 118, 154, 190, 226, 220, 214, 208, 202, 196, 160, 124, 88, 52]

    # Caracteres por densidade
    ramp = " .:-=+*#%@"

    def render_frame(cx, cy, sc, iters):
        # Ajuste de aspect ratio (caractere é mais alto que largo)
        aspect = (h / w) * 1.8

        lines = []
        for py in range(h):
            y0 = cy + (py / h - 0.5) * sc * aspect
            row = []
            for px in range(w):
                x0 = cx + (px / w - 0.5) * sc
                x, y = 0.0, 0.0
                iteration = 0
                while x*x + y*y <= 4.0 and iteration < iters:
                    x, y = x*x - y*y + x0, 2*x*y + y0
                    iteration += 1

                if iteration == iters:
                    # Dentro do conjunto: "preto"
                    row.append((0, " "))
                else:
                    # Colorido
                    t = iteration / iters
                    col = palette[int(t * (len(palette) - 1))]
                    ch = ramp[int(t * (len(ramp) - 1))]
                    row.append((col, ch))
            lines.append(row)
        return lines

    try:
        # Faz um “zoom” automático: vai ficando mais impressionante
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            # Zoom suave
            zoom = 0.92 ** (elapsed * 6)  # quanto menor, mais zoom in
            sc = max(0.15, scale * zoom)
            iters = int(max_iter + (1.0 / sc) * 5)  # mais iterações ao dar zoom

            move_home()
            header = f"{bold()}Mandelbrot | Zoom automático | Ctrl+C para sair{reset()}\n"
            sys.stdout.write(header)

            frame = render_frame(center_x, center_y, sc, iters)

            if ANSI_OK:
                for row in frame:
                    # Agrupa segmentos com mesma cor para imprimir rápido
                    current_color = None
                    chunk = []
                    for col, ch in row:
                        if col != current_color:
                            if chunk:
                                sys.stdout.write("".join(chunk))
                                chunk = []
                            current_color = col
                            if col == 0:
                                chunk.append(reset())
                            else:
                                chunk.append(color256(col))
                        chunk.append(ch)
                    if chunk:
                        sys.stdout.write("".join(chunk))
                    sys.stdout.write(reset() + "\n")
            else:
                for row in frame:
                    sys.stdout.write("".join(ch for _, ch in row) + "\n")

            flush()
            time.sleep(0.07)

    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        sys.stdout.write(reset() + "\n")


# -----------------------------
# Menu
# -----------------------------

def menu():
    clear()
    w, _ = term_size()

    banner = f"""
{bold()}╔══════════════════════════════════════════════════════════════╗
║                  PYTHON IMPRESSIONADOR 😄                      ║
╠══════════════════════════════════════════════════════════════╣
║  1) Donut 3D girando (ASCII)                                  ║
║  2) Chuva Matrix (terminal)                                   ║
║  3) Mandelbrot colorido (fractal)                             ║
║                                                              ║
║  0) Sair                                                      ║
╚══════════════════════════════════════════════════════════════╝{reset()}
"""
    # Centraliza aproximado
    lines = banner.strip("\n").splitlines()
    padded = []
    for line in lines:
        pad = max(0, (w - len(line)) // 2)
        padded.append(" " * pad + line)
    print("\n".join(padded))

    if not ANSI_OK:
        print("\nObs.: Seu terminal pode não estar exibindo cores/efeitos ANSI.\n")

def main():
    try:
        while True:
            menu()
            choice = input("Escolha uma opção (0-3): ").strip()

            if choice == "1":
                donut_3d()
            elif choice == "2":
                matrix_rain()
            elif choice == "3":
                mandelbrot()
            elif choice == "0":
                print("Até mais! 👋")
                return
            else:
                print("Opção inválida. Tente 0, 1, 2 ou 3.")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nSaindo…")

if __name__ == "__main__":
    main()
