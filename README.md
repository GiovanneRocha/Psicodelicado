# Impressionador ULTRA • HyperVision 3D 🚀🌀

O **Impressionador ULTRA** é um *showcase* de efeitos visuais no terminal, feito em Python.
Ele nasceu dos clássicos (Donut 3D, Matrix Rain e Mandelbrot) e evoluiu para um pacote completo com:

- Temas de cor (Normal → Suave → Neon → Psicodélico → Mono)
- HUD/Indicadores (FPS, resolução, viewport, etc.)
- AUTO‑SHOW com transições
- E agora um bloco de **ilusões hipnóticas integradas** (Run Tunnel / Moiré / Spiral)

> ⚠️ Se alguém sentir desconforto com padrões hipnóticos, use `X` (Safe Mode) ou tema `Mono`.

---

## Cenas
1) Donut 3D
2) Matrix Rain
3) Mandelbrot
4) Starfield
5) DOOM Fire
6) Plasma
7) Metaballs
8) Game of Life
9) Tunnel (Motion Blur)
10) Wireframe Cube
11) Terrain (Normal Shading)
12) Run Tunnel (Infinite Run) 🌀
13) Moiré Vortex (Op‑Art) 🌀
14) Spiral Trance (Hipnótico) 🌀

---

## Controles
- Troca rápida: `1..9`
- Atalhos: `J`=10 | `K`=11 | `L`=12 | `M`=13 | `N`=14
- `I` alterna variações (quando suportado)
- `T` tema | `C` cor
- `+/-` velocidade | `P` pausa
- `A` Auto‑Show | `F` HUD | `H` ajuda
- `S` screenshot | `O` gravação `.ans`
- `X` Safe Mode
- `Q/ESC` menu

---

## Como executar (Windows)
### Rápido
Duplo clique em `run.bat`.

### Manual
```powershell
chcp 65001
python impressionador_ultra_hypervision.py
```

## Mais detalhe
```bat
mode con: cols=200 lines=60
```

---

## Estrutura
- `impressionador_ultra_hypervision.py` (principal)
- `impressionador_ultra_3dppp.py` (launcher compatível)
- `run.bat`
- `README.md`
