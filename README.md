# Impressionador ULTRA 3D++ 🚀

> **Um show de efeitos em ASCII/ANSI no terminal** — feito em Python, pensado para impressionar quem executa.
>
> Começou com 3 clássicos de “terminal demo” (**Donut 3D**, **Chuva Matrix** e **Mandelbrot**) e evoluiu para um pacote completo com **temas**, **indicadores**, cenas extras e **3D fake** (túnel, cubo aramado e terreno com sombreamento).

---

## ✨ Visão geral
O **Impressionador ULTRA 3D++** é um *showcase* de computação visual no terminal. Ele combina:

- **ASCII Art + ANSI Escape Codes** para animações e cores.
- **Viewport seguro** (evita “wrap” na última coluna e previne desenho “fora da tela”).
- **Temas de cor** do *Normal* ao **Psicodélico**, além de **Mono**.
- **Indicadores em tempo real**: FPS, tamanho do terminal, viewport, tema, cor, velocidade e pausa.

✅ Recomendado: **Windows Terminal** (Windows 10/11). Funciona em terminais modernos com suporte ANSI.

---

## 🧠 A história (contexto)
Este projeto nasceu com um objetivo simples: **causar o “uau” instantâneo** em qualquer terminal.

1. **Essência (clássicos)**: três demos que “vendem a ideia” imediatamente:
   - **Donut 3D girando** (ASCII 3D)
   - **Chuva Matrix** (cyber verde)
   - **Mandelbrot** (fractal / arte generativa)

2. **Refino**: ajustes para ficar *apresentável em qualquer terminal*:
   - centralização real
   - correção de cortes e “torto”
   - cor ON/OFF

3. **ULTRA**: virou um “show” com mais cenas, temas e indicadores.

4. **3D+ / 3D++**: entraram os efeitos de **pseudo-3D** e depois refinamentos:
   - **Túnel infinito com motion blur fake** (persistência temporal)
   - **Terreno em perspectiva com shading por normal** (luz/sombra no relevo)

---

## 🎬 Cenas disponíveis
### ✅ Principais (a essência)
1. **Donut 3D girando**
2. **Chuva Matrix**
3. **Fractal Mandelbrot**

### 🔥 Extras
4. **Hyperspace Starfield** (warp speed)
5. **DOOM Fire** (fogo clássico)
6. **Plasma / Nebula**
7. **Metaballs** (bolhas orgânicas)
8. **Game of Life**

### 🧊 3D Fake
9. **Túnel Infinito (Motion Blur)**
10. **Wireframe Rotating Cube**
11. **Terrain Wave (Normal Shading)**

---

## 🎨 Temas de cor (tecla `T`)
O projeto tem um ciclo de temas:

- **Normal**: cores clássicas e sóbrias
- **Suave**: menos saturação (mais confortável)
- **Neon**: vibrante sem virar arco‑íris
- **Psicodélico**: arco‑íris dinâmico total
- **Mono**: preto e branco

Além disso:
- **`C`** liga/desliga cor em tempo real (mesmo dentro de um tema).

---

## 🎛️ Controles
Dentro de qualquer cena:

- `1..9` → troca de cena instantânea
- `J` → vai para a cena **10** (Cubo)
- `K` → vai para a cena **11** (Terreno)
- `T` → troca tema
- `C` → cor ON/OFF
- `+` / `-` → velocidade
- `P` → pausa
- `R` → reset da cena atual
- `H` → overlay de ajuda
- `Q` ou `ESC` → volta ao menu
- `Ctrl + C` → sair

> Observação: como `10` e `11` são números de dois dígitos, no modo “tecla única” usamos **J/K** como atalhos rápidos.

---

## ▶️ Como executar (Windows)

### ✅ Opção 1 — Rodar pelo `run.bat` (recomendado)
1. Extraia o `.zip` do projeto
2. Dê **duplo clique** em `run.bat`

Esse arquivo geralmente:
- ativa UTF‑8 (`chcp 65001`)
- define um terminal grande (`mode con: cols=200 lines=60`)
- inicia o Python

### ✅ Opção 2 — Manual (PowerShell)
Abra a pasta no PowerShell e rode:

```powershell
chcp 65001
python impressionador_ultra_3dpp.py
```

---

## 🖥️ Dica: MAIS COLUNAS = MAIS DETALHE
O terminal é a “resolução” do show. Quanto mais colunas, mais detalhe.

Sugestões:

```bat
mode con: cols=160 lines=45
```

```bat
mode con: cols=200 lines=60
```

```bat
mode con: cols=220 lines=65
```

Se o FPS cair, reduza `cols/lines`.

---

## 🧩 Como funciona (explicação técnica resumida)

### Donut 3D
- Projeta um toro 3D em 2D e usa luminância (dot product) para sombreamento.

### Matrix Rain
- Cada coluna tem uma “gota” com velocidade própria e uma trilha aleatória.

### Mandelbrot
- Itera a equação complexa por ponto e mapeia iterações → densidade/cor.

### Túnel com Motion Blur
- Usa coordenadas polares (raio/ângulo) para “profundidade”.
- O motion blur é **persistência temporal**: mistura brilho atual com brilho anterior e decai (trilha).

### Terreno com Normal Shading
- O terreno é um **heightmap** (seno/cosseno combinados).
- Calcula normal por diferenças finitas e aplica uma luz direcional para gerar **luz/sombra**.

---

## 🧯 Troubleshooting

### Caracteres estranhos / símbolos quebrados
- Garanta UTF‑8:
  - `chcp 65001`

### Sem cor ou cores ruins
- Use **Windows Terminal**.
- Teste `C` (cor ON/OFF) e `T` (tema).

### FPS baixo
- Reduza o tamanho (`cols/lines`).
- Use tema **Suave** ou **Mono** (menos custo de ANSI).

---

## 📁 Estrutura do projeto
- `impressionador_ultra_3dpp.py` → script principal
- `run.bat` → inicialização rápida no Windows
- `README.md` → este guia

---

## 🗺️ Roadmap (ideias futuras)
- Neblina/fog no terreno
- Reflexo/specular no terreno
- Motion blur no cubo aramado
- Modo AutoShow (troca de cenas automático com transições)

---

## 📜 Licença
Uso livre para fins educacionais e demonstrações. Se publicar, é bacana citar/creditar.
