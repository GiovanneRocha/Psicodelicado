@echo off
chcp 65001 >nul
REM MAIS colunas = MAIS detalhes
mode con: cols=200 lines=60
python impressionador_ultra_hypervision.py
