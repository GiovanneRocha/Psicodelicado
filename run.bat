@echo off
chcp 65001 >nul
REM MAIS colunas = MAIS detalhes (para todas as cenas)
mode con: cols=200 lines=60
python impressionador_ultra.py
