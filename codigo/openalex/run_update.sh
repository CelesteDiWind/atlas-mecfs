#!/usr/bin/env bash
# Recalcula todo el corpus OpenAlex tras añadir 2026 y limpiar (o01b). La homología (o05) corre aparte, en paralelo.
set -e
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2
step() { echo "== $1 $(date +%H:%M)"; python "$1"; }
step o03_selection.py
step o04_network.py
( OMP_NUM_THREADS=1 python o05_ph.py > ../out/o05.log 2>&1; echo "== o05 terminado $(date +%H:%M)" >> ../out/o05.log ) &
step o06_indicators.py
step o07_episodes.py
step o08_xmrv.py
step o11_layout.py
step o12_export.py
step o14_dialogue.py
step o15_dialogue_figs.py
step o16_synth_inputs.py
echo "== esperando a la homología (o05) $(date +%H:%M)"
wait
step o09_figure.py
step o13_sys.py
echo "== todo listo $(date +%H:%M)"
