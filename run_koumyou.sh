#!/bin/bash
# Structural color simulator launcher (WSL / s4env venv, S4 + rcwa_mh backends both available).
# Usage: bash ~/run_koumyou.sh   (or: wsl bash ~/run_koumyou.sh  from Windows)
set -e
source ~/s4env/bin/activate
cd "/mnt/c/Users/bozu1/OneDrive/デスクトップ/lab/koumyou_sim"
streamlit run R8_koumyou_v3.py
