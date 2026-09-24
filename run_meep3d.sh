#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mp
cd "/mnt/c/Users/bozu1/OneDrive/デスクトップ/lab/koumyou_sim"
streamlit run meep3d_app.py
