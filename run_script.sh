#!/bin/bash
clear
eval "$($(which conda) 'shell.bash' 'hook')"
echo "Activating conda environment"
# ########################   Ubuntu-CPU   ########################
conda activate bic_env

echo "Running BIC implementation ... please wait:)"
python main.py 

conda deactivate 
