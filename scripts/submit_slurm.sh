#!/bin/bash

declare -A datasets

# datasets['Neurolib']="336"
# datasets['Solar']="336"
datasets['Atec']="336"
# datasets['Meter']="336"

# datasets['Temp']="168"
# datasets['Wind']="168"
# datasets['Traffic-CA']="168"
# datasets['Traffic-GLA']="168"
# datasets['Traffic-GBA']="168"

# datasets['Air Quality']="28"

# datasets['SIRS']="7"
# datasets['SP500']="7"
# datasets['M5']="7"
# datasets['Measles']="7"
# datasets['Wiki-20k']="7"
# datasets['Mobility']="7"

model=UCast
for data in "${!datasets[@]}"; do
    sbatch scripts/slurm.sh ${model} "${data}"
done





