#!/bin/bash/

START=1
R=1000
H=0.5
S=0.0


Rscript commandNE.R $S $H $R $START

for ((i=$START; i<=$R; i++))
do
    /home/ulas/slim/slim /mnt/NEOGENE1/projects/deepLearn_selection_2018/scripts/simulations/SLiM_scripts/NE_"$i"
done
