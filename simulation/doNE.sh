#!/bin/bash 

REP_FROM=1
REP_TO=10000
H=0.5
S=0.0

Rscript commandNE.R $S $H $REP_TO $REP_FROM

for ((i=$REP_FROM; i<=$REP_TO; i++))
do
	/usr/local/sw/SLiM-3.2/build/slim /mnt/NEOGENE1/projects/deepLearn_selection_2018/scripts/simulations/SLiM_scripts/NE_"$i"
done
