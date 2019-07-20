#!/bin/bash/

START=1
R=1000
H=0.5
S=0.0

while read T A B
do 
    Rscript commandFD.R $S $H $T $R $START $A $B

    for (( i=$START; i<=$R; i++))
    do
	    /home/ulas/slim/slim /mnt/NEOGENE1/projects/deepLearn_selection_2018/scripts/simulations/SLiM_scripts/FD_"$T"_"$A"_"$i"
    done

done << EOF
40 1.01 0.02
35 1.015 0.03
30 1.02 0.04
25 1.025 0.05
20 1.03 0.06
EOF

