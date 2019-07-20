#!/bin/bash/

START=1
R=1000
S=0.001

while read T H
do 
     Rscript commandOD.R $S $H $T $R $START

     for ((i=$START; i<=$R; i++ ))
     do
	     /home/ulas/slim/slim /mnt/NEOGENE1/projects/deepLearn_selection_2018/scripts/simulations/SLiM_scripts/OD_"$T"_"$H"_"$i"
     done

done << EOF
40 6
35 8
30 10
25 14
20 20
EOF

