#/bin/bash/

START=1
R=1000
H=0.5

while read T S
do 
     Rscript commandIS.R $S $H $T $R $START

     for ((i=$START; i<=$R; i++))
     do
	     /home/ulas/slim/slim /mnt/NEOGENE1/projects/deepLearn_selection_2018/scripts/simulations/SLiM_scripts/IS_"$T"_"$S"_"$i"
     done

done << EOF
40 0.0064
35 0.0077
30 0.011
25 0.015
20 0.019
EOF

