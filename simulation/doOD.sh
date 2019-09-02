#!/bin/bash 

REP_FROM=1
REP_TO=10000
S=0.001

while read T H
do 
     Rscript commandOD.R $S $H $T $REP_TO $REP_FROM

     for ((i=$REP_FROM; i<=$REP_TO; i++))
     do
	     /usr/local/sw/SLiM-3.2/build/slim /mnt/NEOGENE1/projects/deepLearn_selection_2018/scripts/simulations/SLiM_scripts/OD_"$T"_"$H"_"$i"
     done

done << EOF
20 20
21 19
22 18
23 17
24 16
25 14
26 14
27 13
28 12
29 11
30 10
31 10
32 9.5
33 9
34 8.5
35 8
36 7.5
37 7
38 6.5
39 6
40 6
EOF

