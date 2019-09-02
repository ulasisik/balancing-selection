#!/bin/bash 

REP_FROM=1
REP_TO=10000
H=0.5

while read T S
do 
     Rscript commandIS.R $S $H $T $REP_TO $REP_FROM

     for ((i=$REP_FROM; i<=$REP_TO; i++))
     do
	     /usr/local/sw/SLiM-3.2/build/slim /mnt/NEOGENE1/projects/deepLearn_selection_2018/scripts/simulations/SLiM_scripts/IS_"$T"_"$S"_"$i"
     done

done << EOF
20 0.019
21 0.018
22 0.017
23 0.016
24 0.016
25 0.015
26 0.014
27 0.014
28 0.013
29 0.012
30 0.011
31 0.011
32 0.010
33 0.0098
34 0.0085
35 0.0077
36 0.0075
37 0.0073
38 0.0070
39 0.0067
40 0.0064
EOF

