#!/bin/bash 

REP_FROM=1
REP_TO=10000
H=0.5
S=0.0

while read T A B
do 
    Rscript commandFD.R $S $H $T $REP_TO $REP_FROM $A $B

    for (( i=$REP_FROM; i<=$REP_TO; i++))
    do
	/usr/local/sw/SLiM-3.2/build/slim /mnt/NEOGENE1/projects/deepLearn_selection_2018/scripts/simulations/SLiM_scripts/FD_"$T"_"$A"_"$i"
    done

done << EOF
20 1.030 0.060
21 1.029 0.058
22 1.028 0.056
23 1.027 0.054
24 1.026 0.052
25 1.025 0.050
26 1.024 0.048
27 1.023 0.046
28 1.022 0.044
29 1.021 0.042
30 1.020 0.040
31 1.019 0.038
32 1.018 0.036
33 1.017 0.034
34 1.016 0.032
35 1.015 0.030
36 1.014 0.028
37 1.013 0.026
38 1.012 0.024
39 1.011 0.022
40 1.010 0.020
EOF

