#!/bin/bash 

source Params.txt

CWD=$(pwd)
SIM_FROM=1
SIM_TO=2
H=0.5

while read T S
do 
     Rscript commandIS.R $CWD $S $H $T $SIM_TO $SIM_FROM

     for ((i=$SIM_FROM; i<=$SIM_TO; i++))
     do
	     $DIRSLIM "$DIRTMP"IS_"$T"_"$i"
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

