#!/bin/bash 

source Params.txt

CWD=$(pwd)
SIM_FROM=1
SIM_TO=2
S=0.001

while read T H
do 
     Rscript commandOD.R $CWD $S $H $T $SIM_TO $SIM_FROM

     for ((i=$SIM_FROM; i<=$SIM_TO; i++))
     do
	     $DIRSLIM "$DIRTMP"OD_"$T"_"$i"
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

