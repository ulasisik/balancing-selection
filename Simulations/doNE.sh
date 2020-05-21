#!/bin/bash 

source Params.txt

CWD=$(pwd)
SIM_FROM=1
SIM_TO=2
H=0.5
S=0.0

Rscript commandNE.R $CWD $S $H $SIM_TO $SIM_FROM

for ((i=$SIM_FROM; i<=$SIM_TO; i++))
do
	$DIRSLIM "$DIRTMP"NE_"$i"
done
