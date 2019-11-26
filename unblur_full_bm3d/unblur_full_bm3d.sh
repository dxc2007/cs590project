#!/bin/bash
filename=$(basename $BASH_SOURCE)
extension="${filename##*.}"
filename="${filename%.*}"
echo "${filename}"

stacks=(stack16 stack17 stack18 stack19 stack20)

sigmas=(1)
thres_muls=(3.5 4.0 4.5)
step2_muls=(0 0.02 0.05 0.07)
bfactors=(0 500 1000 1500 2000)
for h in "${stacks[@]}"
do
for i in "${sigmas[@]}"
do
 for j in "${thres_muls[@]}"
 do
  for k in "${step2_muls[@]}"
  do
   for l in "${bfactors[@]}"
    do
	./${filename}_pipeline.sh $i $j $k $l $h
    done      
   done
  done
 done &
done
