#!/bin/bash
filename=$(basename $BASH_SOURCE)
extension="${filename##*.}"
filename="${filename%.*}"
echo "${filename}"

stacks=(stack19)
sigmas=(1)
thres_muls=(0.0)
step2_muls=(0)
bfactors=(0 50 250)
for i in "${sigmas[@]}"
do
 for j in "${thres_muls[@]}"
 do
  for k in "${step2_muls[@]}"
  do
   for l in "${bfactors[@]}"
    do
     for m in "${stacks[@]}"
      do
	./${filename}_pipeline.sh $i $j $k $l $m
    done 
   done
  done &
 done
done
