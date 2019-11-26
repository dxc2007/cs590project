#!/bin/bash
filename=$(basename $BASH_SOURCE)
extension="${filename##*.}"
filename="${filename%.*}"
echo "${filename}"

stacks=(stack19)

sigmas=(1)
thres_muls=(3.5 4.0 4.5)
step2_muls=(0.1 0.2 0.3 0.4)
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
 done &
done
wait
done
