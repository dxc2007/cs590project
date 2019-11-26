#!/bin/bash
filename=$(basename $BASH_SOURCE)
extension="${filename##*.}"
filename="${filename%.*}"
echo "${filename}"

stacks=(stack19)
bfactors=(50 250 750)
for i in "${stacks[@]}"
 do
  for j in "${bfactors[@]}"
   do
    ./${filename}_pipeline.sh $i $j
   done &
 done

