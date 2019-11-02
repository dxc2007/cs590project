#!/bin/bash
sigmas=(0.5 1 3)
thres_muls=(1.7)
step2_muls=(0.33)
bfactors=(250 750 1500 3000)
for i in "${sigmas[@]}"
do
 for j in "${thres_muls[@]}"
 do
  for k in "${step2_muls[@]}"
  do
   for l in "${bfactors[@]}"
    do
   ./refine_pipeline_template.sh $i $j $k $l
   done
  done
 done
done
