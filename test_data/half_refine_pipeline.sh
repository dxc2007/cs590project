#!/bin/bash
sigmas=(1)
thres_muls=(0.7 1.2 1.7 2.2)
step2_muls=(1.00)
bfactors=(50 250 750 1000 1500 2000 3000)
for i in "${sigmas[@]}"
do
 for j in "${thres_muls[@]}"
 do
  for k in "${step2_muls[@]}"
  do
   for l in "${bfactors[@]}"
    do
   ./half_refine_pipeline_template.sh $i $j $k $l
   done
  done
 done
done
