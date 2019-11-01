#!/bin/bash
sigmas=(0.1 0.3 0.5 0.8)
thres_muls=(0.7)
step2_muls=(0.33)
for i in "${sigmas[@]}"
do
 for j in "${thres_muls[@]}"
 do
  for k in "${step2_muls[@]}"
  do
   ./refine_pipeline_template.sh $i $j $k
  done
 done
done
