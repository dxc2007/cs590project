#!/bin/bash
stacks=(stack1 stack6 stack11 stack16)
sigmas=(1)
thres_muls=(0)
step2_muls=(0)
bfactors=(1250 2500)
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
	./half_refine_pipeline_template.sh $i $j $k $l $m
    done  
   done
  done 
 done
done
