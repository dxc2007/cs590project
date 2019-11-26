#!/bin/bash
stacks=(stack1 stack2 stack3 stack4 stack5 stack6 stack7 stack8 stack9 stack10 stack11 stack12 stack13 stack14 stack15 stack16 stack17 stack18 stack19 stack20)
sigmas=(1)
thres_muls=(0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
step2_muls=(0)
bfactors=(0 250 500 750 1000 1250 1500 1750 2000)
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
  done &
 done
done
