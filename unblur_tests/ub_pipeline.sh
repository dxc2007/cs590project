#!/bin/bash
stacks=(stack2 stack3 stack4 stack5 stack7 stack8 stack9 stack10 stack12 stack13 stack14 stack15 stack17 stack18 stack19 stack20)
bfactors=(50 250 750 1000 1500 2000 3000)
for i in "${stacks[@]}"
 do
  for j in "${bfactors[@]}"
   do
    ./ub_pipeline_template.sh $i $j
   done &
 done

