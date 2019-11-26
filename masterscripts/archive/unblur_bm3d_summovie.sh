#!/bin/bash
stacks=(stack1 stack10 stack15 stack18 stack19)
bfactors=(50 250 750 1500 2000)
for i in "${stacks[@]}"
 do
  for j in "${bfactors[@]}"
   do
   # ./unblur_bm3d_summovie_cubic.sh $i $j
   ./unblur_bm3d_summovie_nn.sh $i $j
   done &
 done

