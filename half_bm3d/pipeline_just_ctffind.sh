#!/bin/bash
stacks=(stack1 stack6 stack11 stack16)
thres_muls=(0.0 0.5 0.75 1.0 1.25 1.5 1.75 2.25 2.75 3.25 3.75 4.25 4.75 5.25 5.5 6.0 6.5 7.0 7.5 10.0)
sigmas=(1)
step2_muls=(0)
bfactors=(50 250 750 1000 1500 2000 3000)
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
	./just_ctffind.sh $i $j $k $l $m
    done  
   done
  done & 
 done
done
