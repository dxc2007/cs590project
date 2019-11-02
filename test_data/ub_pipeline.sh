#!/bin/bash
bfactors=(50 250 750 1000 1500 2000 3000)
for i in "${bfactors[@]}"
do
 ./ub_pipeline_template.sh $i
done

