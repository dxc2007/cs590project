#!/bin/bash
# for each .mrc file in directory $1, downsample by $2 and put into the same dir
counter=1
for f in $1/*.mrc
do
 echo "formatting {$f}"
 newstack -in $f -shrink $2 -ou $1/stack$counter.mrc
 let counter++
done
