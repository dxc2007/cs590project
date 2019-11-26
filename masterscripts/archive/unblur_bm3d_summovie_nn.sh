#!/bin/dash
echo 'aligning the frames'
newstack -nearest -in $1.mrc -xform $1_ub_shifts_$2_newstack.txt -ou $1_ub_stacked_$2_nn.mrc
echo 'averaging the frames'
clip average $1_ub_stacked_$2_nn.mrc $1_ub_averaged_$2_nn.mrc
echo 'extracting ctf score'
bash ctffind.sh $1_ub_averaged_$2_nn.mrc $1_ub_diagnostic_$2_nn.mrc > $1_ub_summary_$2_nn.txt
