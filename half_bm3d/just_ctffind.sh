#!/bin/dash
echo 'extracting ctf score'
bash ctffind.sh $5_bm3d_averaged_$1_$2_$3_$4_half.mrc $5_bm3d_diagnostic_$1_$2_$3_$4_half.mrc > $5_bm3d_summary_$1_$2_$3_$4_half.txt
