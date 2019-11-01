#!/bin/dash
echo 'filtering using bm3d algorithm'
python bm3d_filtering.py shrinked_stacked.mrc bm3d_filtered.mrc $1 $2 $3
echo 'performing unblur and extracting frame shifts'
bash unblur.sh bm3d_filtered.mrc > bm3d_shifts_raw.txt
echo 'formating frame shifts'
python format_unblur_output.py bm3d_shifts_raw.txt 1.5
echo 'aligning the frames'
newstack -in shrinked_stacked.mrc -xform bm3d_shifts.txt -ou bm3d_stacked.mrc > /dev/null
echo 'taking the average'
clip average bm3d_stacked.mrc bm3d_averaged.mrc > /dev/null
echo 'extracting ctf score'
bash ctffind.sh bm3d_stacked.mrc bm3d_diagnostic.mrc > bm3d_summary_$1_$2_$3_half.txt
