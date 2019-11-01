#!/bin/dash
python bm3d_filtering.py shrinked_stacked.mrc bm3d_filtered.mrc
bash unblur.sh bm3d_filtered.mrc > bm3d_shifts_raw.txt
python format_unblur_output.py bm3d_shifts_raw.txt 1.5
newstack -in shrinked_stacked.mrc -xform bm3d_shifts.txt -ou bm3d_stacked.mrc
clip average bm3d_stacked.mrc bm3d_averaged.mrc
bash ctffind.sh bm3d_stacked.mrc bm3d_diagnostic.mrc

python bm3d_filtering.py bm3d_stacked.mrc bm3d_filtered2.mrc
bash unblur.sh bm3d_filtered2.mrc > bm3d_shifts2_raw.txt
python format_unblur_output.py bm3d_shifts2_raw.txt 1.5
newstack -in bm3d_stacked.mrc -xform bm3d_shifts2.txt -ou bm3d_stacked2.mrc
clip average bm3d_stacked2.mrc bm3d_averaged2.mrc
bash ctffind.sh bm3d_stacked2.mrc bm3d_diagnostic2.mrc

python bm3d_filtering.py bm3d_stacked2.mrc bm3d_filtered3.mrc
bash unblur.sh bm3d_filtered3.mrc > bm3d_shifts3_raw.txt
python format_unblur_output.py bm3d_shifts3_raw.txt 1.5
newstack -in bm3d_stacked2.mrc -xform bm3d_shifts3.txt -ou bm3d_stacked3.mrc
clip average bm3d_stacked3.mrc bm3d_averaged3.mrc
bash ctffind.sh bm3d_stacked3.mrc bm3d_diagnostic3.mrc
