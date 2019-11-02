#!/bin/dash
echo 'filtering using bm3d algorithm'
python bm3d_filtering.py shrinked_stacked.mrc bm3d_filtered.mrc $1 $2 $3
echo 'performing unblur and extracting frame shifts'
bash unblur.sh bm3d_filtered.mrc $1 $2 $3 full $4 > bm3d_shifts_$1_$2_$3_$4_full_raw.txt
echo 'formating frame shifts'
python format_summovie_shifts.py bm3d_shifts_$1_$2_$3_$4_full_raw.txt 3
echo 'aligning and averaging the frames'
bash summovie.sh shrinked_stacked.mrc bm3d_averaged_$1_$2_$3_$4_full.mrc bm3d_shifts_$1_$2_$3_$4_full.txt bm3d_frc_$1_$2_$3_$4_full.txt
echo 'extracting ctf score'
bash ctffind.sh bm3d_averaged_$1_$2_$3_$4_full.mrc bm3d_diagnostic_$1_$2_$3_$4_full.mrc > bm3d_summary_$1_$2_$3_$4_full.txt
