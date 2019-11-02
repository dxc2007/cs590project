#!/bin/dash
echo 'filtering using first_bm3d algorithm'
python bm3d_first_step_filtering.py shrinked_stacked.mrc first_bm3d_filtered.mrc $1 $2 $3
echo 'performing unblur and extracting frame shifts'
bash unblur.sh first_bm3d_filtered.mrc $1 $2 $3 half $4 > first_bm3d_shifts_$1_$2_$3_$4_half_raw.txt
echo 'formating frame shifts'
python format_summovie_shifts.py first_bm3d_shifts_$1_$2_$3_$4_half_raw.txt 3
echo 'aligning and averaging the frames'
bash summovie.sh shrinked_stacked.mrc first_bm3d_averaged_$1_$2_$3_$4_half.mrc first_bm3d_shifts_$1_$2_$3_$4_half.txt first_bm3d_frc_$1_$2_$3_$4_half.txt
echo 'extracting ctf score'
bash ctffind.sh first_bm3d_averaged_$1_$2_$3_$4_half.mrc first_bm3d_diagnostic_$1_$2_$3_$4_half.mrc > first_bm3d_summary_$1_$2_$3_$4_half.txt
