#!/bin/dash
FILE="$5_bm3d_$1_$2_$3_half.mrc"
if [ -f "$FILE" ]
then
 echo 'we already have the bm3d file'
else
 echo 'filtering using bm3d algorithm'
 python bm3d_first_step_filtering.py $5.mrc $5_bm3d_$1_$2_$3_half.mrc $1 $2 $3
fi
echo 'performing unblur and extracting frame shifts'
bash unblur.sh $5 $1 $2 $3 $4 half > $5_bm3d_shifts_$1_$2_$3_$4_half_raw.txt
echo 'formating frame shifts'
python format_summovie_shifts.py $5_bm3d_shifts_$1_$2_$3_$4_half_raw.txt 3
echo 'aligning and averaging the frames'
bash summovie.sh $5.mrc $5_bm3d_averaged_$1_$2_$3_$4_half.mrc $5_bm3d_shifts_$1_$2_$3_$4_half.txt $5_bm3d_frc_$1_$2_$3_$4_half.txt
echo 'extracting ctf score'
bash ctffind.sh $5_bm3d_averaged_$1_$2_$3_$4_half.mrc $5_bm3d_diagnostic_$1_$2_$3_$4_half.mrc > $5_bm3d_summary_$1_$2_$3_$4_half.txt
