#!/bin/dash
INPUT_FILE="$5_ub_stacked_$4.mrc"
FILTERED_FILE="$5_ubbm3d_$1_$2_$3_half.mrc"
if [ -f "$INPUT_FILE" ]
then
 echo 'we already have the unblurred input file'
else
 echo 'performing the pre-bm3d unblur'
 bash ub_stack_pipeline.sh $5 $4
fi
if [ -f "$FILTERED_FILE" ]
then
 echo 'we already have the ubbm3d file'
else
 echo 'filtering using bm3d algorithm'
 python bm3d_first_step_filtering.py $5_ub_stacked_$4.mrc $5_ubbm3d_$1_$2_$3_$4_half.mrc $1 $2 $3
fi
echo 'performing unblur and extracting frame shifts'
bash unblur.sh $5_ubbm3d_$1_$2_$3_$4_half $4 > $5_ubbm3d_shifts_$1_$2_$3_$4_half_raw.txt
echo 'formating frame shifts'
python format_summovie_shifts.py $5_ubbm3d_shifts_$1_$2_$3_$4_half_raw.txt 3
echo 'aligning and averaging the frames'
bash summovie.sh $5_ub_stacked_$4.mrc $5_ubbm3d_averaged_$1_$2_$3_$4_half.mrc $5_ubbm3d_shifts_$1_$2_$3_$4_half_summovie.txt $5_ubbm3d_frc_$1_$2_$3_$4_half.txt
echo 'extracting ctf score'
bash ctffind.sh $5_ubbm3d_averaged_$1_$2_$3_$4_half.mrc $5_ubbm3d_diagnostic_$1_$2_$3_$4_half.mrc > $5_ubbm3d_summary_$1_$2_$3_$4_half.txt
