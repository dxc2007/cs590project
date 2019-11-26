#!/bin/dash
INPUT_FILE="$1_ub_stacked_$2.mrc"
if [ -f "$INPUT_FILE" ]
then
 echo 'we already have the unblurred input file'
else
 echo 'performing the pre-bm3d unblur'
 bash ub_stack_pipeline.sh $1 $2
fi
echo 'performing unblur and averaging'
bash unblur.sh $1_ub_stacked_$2 $2 > $1_ubx2_shifts_$2_raw.txt
echo 'formating frame shifts'
python format_summovie_shifts.py $1_ubx2_shifts_$2_raw.txt 1.5 
echo 'aligning and averaging the frames'
bash summovie.sh $1_ub_stacked_$2.mrc $1_ubx2_averaged_$2.mrc $1_ubx2_shifts_$2_summovie.txt $1_ubx2_frc_$2.txt > /dev/null
echo 'extracting ctf score'
bash ctffind.sh $1_ubx2_averaged_$2.mrc $1_ubx2_diagnostic_$2.mrc > $1_ubx2_summary_$2.txt
