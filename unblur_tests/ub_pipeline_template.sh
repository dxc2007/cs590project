#!/bin/dash
echo 'performing unblur and averaging'
bash pure_unblur.sh $1 $2 > $1_ub_shifts_$2_raw.txt
echo 'formating frame shifts'
python format_summovie_shifts.py $1_ub_shifts_$2_raw.txt 3
echo 'aligning and averaging the frames'
bash summovie.sh $1.mrc $1_ub_averaged_$2.mrc $1_ub_shifts_$2.txt $1_ub_frc_$2.txt
echo 'extracting ctf score'
bash pure_ctffind.sh $1_ub_averaged_$2.mrc $1_ub_diagnostic_$2.mrc > $1_ub_summary_$2.txt
