#!/bin/dash
echo 'performing unblur and averaging'
bash pure_unblur.sh shrinked_stacked.mrc $1 > ub_shifts_$1_raw.txt
echo 'formating frame shifts'
python format_summovie_shifts.py ub_shifts_$1_raw.txt 3
echo 'aligning and averaging the frames'
bash summovie.sh shrinked_stacked.mrc ub_averaged_$1.mrc ub_shifts_$1.txt ub_frc_$1.txt
echo 'extracting ctf score'
bash pure_ctffind.sh ub_averaged_$1.mrc ub_diagnostic_$1.mrc > ub_summary_$1.txt
