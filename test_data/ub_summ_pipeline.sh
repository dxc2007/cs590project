#!/bin/dash
echo 'performing unblur and extracting frame shifts'
bash pure_unblur.sh shrinked_stacked.mrc > ub_shifts_raw.txt
echo 'formating frame shifts'
python format_summovie_shifts.py ub_shifts_raw.txt 3
echo 'aligning and averaging the frames'
bash summovie.sh shrinked_stacked.mrc ub_averaged.mrc ub_shifts.txt ub_frc.txt
echo 'extracting ctf score'
bash ctffind.sh ub_averaged.mrc ub_diagnostic.mrc > ub_summary.txt
