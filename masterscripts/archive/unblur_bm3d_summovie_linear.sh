#!/bin/dash
echo 'performing unblur and averaging'
bash unblur.sh $1 $2 > $1_ub_shifts_$2_raw.txt
echo 'formating frame shifts'
python format_newstack_shifts.py $1_ub_shifts_$2_raw.txt 3
echo 'aligning and averaging the frames'
newstack -in $1.mrc -linear -xform $1_ub_shifts_$2_newstack.txt -ou $1_ub_stacked_$2_linear.mrc
echo 'averaging the frames'
clip average $1_ub_stacked_$2_linear.mrc $1_ub_averaged_$2_linear.mrc
echo 'extracting ctf score'
bash ctffind.sh $1_ub_averaged_$2_linear.mrc $1_ub_diagnostic_$2_linear.mrc > $1_ub_summary_$2_linear.txt
