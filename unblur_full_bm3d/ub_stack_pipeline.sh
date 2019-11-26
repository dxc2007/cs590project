#!/bin/dash
echo 'performing unblur and averaging'
bash unblur.sh $1 $2 > $1_ub_shifts_$2_raw.txt
echo 'formating frame shifts'
python format_newstack_shifts.py $1_ub_shifts_$2_raw.txt 3
echo 'aligning and averaging the frames'
newstack -in $1.mrc -xform $1_ub_shifts_$2_newstack.txt -ou $1_ub_stacked_$2.mrc
