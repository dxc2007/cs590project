#!/bin/dash
bash unblur.sh shrinked_stacked.mrc > ub_shifts_raw.txt
python format_unblur_output.py ub_shifts_raw.txt 1.5
newstack -in shrinked_stacked.mrc -xform ub_shifts.txt -ou ub_stacked.mrc
clip average ub_stacked.mrc ub_averaged.mrc
bash ctffind.sh ub_stacked.mrc ub_diagnostic.mrc

bash unblur.sh ub_stacked.mrc > ub_shifts2_raw.txt
python format_unblur_output.py ub_shifts2_raw.txt 1.5
newstack -in ub_stacked.mrc -xform ub_shifts2.txt -ou ub_stacked2.mrc
clip average ub_stacked2.mrc ub_averaged2.mrc
bash ctffind.sh ub_stacked2.mrc ub_diagnostic2.mrc

bash unblur.sh ub_stacked2.mrc > ub_shifts3_raw.txt
python format_unblur_output.py ub_shifts3_raw.txt 1.5
newstack -in ub_stacked2.mrc -xform ub_shifts3.txt -ou ub_stacked3.mrc
clip average ub_stacked3.mrc ub_averaged3.mrc
bash ctffind.sh ub_stacked3.mrc ub_diagnostic3.mrc
