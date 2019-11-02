#!/bin/dash
echo 'performing unblur and averaging'
bash pure_unblur.sh shrinked_stacked.mrc
echo 'extracting ctf score'
bash pure_ctffind.sh pure_ub_averaged.mrc pure_ub_diagnostic.mrc > pure_ub_summary.txt
