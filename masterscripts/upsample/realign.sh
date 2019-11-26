echo 'formating frame shifts'
python format_newstack_shifts.py $1_ub_shifts_$5_raw.txt 1.5
echo 'aligning and averaging the frames'
newstack -in $1.mrc -xform $1_ub_shifts_$5_newstack.txt -ou $1_ub_stacked_$5.mrc > /dev/null
echo 'formating frame shifts'
python format_summovie_shifts.py $1_ubbm3d_shifts_$2_$3_$4_$5_half_raw.txt 1.5
echo 'aligning and averaging the frames'
bash summovie.sh $1_ub_stacked_$5.mrc $1_ubbm3d_averaged_$2_$3_$4_$5_half.mrc $1_ubbm3d_shifts_$2_$3_$4_$5_half_summovie.txt $1_ubbm3d_frc_$2_$3_$4_$5_half.txt
