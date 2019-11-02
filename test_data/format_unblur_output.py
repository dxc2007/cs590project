#!/usr/bin/env python
# coding: utf-8

import sys
import re

input_file = sys.argv[1]
scale = float(sys.argv[2])
print("input file: ", input_file)
f = open(input_file,"r")

fl = f.readlines()
f.close()

save_name = re.sub(r'_raw','',input_file)
print("saving to: {} with scale {}".format(save_name, scale))

sf = open(save_name,"w+")
count = 1
for line in fl:
#     print(line)
    if count > 18:
        _, val_str = line.split("=") 
#         print(val_str)
        dx, dy = val_str.split(",")
        sf.write("{} {} {} {} {} {}\n".format(1, 0, 0, 1, float(dx.strip())/scale, float(dy.strip())/scale))
#         print(1, 0, 0, 1, dx.strip(), dy.strip())
    count+=1
sf.close()
