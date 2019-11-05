#!/usr/bin/env python
# coding: utf-8

import sys
import re

input_file = sys.argv[1]
# input_file = "./test_data/ub_shifts_raw.txt"
scale = float(sys.argv[2])
# scale = 3
print("input file: ", input_file)
f = open(input_file,"r")

fl = f.readlines()
f.close()

save_name = re.sub(r'_raw','',input_file)
print("saving to: {} with scale {}".format(save_name, scale))

dxs=[]
dys=[]

sf = open(save_name,"w+")
count = 1
for line in fl:
#     print(line)
    if count > 25:
        _, val_str = line.split("=") 
#         print(val_str)
        dx, dy = val_str.split(",")
        dxs.append(str(float(dx.strip())/scale))
        dys.append(str(float(dy.strip())/scale))
#         print(1, 0, 0, 1, dx.strip(), dy.strip())
    count+=1
sf.write(" ".join(dxs))
sf.write("\n")
sf.write(" ".join(dys))
sf.write("\n")
sf.close()
