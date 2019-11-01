#!/usr/bin/env python
# coding: utf-8

# In[50]:


import matplotlib.pyplot as plt
import numpy as np

import sys
import re

input_file = sys.argv[1]
# input_file = "./test_data/ub_diagnostic_avrot.txt"
print("input file: ", input_file)
save_name = re.sub(r'avrot.txt','plot.png',input_file)
print("saving to: {}".format(save_name))

f = open(input_file,"r")

fl = f.readlines()
f.close()


# In[51]:


count = 1
axes = []
for line in fl:
    if count > 5:
        vals=line.split()
#         print(vals)
        axes.append(list(map(float,vals)))
    count += 1


# In[52]:


x, y1, y2, y3, y4, y5 = axes
plt.figure(figsize=(20,10))
plt.plot(x, y1)
# plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
# plt.plot(x, y5)
plt.savefig(save_name)
# plt.show()


# In[ ]:




