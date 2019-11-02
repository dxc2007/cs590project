#!/usr/bin/env python
# coding: utf-8

# start of actual process to open mrc file
import mrcfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# replace file name if necessary
input_file = sys.argv[1] 
mrc = mrcfile.open(input_file,mode='r')
print("input file: ", input_file)
# mrc = mrcfile.open('',mode='r')
save_name = sys.argv[2] 

img_stack = np.copy(mrc.data)

# threshold is a certain value multiplied by sigma
sigma = float(sys.argv[3])
# given default was 2.7
thres_mul = float(sys.argv[4])
step2_mul = float(sys.argv[5])
Threshold_Hard3D = thres_mul*sigma

# sigma can be adjusted
sigma_color = [0, 0, 0]
sigma_color[0] = step2_mul*sigma

org_min, org_max = np.min(img_stack), np.max(img_stack)

# convert floating type to int type between 0 and 255 (inclusive)
norm_stack = cv2.normalize(img_stack, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

norm_stack_expanded=np.expand_dims(norm_stack,axis=3)

# do a 2d transform for a stack of images
def transform_2d(image_stack):
  transformed_stack = np.zeros(image_stack.shape,dtype='float32')
  for i in range(transformed_stack.shape[0]):
    transformed_stack[i] = cv2.dct(image_stack[i])
  return transformed_stack

# reverse the 2d transform
def reverse_transform_2d(transformed_stack):
  filtered_stack = np.zeros(transformed_stack.shape)
  for i in range(filtered_stack.shape[0]):
    filtered_stack[i] = cv2.idct(transformed_stack[i])
  return filtered_stack


def Step1_3DFiltering_color(_similar_blocks):
    t1 = cv2.getTickCount()
    chnl = _similar_blocks.shape[3] # chnl = 3 for color image
    # statis_nonzero = np.zeros(chnl, dtype=int)
    m_Shape = _similar_blocks.shape
    new_blocks = np.zeros(_similar_blocks.shape)
    # print("similar blocks shape", _similar_blocks.shape)avg_frame

    # code below is computationally expensive

    # for each pixel location at each channel, do cosine transform, hard thresholding and cosine transform back
    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            for ch in range(chnl):
                tem_Vct_Trans = cv2.dct(_similar_blocks[:, i, j, ch])
                tem_Vct_Trans[np.abs(tem_Vct_Trans[:]) < Threshold_Hard3D] = 0.
                # statis_nonzero[ch] += tem_Vct_Trans.nonzero()[0].size
                new_blocks[:, i, j, ch] = cv2.idct(tem_Vct_Trans).flatten()
    t2 = cv2.getTickCount()
    time = (t2-t1)/cv2.getTickFrequency()
    print("Initial step processing time taken is {} seconds".format(time))
    return new_blocks


def Step2_3DFiltering_color(_Similar_Bscs, _Similar_Imgs):
    t1 = cv2.getTickCount()
    chnl = _Similar_Bscs.shape[3] # chnl = 3 for color image
    m_Shape = _Similar_Bscs.shape
    # Wiener_wight = np.zeros((m_Shape[1], m_Shape[2], m_Shape[3]), dtype=float)
    Count = _Similar_Bscs.shape[0]
    final_blocks = np.zeros(_Similar_Bscs.shape)

    # Wiener filtering
    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            for ch in range(chnl):
                tem_vector = _Similar_Bscs[:, i, j, ch]
                tem_Vct_Trans = np.matrix(cv2.dct(tem_vector))
                # find the l2 norm
                Norm_2 = np.float64(tem_Vct_Trans.T * tem_Vct_Trans)

                m_weight = Norm_2/Count/(Norm_2/Count + sigma_color[ch]**2)
                # print("m weight shape", m_weight.shape)
                # print("m_weight = "+str(m_weight))

                tem_vector = _Similar_Imgs[:, i, j, ch]
                tem_Vct_Trans = m_weight * cv2.dct(tem_vector)
                final_blocks[:, i, j, ch] = cv2.idct(tem_Vct_Trans).flatten()
    t2 = cv2.getTickCount()
    time = (t2-t1)/cv2.getTickFrequency()
    print("Final step processing time taken is {} seconds".format(time))
    return final_blocks

def bm3d_cycle(image_stack):
  transformed_stack = transform_2d(image_stack)
  transformed_stack_expanded = np.expand_dims(transformed_stack,axis=3)
  basic_blocks = Step1_3DFiltering_color(transformed_stack_expanded)    
  final_blocks = Step2_3DFiltering_color(basic_blocks, transformed_stack_expanded)
  final_blocks_squeezed = np.squeeze(final_blocks)
  filtered_stack = reverse_transform_2d(final_blocks_squeezed)
  return filtered_stack

def bm3d_half_cycle(image_stack):	
  transformed_stack = transform_2d(image_stack)
  transformed_stack_expanded = np.expand_dims(transformed_stack,axis=3)
  final_blocks = Step1_3DFiltering_color(transformed_stack_expanded)    
  final_blocks_squeezed = np.squeeze(final_blocks)
  filtered_stack = reverse_transform_2d(final_blocks_squeezed)
  return filtered_stack


# filtered_stack = bm3d_cycle(norm_stack)
filtered_stack = bm3d_cycle(img_stack)
# filtered_stack = bm3d_half_cycle(img_stack)

# get the original average frame
avg_frame = norm_stack.mean(axis=0)

# get the filtered average frame
avg_filtered_frame = filtered_stack.mean(axis=0)


# plt.figure(figsize=(20,10))
# plt.imshow(avg_frame,cmap='gray')
# plt.show()

# plt.figure(figsize=(20,10))
# plt.imshow(avg_filtered_frame,cmap='gray')
# plt.show()


scaled_filtered = cv2.normalize(filtered_stack, None, alpha = org_min, beta = org_max, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

# save the file
print("saving to: {}".format(save_name))
with mrcfile.new(save_name, overwrite=True) as mrc:
    mrc.set_data(scaled_filtered)
