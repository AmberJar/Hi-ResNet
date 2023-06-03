import cv2
import numpy as np

water_mask_path = '/data/fpc/inference/test/taibei_test/standard_taibei_36.png'
green_mask_path = '/data/fpc/inference/standard_taibei_36.png'
road_mask_path = '/data/fpc/output/outputs_10_24_8/normal_mask_output/standard_taibei_36.png'

water_mask = cv2.imread(water_mask_path, 0)
green_mask = cv2.imread(green_mask_path, 0)
road_mask = cv2.imread(road_mask_path, 0)

shape = water_mask.shape

res_matrix = np.zeros(shape)

res_matrix[green_mask == 8] = 1
res_matrix[green_mask == 9] = 2
res_matrix[road_mask == 1] = 3
res_matrix[water_mask == 7] = 4

cv2.imwrite('/data/fpc/inference/res_matrix.png', res_matrix)


