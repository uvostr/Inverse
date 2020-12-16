import numpy as np
import math
import random

x_pixel_len = 512
y_pixel_len = 512

x_len = np.pi
y_len = np.pi

num_layers = 3
r_eye = np.pi / 2
delta_z = 2.9 * 10 ** (-3)
lambda_wave = 0.55 * 10 ** (-6)
focal_len = 20 * 10 ** (-3)

w = 2.3397 * 0.001  # 3 * 0.001;
m = 5
f = 20 * 0.001
lambda_wave = 0.55 * 0.000001
# distances to image and object (metres)
d_1 = 57.4 * 0.001
d_0 = 37 * 0.001
lambdad1 = lambda_wave * d_1
# iris (metres)
# r_0 = 0.5 * pi;
r_0 = 4.5 * 0.001 # 0.5 * pi;
# size of support (1 / 2 of square side)
a = r_0 / lambdad1  # 0.5 * pi;
# PSF koefficient
psValue = np.pi * lambda_wave * d_1 * d_1 * w / ((d_0 + w) * (d_0 + w) * (m - 1))

x_len = a
y_len = a
r_eye = r_0
#print(f'coef: {psValue}, a: {a}')



#Функция для создания рамки

def create_frame(frame_x_pixel_len, frame_y_pixel_len, frame_x_pixel_coor, frame_y_pixel_coor, thickness, intensity):
  fo = np.zeros((x_pixel_len, y_pixel_len))
  for i in range(y_pixel_len):
    for j in range(x_pixel_len):
      if ((i >= frame_y_pixel_coor and i < frame_y_pixel_coor + frame_y_pixel_len and j >= frame_x_pixel_coor and j < frame_x_pixel_coor + thickness)
      or (i >= frame_y_pixel_coor and i < frame_y_pixel_coor + thickness and j >= frame_x_pixel_coor and j < frame_x_pixel_coor + frame_x_pixel_len)
      or (i >= frame_y_pixel_coor and i < frame_y_pixel_coor + frame_y_pixel_len and j >= frame_x_pixel_coor + frame_x_pixel_len - thickness and j < frame_x_pixel_coor + frame_x_pixel_len)
      or (i >= frame_y_pixel_coor + frame_y_pixel_len - thickness and i < frame_y_pixel_coor + frame_y_pixel_len and j >= frame_x_pixel_coor and j < frame_x_pixel_coor + frame_x_pixel_len)):
        fo[i, j] = intensity
      else:
        fo[i, j] = 0
  return fo

def create_circle_frame(frame_radius, frame_x_pixel_coor, frame_y_pixel_coor, thickness, intensity):
  fo = np.zeros((x_pixel_len, y_pixel_len))
  for i in range(y_pixel_len):
    for j in range(x_pixel_len):
      tmp = (i - frame_x_pixel_coor) ** 2 + (j - frame_y_pixel_coor) ** 2
      if((tmp <= (frame_radius + thickness) ** 2) and (tmp >= (frame_radius) ** 2)):
        fo[i, j] = intensity
      else:
        fo[i, j] = 0
  return fo
#Пиксели в координаты

#print(f'x_len: {x_len}')
x = np.mgrid[-x_len:x_len:x_pixel_len*1j]
y = np.mgrid[-y_len:y_len:y_pixel_len*1j]
points = np.vstack((x.ravel(), y.ravel()))

#Функция h

def m(x, y):
  return int(x ** 2 + y ** 2 < (r_eye / lambdad1) ** 2)
def p(x, y):
  return lambdad1 * (x ** 2 + y ** 2) #0.5 * np.pi * (x ** 2 + y ** 2)
def ps(x, y, coef):
  return coef * (x ** 2 + y ** 2)
def f (x, y, coef):
  return m(x, y) * np.exp(1j*(ps(x, y, coef)))




def create_h(delta_z):
  #coef = psValue #np.pi * delta_z / (lambda_wave * focal_len ** 2) #/ (1024 * 8192)
  fh = np.zeros((y_pixel_len, x_pixel_len), complex)
  for i in range(y_pixel_len):
   for j in range(x_pixel_len):
       fh[i, j] = f(points[0, j], points[1, i], delta_z)

  fh = np.fft.fft2(fh)
  fh = np.fft.fftshift(fh)
  fh = fh.imag ** 2 + fh.real ** 2
  return fh

#H слои

def create_H_layers():
  H_layers = np.zeros(( 2 * num_layers - 1, x_pixel_len * 2 - 1, y_pixel_len * 2 - 1), complex)
  for j in range(2 * num_layers - 1):
    #h = create_h(delta_z * (j - num_layers + 1)) * delta_z
    h = create_h(psValue * (j - num_layers + 1))
    new_h = np.zeros((x_pixel_len * 2 - 1, y_pixel_len * 2 - 1))
    new_h[:x_pixel_len, :y_pixel_len] = h
    #H_layers[j] = np.fft.fft2(new_h)
  return H_layers

#O слои

def create_O_layers(o_layers):
  O_layers = np.zeros((num_layers, x_pixel_len * 2 - 1, y_pixel_len * 2 - 1), complex)
  for j in range(num_layers):
    new_o = np.zeros((x_pixel_len * 2 - 1, y_pixel_len * 2 - 1))
    new_o[:x_pixel_len, :y_pixel_len] = o_layers[j]
    O_layers[j] = np.fft.fft2(new_o)
  return O_layers

#Прямая задача

#I слои

def create_I_layers(O_layers, H_layers):
  I_layers = np.zeros((num_layers, x_pixel_len * 2 - 1, y_pixel_len * 2 - 1), complex)
  for m in range(num_layers):
    for n in range(num_layers):
      I_layers[m] = I_layers[m] + O_layers[n] * H_layers[m - n + num_layers - 1]
  return I_layers

#i слои

def create_i_layers(I_layers):
  i_layers = np.zeros((num_layers, x_pixel_len * 2 - 1, y_pixel_len * 2 - 1))
  for j in range(num_layers):
    i_layers[j] = np.fft.ifft2(I_layers[j]).real
  return i_layers

#Обратная задача

def create_I_vector(u, v, I_layers):
  I_vector = np.zeros((num_layers), complex)
  for j in range(num_layers):
    I_vector[j] = I_layers[j][u][v]
  return I_vector

def create_H_matrix(u, v, H_layers):
  H_matrix = np.zeros((num_layers, num_layers), complex)
  for j in range(num_layers):
    for k in range(num_layers):
      H_matrix[j][k] = H_layers[j - k + num_layers - 1][u][v]
  return H_matrix

def regul(k, m, u, v, I_layers, H_layers):
  H = create_H_matrix(u, v, H_layers)
  I = create_I_vector(u, v, I_layers)
  E = np.eye(num_layers)
  O = np.zeros(num_layers)
  tmp1 = np.linalg.inv(E + m * (H.real.T - H.imag.T).dot(H))
  tmp2 = m * np.linalg.inv(E + m * (H.real.T - H.imag.T).dot(H)).dot(H.real.T - H.imag.T).dot(I)
  for j in range(k):
    O = tmp1.dot(O) + tmp2
  return O

def create_result(k, m, I_layers, H_layers):
  result = np.zeros((num_layers, x_pixel_len * 2 - 1, y_pixel_len * 2 - 1), complex)
  for u in range(x_pixel_len * 2 - 1):
    for v in range(y_pixel_len * 2 - 1):
      tmp = regul(k, m, u, v, I_layers, H_layers)
      for i in range(num_layers):
        result[i][u][v] = tmp[i]
  for i in range(num_layers):
    result[i] = np.fft.ifft2(result[i])
  return result.real

# Качество восстановления

def rings_array():
  #радиус вписанного кольца в котором находится точка, -1 если не лежит во вписанном кольце\ 1 если лежит в двух кольцах(полный квадрат), 0 иначе
  rings_array = np.zeros((x_pixel_len * 2 - 1, y_pixel_len * 2 - 1, 2), int)
  for u in range(x_pixel_len * 2 - 1):
    for v in range(y_pixel_len * 2 - 1): 
      tmp = np.sqrt((u - x_pixel_len + 1) ** 2 + (v - y_pixel_len + 1) ** 2)
      if(tmp <= x_pixel_len):
        if(tmp % 1 == 0):
          rings_array[u][v][1] = 1
        rings_array[u][v][0] = int(math.floor(tmp))
      else:
        rings_array[u][v][0] = -1
  return rings_array

def VFC(original, recovered, rings_array):
  original = original / np.var(original)
  original = original / np.mean(original)
  original = np.fft.fft2(original)
  original = original / np.max(original)
  recovered = recovered + np.abs(np.amin(recovered))
  recovered = recovered / np.var(recovered)
  recovered = recovered / np.mean(recovered)
  recovered = np.fft.fft2(recovered)
  recovered = recovered / np.max(recovered)
  tmp = recovered / original
  tmp = np.abs(tmp)
  res = np.zeros((4, x_pixel_len)) #радиус, сумма, количество, сумма/количество
  res[0] = np.arange(0, x_pixel_len, 1)
  for u in range(x_pixel_len * 2 - 1):
    for v in range(y_pixel_len * 2 - 1):
        j = rings_array[u][v][0]
        if(j > 0):
          if(rings_array[u][v][1] == 1):
            res[1][j - 1] += tmp[u][v]
            res[2][j - 1] += 1
          res[1][j] += tmp[u][v]
          res[2][j] += 1
  for i in range(x_pixel_len):
    if(res[2][i] > 0):
      res[3][i] = res[1][i] / res[2][i]
    else:
      res[3][i] = 0
  x_plot = res[0]
  y_plot = res[3]
  return x_plot, y_plot

# def rings_array():
#   #радиус вписанного кольца в котором находится точка, -1 если не лежит во вписанном кольце\ 1 если лежит в двух кольцах(полный квадрат), 0 иначе
#   rings_array = np.zeros((x_num, y_pixel_len, 2), int)
#   for u in range(x_num):
#     for v in range(y_pixel_len): 
#       tmp = np.sqrt((u - x_num / 2) ** 2 + (v - y_pixel_len / 2) ** 2)
#       if(tmp < int(x_num / 2)):
#         if(tmp % 1 == 0):
#           rings_array[u][v][1] = 1
#         rings_array[u][v][0] = int(math.floor(tmp))
#       else:
#         rings_array[u][v][0] = -1
#   return rings_array

# def VFC(original, recovered, rings_array):
#   original = original / np.var(original)
#   original = original / np.mean(original)
#   original = np.fft.fft2(original)
#   original = original / np.max(original)
#   recovered = recovered + np.abs(np.amin(recovered))
#   recovered = recovered / np.var(recovered)
#   recovered = recovered / np.mean(recovered)
#   recovered = np.fft.fft2(recovered)
#   recovered = recovered / np.max(recovered)
#   tmp = recovered / original
#   tmp = np.abs(tmp)
#   res = np.zeros((4, int(x_num / 2))) #радиус, сумма, количество, сумма/количество
#   res[0] = np.arange(0, int(x_num / 2), 1)
#   for u in range(x_num):
#     for v in range(y_pixel_len):
#         j = rings_array[u][v][0]
#         if(j > 0):
#           if(rings_array[u][v][1] == 1):
#             res[1][j - 1] += tmp[u][v]
#             res[2][j - 1] += 1
#           res[1][j] += tmp[u][v]
#           res[2][j] += 1
#   for i in range(int(x_num / 2)):
#     if(res[2][i] > 0):
#       res[3][i] = res[1][i] / res[2][i]
#     else:
#       res[3][i] = 0
#   x_plot = res[0]
#   y_plot = res[3]
#   return x_plot, y_plot


#Гауссовский шум

def add_gaussian_noise(i, noise_level):
  i_gaussian_noise = np.zeros((x_pixel_len * 2 - 1, y_pixel_len * 2 - 1))
  max_amp = np.max(i) - np.min(i)
  for u in range(x_pixel_len * 2 - 1):
    for v in range(y_pixel_len * 2 - 1):
      i_gaussian_noise[u][v] = i[u][v] + random.gauss(0, 1) * max_amp * noise_level
  return i_gaussian_noise

#Пуассоновский шум

def add_shot_noise(i, k): 
  i_shot_noise = np.zeros((x_pixel_len * 2 - 1, y_pixel_len * 2 - 1))  
  for u in range(x_pixel_len * 2 - 1):
    for v in range(y_pixel_len * 2 - 1):
      i_shot_noise[u][v] = np.random.poisson(i[u][v] * k) / k
  return i_shot_noise

# Итоговые функции

def result_without_noise(o_layers, k, m):
  H_layers = create_H_layers()
  O_layers = create_O_layers(o_layers)
  I_layers = create_I_layers(O_layers, H_layers)
  i_layers = create_i_layers(I_layers)
  if(i_layers.min() < 0):
    i_layers += np.math.fabs(i_layers.min())
  i_layers /= np.math.fabs(i_layers.max())
  for j in range(num_layers):
    I_layers[j] = np.fft.fft2(i_layers[j])
  o = create_result(k, m, I_layers, H_layers)
  return o

def result_with_gaussian_noise(o_layers, k, m, noise_level):
  H_layers = create_H_layers()
  O_layers = create_O_layers(o_layers)
  I_layers = create_I_layers(O_layers, H_layers)
  i_layers = create_i_layers(I_layers)
  for j in range(num_layers):
    i_layers[j] = add_gaussian_noise(i_layers[j], noise_level)
  for j in range(num_layers):
    I_layers[j] = np.fft.fft2(i_layers[j])
  o = create_result(k, m, I_layers, H_layers)
  return o

def result_with_shot_noise(o_layers, k, m, photons_number):
  H_layers = create_H_layers()
  O_layers = create_O_layers(o_layers)
  I_layers = create_I_layers(O_layers, H_layers)
  i_layers = create_i_layers(I_layers)
  if(i_layers.min() < 0):
    i_layers += np.math.fabs(i_layers.min())
  i_layers /= np.math.fabs(i_layers.max())
  for j in range(num_layers):
    i_layers[j] = add_shot_noise(i_layers[j], photons_number)
  for j in range(num_layers):
    I_layers[j] = np.fft.fft2(i_layers[j])
  o = create_result(k, m, I_layers, H_layers)
  return o


def recovery_quality(original, recovered):
  extended_o = np.zeros((x_pixel_len * 2 - 1, y_pixel_len * 2 - 1))
  extended_o[:x_pixel_len, :y_pixel_len] = original
  rings_array2 = rings_array()
  x_plot, y_plot = VFC(extended_o, recovered, rings_array2)
  return x_plot, y_plot