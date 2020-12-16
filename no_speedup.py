#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
def m_func(x, y, R, n_coef):
    x_n = x * n_coef
    y_n = y * n_coef
    if x_n * x_n + y_n * y_n <= R * R:
        return 1
    else:
        return 0

def p_func(x, y, a):
    return a * (x * x + y * y)

def generate_psf2(dim, m, w_, stationary_defocus):
        
    w = w_ * 2.34 * 0.001

    f = 20 * 0.001
    wavelength = 0.55 * 0.000001
    d_1 = 57.4 * 0.001
    d_0 = 37 * 0.001
    r_0 = 4.5 * 0.001

    lambda_d1 = wavelength * d_1
    a = 2 * r_0 / lambda_d1
    dzeta = 1.0 * w


    ps_value = - np.pi * wavelength * d_1 * d_1 * dzeta / ((d_0 + w) * (d_0 + w))
    n_coef = lambda_d1

    dim_x = dim
    dim_y = dim
    min_x = -a
    max_x = a
    min_y = -a
    max_y = a
    range_x = max_x - min_x
    range_y = max_y - min_y
    R = r_0

    delta_x = (max_x - min_x) / dim_x
    delta_xi = 1.0 / dim_x / delta_x
    dim_delta_xi = delta_xi * dim

    arg_x = np.linspace(start=min_x, stop=max_x, num=dim_x)
    arg_y = np.linspace(start=min_y, stop=max_y, num=dim_y)
    inner_h = np.zeros((dim_x, dim_y), dtype=np.cdouble)
    fft_inner_h = np.zeros((dim_x, dim_y), dtype=np.cdouble)
    x = arg_x
    y = arg_y
    inner_h_view = inner_h
    fft_inner_h_view = fft_inner_h

    h = np.zeros((dim_x, dim_y, 2 * m - 1), dtype=np.double)

    for l in range(2 * m - 1):
        layer = l - m + 1
        for i in range(dim_x):
            for j in range(dim_y):
                temp = p_func(x[i], y[j], ps_value * (layer) / (m-1)) + p_func(x[i], y[j], ps_value * stationary_defocus)
                inner_h_view[i, j] = m_func(x[i], y[j], R, n_coef) * (np.cos(temp) + 1j * np.sin(temp))

        fft_inner_h = np.fft.fft2(inner_h)
        fft_inner_h = np.fft.fftshift(fft_inner_h)

        fft_inner_h = np.abs(fft_inner_h)
        fft_inner_h = fft_inner_h ** 2
        h[:, :, l] = np.real(fft_inner_h)

    h_max = np.amax(h)
    h = h / h_max

    return h


def generate_out_images2(dim, m, w_, stationary_defocus, src):

    ext_dim = dim * 2 - 1
    cut_dim_s = dim / 2
    cut_dim_f = 3 * dim / 2
    h = np.zeros((dim, dim, 2 * m - 1), dtype=np.double)
    out = np.zeros((dim, dim, m), dtype=np.double)
    h = generate_psf2(dim, m, w_, stationary_defocus)
    ext_h = np.zeros((ext_dim, ext_dim, 2 * m - 1), dtype=np.complex)
    ext_src = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)
    ext_out = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)

    for l in range(m):
        ext_src[:dim, :dim, l] = src[:, :, l]
    for l in range(2 * m - 1):
        ext_h[:dim, :dim, l] = h[:, :, l]

    for l in range(m):
        ext_src[:, :, l] = np.fft.fft2(ext_src[:, :, l])
    for l in range(2 * m - 1):
        ext_h[:, :, l] = np.fft.fft2(ext_h[:, :, l])

    for k in range(m):
        for l in range(m):
            ext_out[:, :, k] = ext_out[:, :, k] +                                ext_src[:, :, l] * ext_h[:, :, l - k + m - 1]

    for l in range(m):
        ext_out[:, :, l] = np.fft.ifft2(ext_out[:, :, l])


    for l in range(m):
        out[:, :, l] = np.real(ext_out[int(cut_dim_s):int(cut_dim_f), int(cut_dim_s):int(cut_dim_f), l])

    out = out / np.amax(out)

    bytes_out = np.ndarray((dim, dim, m), np.ubyte)
    bytes_out = (255 * out).astype(np.ubyte)

    return out, bytes_out


def create_H_matrix(m, u, v, H_layers):
  H_matrix = np.zeros((m, m), complex)
  for j in range(m):
    for k in range(m):
      H_matrix[j][k] = H_layers[u][v][j - k + m - 1]
  return H_matrix

def create_I_vector(m, u, v, I_layers):
  I_vector = np.zeros((m), complex)
  for j in range(m):
    I_vector[j] = I_layers[u][v][j]
  return I_vector

def regul_implicit(k, mu, m, u, v, I_layers, H_layers):
  H = create_H_matrix(m, u, v, H_layers)
  I = create_I_vector(m, u, v, I_layers)
  E = np.eye(m)
  O = np.zeros(m)
  tmp1 = np.linalg.inv(E + mu * (H.real.T - H.imag.T).dot(H))
  tmp2 = m * np.linalg.inv(E + mu * (H.real.T - H.imag.T).dot(H)).dot(H.real.T - H.imag.T).dot(I)
  for j in range(k):
    O = tmp1.dot(O) + tmp2
  return O


def solve_inverse_implicit2(out, dim, m, w_, stationary_defocus, mu, k):

    ext_dim = int(dim * 2 - 1)
    cut_dim_s = int(dim / 2)
    cut_dim_f = int(3 * dim / 2)

    ext_h = np.zeros((ext_dim, ext_dim, 2 * m - 1), dtype=np.complex)
    ext_out = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)

    h = generate_psf2(dim, m, w_, stationary_defocus)

    for l in range(m):
        ext_out[cut_dim_s:cut_dim_f, cut_dim_s:cut_dim_f, l] = out[:, :, l]
    for l in range(2 * m - 1):
        ext_h[:dim, :dim, l] = h[:, :, l]

    for l in range(m):
        ext_out[:, :, l] = np.fft.fft2(ext_out[:, :, l])
    for l in range(2 * m - 1):
        ext_h[:, :, l] = np.fft.fft2(ext_h[:, :, l])



    result = np.zeros((ext_dim, ext_dim, m), complex)
    for u in range(ext_dim):
        for v in range(ext_dim):
          tmp = regul_implicit(k, mu, m, u, v, ext_out, ext_h)
          for i in range(m):
            result[u][v][i] = tmp[i]

    for i in range(m):
        result[:,:,i] = np.fft.ifft2(result[:,:,i])

    return result.real


# In[ ]:




