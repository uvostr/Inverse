import numpy as np
cimport numpy as np
import math
from cython import boundscheck, wraparound
from libc.math cimport cos, sin, signbit
from cython.parallel cimport prange, parallel


ctypedef double real_t
ctypedef double complex complex_t


cpdef real_t m_func(real_t x, real_t y, real_t R, real_t n_coef):
    cdef:
        real_t x_n = x * n_coef
        real_t y_n = y * n_coef
    if x_n * x_n + y_n * y_n <= R * R:
        return 1
    else:
        return 0

cpdef real_t p_func(real_t x, real_t y, real_t a):
    return a * (x * x + y * y)


@boundscheck(False)
@wraparound(False)
cpdef generate_psf(int dim, int m, real_t w_):
    cdef:
        
        w = w_ * 2.34 * 0.001

        real_t f = 20 * 0.001
        real_t wavelength = 0.55 * 0.000001
        real_t d_1 = 57.4 * 0.001
        real_t d_0 = 37 * 0.001
        real_t r_0 = 4.5 * 0.001

        real_t lambda_d1 = wavelength * d_1
        real_t a = 2 * r_0 / lambda_d1
        real_t dzeta = 1.0 * w


        real_t ps_value = - np.pi * wavelength * d_1 * d_1 * dzeta / ((d_0 + w) * (d_0 + w))
        real_t n_coef = lambda_d1

        int dim_x = dim
        int dim_y = dim
        real_t min_x = -a
        real_t max_x = a
        real_t min_y = -a
        real_t max_y = a
        real_t range_x = max_x - min_x
        real_t range_y = max_y - min_y
        real_t R = r_0

        real_t delta_x = (max_x - min_x) / dim_x
        real_t delta_xi = 1.0 / dim_x / delta_x
        real_t dim_delta_xi = delta_xi * dim

        np.ndarray arg_x = np.linspace(start=min_x, stop=max_x, num=dim_x)
        np.ndarray arg_y = np.linspace(start=min_y, stop=max_y, num=dim_y)
        np.ndarray inner_h = np.zeros((dim_x, dim_y), dtype=np.cdouble)
        np.ndarray fft_inner_h = np.zeros((dim_x, dim_y), dtype=np.cdouble)
        real_t [:] x = arg_x
        real_t [:] y = arg_y
        complex_t [:,:] inner_h_view = inner_h
        complex_t [:,:] fft_inner_h_view = fft_inner_h

    cdef h = np.zeros((dim_x, dim_y, 2 * m - 1), dtype=np.double)

    for l in range(2 * m - 1):
        layer = l - m + 1
        for i in range(dim_x):
            for j in range(dim_y):
                temp = p_func(x[i], y[j], ps_value * (layer) / (m-1)) + p_func(x[i], y[j], ps_value * 0.05)
                inner_h_view[i, j] = m_func(x[i], y[j], R, n_coef) * (cos(temp) + 1j * sin(temp))

        fft_inner_h = np.fft.fft2(inner_h)
        fft_inner_h = np.fft.fftshift(fft_inner_h)

        fft_inner_h = np.abs(fft_inner_h)
        fft_inner_h = fft_inner_h ** 2
        h[:, :, l] = np.real(fft_inner_h)

    h_max = np.amax(h)
    h = h / h_max

    return h


@boundscheck(False)
@wraparound(False)
cpdef generate_out_images(int dim, int m, real_t w_, np.ndarray[double, ndim=3] src):

    cdef int ext_dim = dim * 2 - 1
    cdef int cut_dim_s = dim / 2
    cdef int cut_dim_f = 3 * dim / 2
    cdef np.ndarray h = np.zeros((dim, dim, 2 * m - 1), dtype=np.double)
    cdef np.ndarray out = np.zeros((dim, dim, m), dtype=np.double)
    h = generate_psf(dim, m, w_)
    cdef np.ndarray ext_h = np.zeros((ext_dim, ext_dim, 2 * m - 1), dtype=np.complex)
    cdef np.ndarray ext_src = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)
    cdef np.ndarray ext_out = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)

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
            ext_out[:, :, k] = ext_out[:, :, k] + \
                               ext_src[:, :, l] * ext_h[:, :, l - k + m - 1]

    for l in range(m):
        ext_out[:, :, l] = np.fft.ifft2(ext_out[:, :, l])


    for l in range(m):
        out[:, :, l] = np.real(ext_out[cut_dim_s:cut_dim_f, cut_dim_s:cut_dim_f, l])

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


@boundscheck(False)
@wraparound(False)
def solve_inverse_implicit(out, dim, m, w_, mu, k):
    h = generate_psf(dim, m, w_)


    cdef int ext_dim = dim * 2 - 1
    cdef int cut_dim_s = dim / 2
    cdef int cut_dim_f = 3 * dim / 2

    cdef np.ndarray ext_h = np.zeros((ext_dim, ext_dim, 2 * m - 1), dtype=np.complex)
    cdef np.ndarray ext_out = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)

    h = generate_psf(dim, m, w_)

    for l in range(m):
        ext_out[cut_dim_s:cut_dim_f, cut_dim_s:cut_dim_f, l] = out[:, :, l]
    for l in range(2 * m - 1):
        ext_h[:dim, :dim, l] = h[:, :, l]

    for l in range(m):
        ext_out[:, :, l] = np.fft.fft2(ext_out[:, :, l])
    for l in range(2 * m - 1):
        ext_h[:, :, l] = np.fft.fft2(ext_h[:, :, l])



    cdef np.ndarray result = np.zeros((ext_dim, ext_dim, m), complex)
    for u in range(ext_dim):
        for v in range(ext_dim):
          tmp = regul_implicit(k, mu, m, u, v, ext_out, ext_h)
          for i in range(m):
            result[u][v][i] = tmp[i]

    for i in range(m):
        result[:,:,i] = np.fft.ifft2(result[:,:,i])

    return result.real



def regul_explicit(k, mu, m, u, v, I_layers, H_layers):
  H = create_H_matrix(m, u, v, H_layers)
  I = create_I_vector(m, u, v, I_layers)
  E = np.eye(m)
  O = np.zeros(m)
  tmp1 = E - mu * (H.real.T - H.imag.T).dot(H)
  tmp2 = mu * (H.real.T - H.imag.T).dot(I)
  for j in range(k):
    O = tmp1.dot(O) + tmp2
  return O

@boundscheck(False)
@wraparound(False)
def solve_inverse_explicit(out, dim, m, w_, mu, k):
    h = generate_psf(dim, m, w_)
    


    cdef int ext_dim = dim * 2 - 1
    cdef int cut_dim_s = dim / 2
    cdef int cut_dim_f = 3 * dim / 2

    cdef np.ndarray ext_h = np.zeros((ext_dim, ext_dim, 2 * m - 1), dtype=np.complex)
    cdef np.ndarray ext_out = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)

    h = generate_psf(dim, m, w_)

    for l in range(m):
        ext_out[cut_dim_s:cut_dim_f, cut_dim_s:cut_dim_f, l] = out[:, :, l]
    for l in range(2 * m - 1):
        ext_h[:dim, :dim, l] = h[:, :, l]

    for l in range(m):
        ext_out[:, :, l] = np.fft.fft2(ext_out[:, :, l])
    for l in range(2 * m - 1):
        ext_h[:, :, l] = np.fft.fft2(ext_h[:, :, l])



    cdef np.ndarray result = np.zeros((ext_dim, ext_dim, m), complex)
    for u in range(ext_dim):
        for v in range(ext_dim):
          tmp = regul_explicit(k, mu, m, u, v, ext_out, ext_h)
          for i in range(m):
            result[u][v][i] = tmp[i]

    for i in range(m):
        result[:,:,i] = np.fft.ifft2(result[:,:,i])

    return result.real


@boundscheck(False)
@wraparound(False)
def rings_array(dim):
  rings_array = np.zeros((dim * 2 - 1, dim * 2 - 1), int)
  for u in range(dim * 2 - 1):
    for v in range(dim * 2 - 1):
      tmp = np.sqrt((u - dim + 1) ** 2 + (v - dim + 1) ** 2)
      rings_array[u][v] = int(math.floor(tmp))
  return rings_array


@boundscheck(False)
@wraparound(False)
def solve_inverse_implicit_split(out, dim, m, w_, mu1, k1, mu2, k2, r):
    h = generate_psf(dim, m, w_)
    
    rings = rings_array(dim)

    cdef int ext_dim = dim * 2 - 1
    cdef int cut_dim_s = dim / 2
    cdef int cut_dim_f = 3 * dim / 2

    cdef np.ndarray ext_h = np.zeros((ext_dim, ext_dim, 2 * m - 1), dtype=np.complex)
    cdef np.ndarray ext_out = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)

    h = generate_psf(dim, m, w_)

    for l in range(m):
        ext_out[cut_dim_s:cut_dim_f, cut_dim_s:cut_dim_f, l] = out[:, :, l]
    for l in range(2 * m - 1):
        ext_h[:dim, :dim, l] = h[:, :, l]

    for l in range(m):
        ext_out[:, :, l] = np.fft.fft2(ext_out[:, :, l])
    for l in range(2 * m - 1):
        ext_h[:, :, l] = np.fft.fft2(ext_h[:, :, l])



    cdef np.ndarray result = np.zeros((ext_dim, ext_dim, m), complex)
    for u in range(ext_dim):
        for v in range(ext_dim):
          if(rings[u][v] >= r):
              tmp = regul_implicit(k2, mu2, m, u, v, ext_out, ext_h)
          else:
              tmp = regul_implicit(k1, mu1, m, u, v, ext_out, ext_h)
          for i in range(m):
            result[u][v][i] = tmp[i]

    for i in range(m):
        result[:,:,i] = np.fft.ifft2(result[:,:,i])

    return result.real


def rings_array_VFC(dim):
  #радиус вписанного кольца в котором находится точка, -1 если не лежит во вписанном кольце\ 1 если лежит в двух кольцах(полный квадрат), 0 иначе
  rings = np.zeros((dim * 2 - 1, dim * 2 - 1, 2), int)
  for u in range(dim * 2 - 1):
    for v in range(dim * 2 - 1): 
      tmp = np.sqrt((u - dim + 1) ** 2 + (v - dim + 1) ** 2)
      if(tmp <= dim):
        if(tmp % 1 == 0):
          rings[u][v][1] = 1
        rings[u][v][0] = int(math.floor(tmp))
      else:
        rings[u][v][0] = -1
  return rings

def VFC(original, recovered, rings, dim):
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
  res = np.zeros((4, dim)) #радиус, сумма, количество, сумма/количество
  res[0] = np.arange(0, dim, 1)
  for u in range(dim * 2 - 1):
    for v in range(dim * 2 - 1):
        j = rings[u][v][0]
        if(j > 0):
          if(rings[u][v][1] == 1):
            res[1][j - 1] += tmp[u][v]
            res[2][j - 1] += 1
          res[1][j] += tmp[u][v]
          res[2][j] += 1
  for i in range(dim):
    if(res[2][i] > 0):
      res[3][i] = res[1][i] / res[2][i]
    else:
      res[3][i] = 0
  x_plot = res[0]
  y_plot = res[3]
  return x_plot, y_plot

def recovery_quality(original, recovered, dim):
  extended_o = np.zeros((dim * 2 - 1, dim * 2 - 1))
  extended_o[:dim, :dim] = original
  rings = rings_array_VFC(dim)
  x_plot, y_plot = VFC(extended_o, recovered, rings, dim)
  return x_plot, y_plot



