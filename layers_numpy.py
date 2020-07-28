import numpy as np


def quantize_unsigned(x, num_bits, max_value):
    y = np.maximum(np.minimum(x, max_value), 0.0)
    y = np.round(y / max_value * (2**num_bits-1))
    return y


def quantize_rescale(x, num_bits, max_value):
    return x / (2**num_bits-1) * max_value



def conv(x, w, b, stride=1, pad=(0,0)):
  N, H, W, C = x.shape
  HH, WW, _, F = w.shape
  HC = int(1 + (H + pad[0] + pad[1] - HH) / stride)
  WC = int(1 + (W + pad[0] + pad[1] - WW) / stride)
  
  out = np.zeros([N,HC,WC,F])
  x_pad = np.pad(x, ((0,0),(pad[0],pad[1]),(pad[0],pad[1]),(0,0)), mode='constant')
  for i, xi in enumerate(x_pad):
    for f in range(F):
        wi = w[:, :, :, f]
        for row in range(HC):
            for column in range(WC):
                tmp = xi[stride*row : stride*row+HH, stride*column : stride*column+WW, :]
                out[i, row, column, f] = tmp.flatten().dot(wi.flatten()) + b[f]
  return out


def max_pool(x, pool_size, stride=None, pad=(0,0)):
  HH = pool_size
  WW = pool_size
  if stride is None:
    stride = pool_size

  x = np.pad(x, ((0,0),(pad[0],pad[1]),(pad[0],pad[1]),(0,0)), mode='constant')
  N, H, W, C = x.shape
  HP = int((H - HH) / stride + 1)
  WP = int((W - WW) / stride + 1)

  out = np.zeros([N,HP,WP,C])
  for i, xi in enumerate(x):
    for c in range(C):
        ci = xi[:,:,c]
        for h in range(HP):
            for w in range(WP):
                out[i,h,w,c] = np.max(ci[h*stride:h*stride+HH, w*stride:w*stride+WW])
  return out


def avg_pool_flatten(x):
  return np.mean(x, axis=(1,2))


def relu(x):
  return x * (x>0)


def flatten(x):
    N, H, W, C = x.shape
    return x.reshape(N, -1)


def dense(x, w, b):
  out = x.dot(w) + b
  return out


def batch_normalization(x, gamma, beta, mean, variance, eps=0.001):
	out = (x - mean) / np.sqrt(variance + eps)
	return out * gamma + beta


def merge_conv_batchnorm(w, b, gamma, beta, mean, variance, eps=0.001):
	std = np.sqrt(variance + eps)
	w_scaling = gamma / std
	W_folded = W * w_scaling
	b_folded = (b - mean) * w_scaling + beta
	return (W_folded, b_folded)
	