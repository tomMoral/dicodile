import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import fftconvolve


if __name__ == "__main__":
    n_atoms = 25
    valid_shape = (50, 50)
    atom_shape = (12, 12)

    ztz_shape = (n_atoms, n_atoms) + tuple([
        2 * size_atom_ax - 1 for size_atom_ax in atom_shape
    ])

    z = np.random.randn(n_atoms, *valid_shape)
    z *= np.random.rand(*z.shape) > .9
    padding_shape = [(0, 0)] + [
        (size_atom_ax - 1, size_atom_ax - 1) for size_atom_ax in atom_shape]
    padding_shape = np.asarray(padding_shape, dtype='i')
    z_pad = np.pad(z, padding_shape, mode='constant')

    t_start = time.time()
    ztz = np.empty(ztz_shape)
    for i in range(ztz.size):
        i0 = k0, k1, *pt = np.unravel_index(i, ztz.shape)
        zk1_slice = tuple([k1] + [
            slice(v, v + size_ax) for v, size_ax in zip(pt, valid_shape)])
        ztz[i0] = np.dot(z[k0].ravel(), z_pad[zk1_slice].ravel())
    print("A la mano: {:.3f}s".format(time.time() - t_start))

    # compute the cross correlation between z and z_pad
    t_fft = time.time()
    flip_axis = tuple(range(1, z.ndim))
    ztz_fft = np.array([[fftconvolve(z_pad_k0, z_k, mode='valid')
                         for z_k in z]
                        for z_pad_k0 in np.flip(z_pad, axis=flip_axis)])
    print("FFT: {:.3f}s".format(time.time() - t_fft))
    assert ztz_fft.shape == ztz_shape, (ztz.shape, ztz_shape)
    plt.imshow((ztz - ztz_fft).reshape(25*25, 23*23))
    plt.show()
    assert np.allclose(ztz, ztz_fft), abs(ztz - ztz_fft).max()

    # Sparse the cross correlation between z and z_pad
    t_sparse = time.time()
    ztz_sparse = np.zeros(ztz_shape)
    for k0, *pt in zip(*z.nonzero()):
        z_pad_slice = tuple([slice(None)] + [
            slice(v, v + 2 * size_ax - 1)
            for v, size_ax in zip(pt, atom_shape)])
        ztz_sparse[k0] += z[(k0, *pt)] * z_pad[z_pad_slice]
    print("Sparse: {:.3f}s".format(time.time() - t_sparse))
    assert np.allclose(ztz_sparse, ztz), abs(ztz_sparse - ztz).max()
