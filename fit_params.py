import multiprocessing

import numpy as np
import tqdm

import modeling


NUM_PROCESSES = 4


def gp_lmom_fit(lmoments):
    t3 = lmoments[2]
    if lmoments[1] <= 0 or abs(t3) >= 1:
        raise ValueError('Invalid L-moments')

    g = (1 - 3 * t3) / (1 + t3)

    shape = -g
    scale = (1 + g) * (2 + g) * lmoments[1]
    loc = lmoments[0] - scale / (1 + g)
    return shape, loc, scale


def process_region(ii):
    topography = np.load('data/elevation.npy')
    lmoments = np.load('data/lmoments.npz')['arr_0']
    index_floods = np.load('data/index_floods.npz')['arr_0']
    params = np.zeros((100, 3600, 3, 2))

    r = tqdm.trange if ii % NUM_PROCESSES == NUM_PROCESSES - 1 else range
    for lat in r(100 * ii, 100 * (ii + 1)):
        for lon in range(3600):
            if topography[lat, lon] <= 0:
                continue
            pts, rads = modeling.weighted_knn(lat, lon, k=20)

            clust_lmoms = lmoments[pts[:, 0], pts[:, 1]]
            for i, pt in enumerate(pts):
                clust_lmoms[i, :, 0:2] /= index_floods[pt[0], pt[1]].reshape(3, 1)
            avg_lmoms = clust_lmoms.mean(axis=0)
            for i in range(3):
                shape, loc, scale = gp_lmom_fit(avg_lmoms[i])
                params[lat % 100, lon, i] = shape, scale
    np.save(f'data/fitted_params_{ii}.npy', params)


def extract_index_floods():
    # loc = l1 - (2 + k) * l2
    lmoms = np.load('data/lmoments.npz')['arr_0']
    l1 = lmoms[:, :, :, 0]
    l2 = lmoms[:, :, :, 1]
    t3 = lmoms[:, :, :, 2]
    shape = (1 - 3*t3) / (1 + t3)
    loc = l1 - (2 + shape) * l2
    loc[loc < 1] = 1
    np.savez_compressed('data/index_floods.npz', loc)


if __name__ == '__main__':
    # extract_index_floods()

    with multiprocessing.Pool(NUM_PROCESSES) as p:
        p.map(process_region, range(12))
