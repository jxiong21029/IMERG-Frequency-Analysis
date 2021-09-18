import os

import numpy as np


script_path = os.path.dirname(os.path.realpath(__file__))


def flatten(list_):
    # Flattens a list of lists, a list of numpy arrays, a 2d numpy array, etc. into a single list
    return [item for sublist in list_ for item in sublist]


def coords_to_idx(lat: float, lon: float, full_range=False):
    """Converts latitude/longitude coordinates to indexes in a 1200x3600 array (or 1800x3600 if full_range is True)"""
    if full_range:
        return (
            round((lat + 89.95) * 10),
            round((lon + 179.95) * 10)
        )
    else:
        return (
            round((lat + 59.95) * 10),
            round((lon + 179.95) * 10)
        )


def idx_to_coords(y: int, x: int, full_range=False):
    """Converts indexes in a 1200x3600 array (or 1800x3600 if full_range is True) to latitue/longitude coordinates"""
    if full_range:
        return y / 10 - 89.95, x / 10 - 179.95
    else:
        return y / 10 - 59.95, x / 10 - 179.95


def idx_fasthaver(y0, x0, y1, x1, full_range=False):
    """Calculates the haversine distance in km between two pairs of array indexes.

    The indexes are those for a 1200x3600 array (or 1800x3600 if full_range is True).
    """
    if full_range:
        lat0, lon0 = y0 / 10 - 89.95, x0 / 10 - 179.95
        lat1, lon1 = y1 / 10 - 89.95, x1 / 10 - 179.95
    else:
        lat0, lon0 = y0 / 10 - 59.95, x0 / 10 - 179.95
        lat1, lon1 = y1 / 10 - 59.95, x1 / 10 - 179.95
    lon0, lat0, lon1, lat1 = map(np.radians, [lon0, lat0, lon1, lat1])

    dlon = lon1 - lon0
    dlat = lat1 - lat0

    a = np.sin(dlat/2.0)**2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def generate_random_latlon(n, land_only=False):
    global elevation
    if elevation is None:
        elevation = np.load(os.path.join(script_path, 'data', 'elevation.npz'))['arr_0']
    """Generates random array indices"""
    rng = np.random.default_rng()
    if land_only:
        probs = np.asarray([1 if elevation[i] > 0 else 0 for i in range(1200 * 3600)])
        probs = probs / probs.sum()
        flat_idxs = rng.choice(
            np.arange(1200 * 3600),
            size=n,
            replace=False,
            p=probs
        )
    else:
        flat_idxs = rng.choice(np.arange(1200 * 3600), size=n, replace=False)
    idxs = np.concatenate([
        (flat_idxs // 3600).reshape(-1, 1),
        (flat_idxs % 3600).reshape(-1, 1)
    ], axis=1)
    return idxs


def load_nc4(filename, full_range=False):
    import netCDF4
    with netCDF4.Dataset(filename, 'r') as f:
        precip_cal = np.asarray(f.variables['precipitationCal'])[0]
    precip_cal[precip_cal == -9999.9] = np.nan
    if full_range:
        return np.transpose(precip_cal)
    return np.transpose(precip_cal)[300:1500]


def decluster(data, threshold_value=None, threshold_q=None, r=5):
    """
    Returns temporally declustered extreme values.

    Returns values exceeding the threshold_value, but values separated by less than r values below the threshold are
    considered as "within the same cluster," so only the larger one is used. See "An Introduction to the Statistical
    Modeling of Extreme Values" (https://doi.org/10.1007/978-1-4471-3675-0) page 99 for more information.
    :param data: A list or np.ndarray consisting of one full timeseries of precipitation data
    :param threshold_value: The value above which the timeseries entries are considered extreme
    :param threshold_q: The quantile used to compute threshold_value. Only one of threshold_value and threshold_q should
    be defined
    :param r: The # of nonextreme values required to begin a new cluster
    :return: A list of only declustered data that lies above the threshold
    """
    if threshold_value is None:
        if threshold_q is not None:
            threshold_value = np.nanpercentile(data, threshold_q)
        else:
            threshold_value = np.nanpercentile(data, 99)

    num_under_thresh = r
    ret = []
    max_excess = 0
    for entry in data:
        if entry <= threshold_value:
            if num_under_thresh < r:
                num_under_thresh += 1
                if num_under_thresh == r:
                    ret.append(max_excess)
                    max_excess = 0
        else:
            num_under_thresh = 0
            max_excess = max(max_excess, entry)
    if max_excess > 0:
        ret.append(max_excess)

    return ret


def lmom_ratios(x, nmom=5):
    """Computes L-moments from a sample of data x.

    See https://doi.org/10.1017/CBO9780511529443 for further explanation on L-moments and their uses in regional
    frequency analysis.
    """
    from scipy.special import comb

    try:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        x.sort()
    except ValueError:
        raise ValueError("Input data to estimate L-moments must be numeric.")

    if nmom <= 0 or nmom > 5:
        raise ValueError("Invalid number of sample L-moments")

    if n < nmom:
        raise ValueError("Insufficient length of data for specified nmoments")

    # First L-moment
    l1 = np.sum(x) / comb(n, 1, exact=True)
    if nmom == 1:
        return l1

    # Second L-moment
    comb1 = range(n)
    coefl2 = 0.5 / comb(n, 2, exact=True)
    sum_xtrans = sum([(comb1[i] - comb1[n - i - 1]) * x[i] for i in range(n)])
    l2 = coefl2 * sum_xtrans
    if nmom == 2:
        return [l1, l2]

    # Third L-moment
    comb3 = [comb(i, 2, exact=True) for i in range(n)]
    coefl3 = 1.0 / 3.0 / comb(n, 3, exact=True)
    sum_xtrans = sum([(comb3[i] - 2 * comb1[i] * comb1[n - i - 1] + comb3[n - i - 1]) * x[i] for i in range(n)])
    l3 = coefl3 * sum_xtrans / l2
    if nmom == 3:
        return [l1, l2, l3]

    # Fourth L-moment
    comb5 = [comb(i, 3, exact=True) for i in range(n)]
    coefl4 = 0.25 / comb(n, 4, exact=True)
    sum_xtrans = sum(
        [(comb5[i] - 3 * comb3[i] * comb1[n - i - 1] + 3 * comb1[i] * comb3[n - i - 1] - comb5[n - i - 1]) * x[i]
         for i in range(n)])
    l4 = coefl4 * sum_xtrans / l2
    if nmom == 4:
        return [l1, l2, l3, l4]

    # Fifth L-moment
    comb7 = [comb(i, 4, exact=True) for i in range(n)]
    coefl5 = 0.2 / comb(n, 5, exact=True)
    sum_xtrans = sum(
        [(comb7[i] - 4 * comb5[i] * comb1[n - i - 1] + 6 * comb3[i] * comb3[n - i - 1] -
          4 * comb1[i] * comb5[n - i - 1] + comb7[n - i - 1]) * x[i]
         for i in range(n)])
    l5 = coefl5 * sum_xtrans / l2
    return [l1, l2, l3, l4, l5]


prop_wet_days = mean_wet_prec = elevation = None


def weighted_knn(start_lat, start_lon, k=20):
    """Finds the k-nearest-neighbor grid cells from some starting cell taking into account four site characteristics.

    After normalizing each site characteristic (based on the standard deviation of the positive difference between two
    points within 200 km, using the same process used for calculating homogeneity correlations), each variable is then
    multiplied by some weight. The set of weights was found using coordinate ascent to maximize average cluster
    homogeneity. The nearest k points from within 200 km of the original point based on the euclidean distance using the
    four variables are returned, along with the values of these euclidean distances.
    """
    global prop_wet_days, mean_wet_prec, elevation
    if prop_wet_days is None:
        prop_wet_days = np.load(os.path.join(script_path, 'data', 'prop_wet_days.npz'))['arr_0']
    if mean_wet_prec is None:
        mean_wet_prec = np.load(os.path.join(script_path, 'data', 'mean_wet_prec.npz'))['arr_0']
    if elevation is None:
        elevation = np.load(os.path.join(script_path, 'data', 'elevation.npz'))['arr_0']

    std = [46.657784, 0.043044, 1.395474, 417.914331]
    weights = [1, 1.8, 1.2, 0.8]

    lats = start_lat + np.arange(max(-18, -start_lat), min(19, 1200 - start_lat))
    lons = (start_lon + np.arange(-36, 37)) % 3600
    mlats, mlons = np.meshgrid(lats, lons, indexing='ij')
    mlats, mlons = mlats.reshape(-1), mlons.reshape(-1)

    dists = idx_fasthaver(start_lat, start_lon, mlats, mlons)

    clust_data = np.zeros((len(mlats), 4))
    clust_data[:, 0] = dists
    clust_data[:, 1] = prop_wet_days[mlats, mlons] - prop_wet_days[start_lat, start_lon]
    clust_data[:, 2] = mean_wet_prec[mlats, mlons] - mean_wet_prec[start_lat, start_lon]
    clust_data[:, 3] = elevation[mlats, mlons] - elevation[start_lat, start_lon]

    clust_data_scaled = clust_data * weights / std

    idxs = sorted(range(len(mlats)), key=lambda i: (clust_data_scaled[i] ** 2).sum())
    idxs = np.asarray([idx for idx in idxs if clust_data[idx, 0] < 200])[:k]
    return np.concatenate((mlats[idxs].reshape(-1, 1), mlons[idxs].reshape(-1, 1)), axis=1), \
        [(clust_data_scaled[idx] ** 2).sum() for idx in idxs]


def ari_to_precip(years, params_filename, threshold_q=98):
    """Converts an average recurrence interval (ARI) in years to a global array of estimated precipitation in mm.

    The given ARI should be large enough so that such events are rarer than events that lie above the threshold (i.e.
    greater than one, at least). Currently, returns only values for one day precipitation accumulations. Utilizes
    precomputed GP parameters from a file (which should have been computed with clustering). The ARI computation
    procedure roughly follows the regional L-moment algorithm presented in https://doi.org/10.1017/CBO9780511529443,
    except the index flood is taken as the value of the threshold_q percentile of the data (or 1, if smaller).

    :param years: The length of the ARI interval, in years, for which to estimate precipitation accumulations.
    :param params_filename: The name of the file containing precomputed GP parameters produced by fit_params.py.
    :param threshold_q: The percentile of the data used as the GP threshold.
    :return: A 1200x3600 array for the 60S to 60N range containing precipitation estimates for the given ARI.
    """
    if threshold_q == 98:
        idx3 = 0
    elif threshold_q == 99:
        idx3 = 1
    elif threshold_q == 99.5:
        idx3 = 2
    else:
        raise ValueError(f'threshold_q must be 95, 98, or 99')
    fitted_params = np.load(params_filename)['arr_0']
    idx_floods = np.load('data/index_floods.npz')['arr_0'][:, :, idx3]

    shapes = fitted_params[:, :, idx3, 0]
    scales = fitted_params[:, :, idx3, 1]
    precip_for_scaled = 1 + scales / shapes * (
        (years * 365.25 * (1 - threshold_q / 100)) ** shapes - 1
    )
    aris = precip_for_scaled * idx_floods
    return aris


def precip_to_ari(precip, params_filename, threshold_q=98):
    """Converts a global array of estimated precipitation in mm to their corresponding ARI intervals, in years.

    Currently, returns only values for one day precipitation accumulations. Utilizes precomputed GP parameters from a
    file (which should have been computed with clustering). The ARI computation procedure roughly follows the regional
    L-moment algorithm presented in https://doi.org/10.1017/CBO9780511529443, except the index flood is taken as the
    value of the threshold_q percentile of the data (or 1, if smaller).

    :param precip: A 1200x3600 array containing precipitation estimates for which to estimate ARIs.
    :param params_filename: The name of the file containing precomputed GP parameters produced by fit_params.py.
    :param threshold_q: The percentile of the data used as the GP threshold.
    :return: A 1200x3600 array of estimated ARI values.
    """
    if threshold_q == 98:
        idx3 = 0
    elif threshold_q == 99:
        idx3 = 1
    elif threshold_q == 99.5:
        idx3 = 2
    else:
        raise ValueError(f'threshold_q must be 95, 98, or 99')
    fitted_params = np.load(params_filename)['arr_0']
    idx_floods = np.load('data/index_floods.npz')['arr_0'][:, :, idx3]

    shapes = fitted_params[:, :, idx3, 0]
    scales = fitted_params[:, :, idx3, 1]
    return (
        (shapes / scales * (precip / idx_floods - 1) + 1) ** (1 / shapes)
    ) / 365.25 / (1 - threshold_q / 100)


# Homogeneity testing, requires rpy2. Code may need some updates.

# # For rpy2 installation in Windows, I used https://jianghaochu.github.io/how-to-install-rpy2-in-windows-10.html
# from rpy2 import robjects
# from rpy2.robjects.packages import importr
# importr('homtest')
# importr('extRemes')
#
# def homogeneity_hw(dataset: List[np.ndarray], nsim=500) -> float:
#     """
#     Compute the first Hosking-Wallace heterogeneity statistic for a region containing multiple sites.
#
#     See https://doi.org/10.1029/2006WR005095 for an explanation of this homogeneity test.
#     :param dataset: A list where each element is a np.ndarray timeseries of observations (possibly only extreme ones)
#     :param nsim: The # of simulations to run when computing the statistic
#     :return: The value of the statistic: the region is 'acceptably homogeneous' if the value < 1, 'possibly
#     heterogeneous' if 1 <= the value < 2, and 'definitely heterogeneous' if the value >= 2.
#     """
#     flat = flatten(dataset)
#     cod = []
#     for i, series in enumerate(dataset):
#         cod.extend([i] * len(series))
#     return robjects.r['HW.tests'](robjects.FloatVector(flat), robjects.IntVector(cod), Nsim=nsim)[0]
#
#
# def homogeneity_ad(dataset: List[np.ndarray], nsim=500) -> float:
#     """
#     Compute the p-value given by the bootstrap k-sample Anderson-Darling test.
#
#     See https://doi.org/10.1029/2006WR005095 for an explanation of this homogeneity test.
#     :param dataset: A list where each element is a np.ndarray timeseries of observations
#     :param nsim: The # of simulations to run when computing the statistic
#     :return: The p-value computed by the test. Lower is better; reject the null hypothesis when this value > (1 - a)
#     """
#     flat = flatten(dataset)
#     cod = []
#     for i, series in enumerate(dataset):
#         cod.extend([i] * len(series))
#     return robjects.r['ADbootstrap.test'](robjects.FloatVector(flat), robjects.IntVector(cod), Nsim=nsim)[1]
#
#
# def homogeneity_joint(dataset: List[np.ndarray], nsim=500, return_all=False):
#     """
#     Combines the Hosking-Wallace and Anderson-Darling tests based on https://doi.org/10.1029/2006WR005095.
#
#     :param dataset: A list where each element is a np.ndarray timeseries of observations
#     :param nsim: The # of simulations to run when computing the statistic
#     :param return_all: If True, returns the values of both tests preceded by the index of the recommended one
#     """
#     flat = flatten(dataset)
#     cod = []
#     for i, series in enumerate(dataset):
#         cod.extend([i] * len(series))
#     if return_all:
#         return (
#             robjects.r['HW.tests'](robjects.FloatVector(flat), robjects.IntVector(cod), Nsim=nsim)[0],
#             robjects.r['ADbootstrap.test'](robjects.FloatVector(flat), robjects.IntVector(cod), Nsim=nsim)[1],
#             0 if robjects.r['regionalLmoments'](robjects.FloatVector(flat), robjects.IntVector(cod))[3] < 0.23 else 1
#         )
#     if robjects.r['regionalLmoments'](robjects.FloatVector(flat), robjects.IntVector(cod))[3] < 0.23:
#         return robjects.r['HW.tests'](robjects.FloatVector(flat), robjects.IntVector(cod), Nsim=nsim)[0]
#     return robjects.r['ADbootstrap.test'](robjects.FloatVector(flat), robjects.IntVector(cod), Nsim=nsim)[1]
