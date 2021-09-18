import matplotlib.pyplot as plt
import matplotlib.colors

import modeling
from plotting import plot_clust_vars


def zhengzhou_clust_plots():
    start_lat, start_lon = modeling.coords_to_idx(34.7466, 113.6253)
    pts = modeling.weighted_knn(start_lat, start_lon, 'fixedcustom_1.8_1.2_0.8')[0]

    plot_clust_vars(
        ['wetprop', 'avgwetprec', 'topo'],
        llcrnry=start_lat - 15,
        llcrnrx=start_lon - 15,
        urcrnry=start_lat + 15,
        urcrnrx=start_lon + 15,
        cluster_points=pts,
        cluster_center=pts[0]
    )


# Outdated code, provided so it may act as a reference for creating goodness-of-
# fit plots, such as probability density comparison plots and/or QQ plots.

# def zhengzhou_fit_quality(q):
#     start_lat, start_lon = modeling.coords_to_idx(34.7466, 113.6253)
#     pts, rads = modeling.weighted_knn(start_lat, start_lon, k=20)
#     # for i, rad in enumerate(rads):
#     #     if rad > 1:
#     #         pts = pts[:i]
#     #         break
#
#     dataset = []
#     for pt in pts:
#         dataset.append(
#             np.loadtxt(f'D:/IMERG_TIMESERIES_PYTHON_GROUPED2/Lon{pt[1]}/Lat{pt[0] + 300}.txt')
#         )
#     idx_floods = [np.nanpercentile(series, q) for series in dataset]
#     decl_data = [modeling.decluster(series, threshold_value=idx_floods[i]) for i, series in enumerate(dataset)]
#     scaled_data = [decl_data[i] / idx_floods[i] for i in range(len(idx_floods))]
#
#     modeler = modeling2.RegionalGpModel(dataset, q=q)  # very outdated code
#     x = np.linspace(modeler.loc, max(modeling.flatten(scaled_data)), 100)
#     y = genpareto.pdf(x, modeler.shape, loc=modeler.loc, scale=modeler.scale)
#
#     fig, axes = plt.subplots(ncols=2, constrained_layout=True)
#     axes[0].set_title('Comparison of Probability Density')
#     axes[0].hist(
#         modeling.flatten(scaled_data),
#         density=True,
#         bins='doane'
#     )
#     axes[0].plot(x, y)
#
#     quantiles = np.linspace(0, 99, 100)
#     xx = genpareto.ppf(quantiles / 100, modeler.shape, loc=modeler.loc, scale=modeler.scale)
#     yy = [np.nanpercentile(modeling.flatten(scaled_data), quantile) for quantile in quantiles]
#
#     axes[1].set_title('QQ Plot')
#     axes[1].set_xlabel('Theoretical Quantiles')
#     axes[1].set_ylabel('Data Quantiles')
#     axes[1].scatter(xx, yy, facecolors='none', edgecolors='b')
#     axes[1].plot(np.linspace(min(xx), max(xx), 100), np.linspace(min(xx), max(xx), 100), color='k')
#
#     plt.show()


def zhengzhou_ari_maps(params_filename):
    lat, lon = modeling.coords_to_idx(34.7466, 113.6253)
    ari20_to_precip = modeling.ari_to_precip(20, params_filename)[lat-15:lat+15, lon-15:lon+15]
    ari100_to_precip = modeling.ari_to_precip(100, params_filename)[lat-15:lat+15, lon-15:lon+15]

    july20_global_data = modeling.load_nc4('data/zhengzhou.nc4')
    july20_to_ari = modeling.precip_to_ari(july20_global_data, params_filename)[lat-15:lat+15, lon-15:lon+15]

    fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)

    axes[0, 0].set_title('Precip (mm) for ARI=20 years')
    im = axes[0, 0].imshow(ari20_to_precip, origin='lower', vmin=0, vmax=420)
    fig.colorbar(im, ax=axes[0, 0])

    axes[0, 1].set_title('Precip (mm) for ARI=100 years')
    im = axes[0, 1].imshow(ari100_to_precip, origin='lower', vmin=0, vmax=420)
    fig.colorbar(im, ax=axes[0, 1])

    july20_region_data = july20_global_data[lat-15:lat+15, lon-15:lon+15]
    axes[1, 0].set_title('Precip (mm) on July 20th')
    im = axes[1, 0].imshow(july20_region_data, origin='lower', vmin=0, vmax=420)
    fig.colorbar(im, ax=axes[1, 0])

    axes[1, 1].set_title('ARI (years, log scale) for July 20th Precip')
    im = axes[1, 1].imshow(july20_to_ari, origin='lower', norm=matplotlib.colors.LogNorm())
    fig.colorbar(im, ax=axes[1, 1])

    plt.show()


if __name__ == '__main__':
    zhengzhou_ari_maps('data/fitted_params.npz')
