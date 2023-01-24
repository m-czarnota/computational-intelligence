import os

import imageio
import laspy as lp
import lasio
import numpy as np
import rasterio
import pandas as pd
from matplotlib import pyplot as plt
from whitebox import WhiteboxTools
from rasterio.plot import show


IMAGES_DIR = './images'
DATA_DIR = './data'
WORK_DIR = r'D:\Programming\Python\computational-intelligence\languages-for-data-analysis\zad12\\'


def zad1():
    """
    została wybrana metoda interpolacji jako metoda najbliższych sąsiadów ze względu na swoją prostotę oraz nie długi czas oczekiwania na wynik
    """

    in_interpolation = f'swidwie_dense_cloud_crop.las'
    out_interpolation = f'swidwie_budynek_interpolation.tif'

    wbt = WhiteboxTools()
    wbt.work_dir = WORK_DIR
    wbt.lidar_nearest_neighbour_gridding(in_interpolation, out_interpolation, resolution=0.1)


def zad2():
    cloud_filename = f'swidwie_dense_cloud_crop.las'
    cloud = lp.read(cloud_filename)

    points = np.vstack((cloud.x, cloud.y, cloud.z)).transpose()
    voxel_size = 3

    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(
        ((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted = np.argsort(inverse)

    voxel_grid = {}
    grid_barycenter, grid_candidate_center, grid_min, grid_max = [], [], [], []
    last_seen = 0

    for idx, vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]]

        grid_barycenter.append(np.mean(voxel_grid[tuple(vox)], axis=0))
        grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(
            voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)], axis=0), axis=1).argmin()])

        grid_min.append(np.min(voxel_grid[tuple(vox)], axis=0))
        grid_max.append(np.max(voxel_grid[tuple(vox)], axis=0))

        last_seen += nb_pts_per_voxel[idx]

    np.savetxt(f"swidwie_center_{voxel_size}.xyz", grid_candidate_center, delimiter=";", fmt="%s")
    np.savetxt(f"swidwie_barycenter_{voxel_size}.xyz", grid_barycenter, delimiter=";", fmt="%s")

    points = pd.DataFrame(grid_barycenter, columns=['x', 'y', 'z'])

    ax = plt.axes(projection='3d')
    c = np.linspace(np.min(grid_min), np.max(grid_max), points.z.shape[0])
    sc2 = ax.scatter(points.x, points.y, points.z, c=c, s=10, marker='*', cmap="Spectral_r")

    ax.set_title('Świdwie lidar colored by calculated vals - min and max')
    ax.set_ylabel('latitude')
    ax.set_xlabel('longitude')
    ax.set_zlabel('wysokosc')

    plt.colorbar(sc2)
    plt.rcParams["figure.figsize"] = (20, 10)

    plt.savefig('zad2.png')
    # plt.show()


def zad3():
    las = lp.read('szczecin.laz')

    plt.figure()
    plt.scatter(las.xyz[:, 0], las.xyz[:, 1], s=0.1, c=las.classification, cmap="Spectral_r")
    plt.legend(np.unique(las.classification))

    plt.show()
    # plt.savefig('zad3.jpg')


    # wbt = WhiteboxTools()
    # wbt.work_dir = WORK_DIR
    #
    # wbt.lidar_thin("szczecin.laz", "szczecin_red.las", resolution=0.1, method="lowest", save_filtered=1)
    # wbt.lidar_ground_point_filter("swidwie_red.las", "szczecin_red.las", radius=1, min_neighbours=2, slope_threshold=10,
    #                               height_threshold=1, classify=2, slope_norm=0, height_above_ground=0)
    # wbt.filter_lidar_classes('szczecin_red.las', 'szczecin_red.las', exclude_cls='1')
    # wbt.lidar_remove_outliers('szczecin_red.las', 'szczecin_red_filter.las', radius=1, elev_diff=1, use_median=1,
    #                           classify=1)
    # wbt.lidar_nearest_neighbour_gridding("szczecin_red_filter.las", "szczecin.tif", parameter="elevation",
    #                                      returns="all", resolution=0.1, radius=0.1)
    #
    # raster = imageio.imread('szczecin.tif')
    # fig = plt.figure(figsize=(16, 11))
    # plt.imshow(raster)
    # plt.show()


if __name__ == '__main__':
    zad3()
