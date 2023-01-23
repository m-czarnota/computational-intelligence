import laspy as lp
import lasio
import numpy as np
import rasterio
import pandas as pd
from matplotlib import pyplot as plt
from whitebox import WhiteboxTools
from rasterio.warp import calculate_default_transform


IMAGES_DIR = './images'
DATA_DIR = './data'


def zad1():
    """
    została wybrana metoda interpolacji jako metoda najbliższych sąsiadów ze względu na swoją prostotę oraz nie długi czas oczekiwania na wynik
    """

    in_interpolation = f'swidwie_dense_cloud_crop.las'
    out_interpolation = f'swidwie_budynek_interpolation.tif'

    wbt = WhiteboxTools()
    wbt.work_dir = r'C:\Users\michal.czarnota\Documents\computational-intelligence\languages-for-data-analysis\zad12\\'
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
    cloud_filename = f'4720_275414_N-33-90-C-a-3-4-1-4'

    in_interpolation = f'{cloud_filename}.laz'
    out_interpolation = f'{cloud_filename}.las'

    wbt = WhiteboxTools()
    wbt.work_dir = r'C:\Users\michal.czarnota\Documents\computational-intelligence\languages-for-data-analysis\zad12\\'
    wbt.lidar_nearest_neighbour_gridding(in_interpolation, out_interpolation, resolution=0.1)

    las = lp.read(out_interpolation)
    print(las)
    cos2 = 2


if __name__ == '__main__':
    zad3()
