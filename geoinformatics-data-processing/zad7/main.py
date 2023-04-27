import laspy as lp
from matplotlib import pyplot as plt
from whitebox import WhiteboxTools

WORK_DIR = r'D:\Programming\Python\computational-intelligence\geoinformatics-data-processing\zad7\\'


def zad1():
    las = lp.read('szczecin.laz')

    plt.figure()

    plt.scatter(las.xyz[:, 0], las.xyz[:, 1], c=las.raw_classification == 2, s=0.1, label='ground')
    plt.scatter(las.xyz[:, 0], las.xyz[:, 1], c=las.raw_classification == 7, s=0.1, label='water')
    plt.scatter(las.xyz[:, 0], las.xyz[:, 1], c=las.raw_classification == 6, s=0.1, label='buildings')

    plt.scatter(las.xyz[:, 0], las.xyz[:, 1], c=las.raw_classification == 3, s=0.1, label='vegetation')
    plt.scatter(las.xyz[:, 0], las.xyz[:, 1], c=las.raw_classification == 4, s=0.1, label='vegetation')
    plt.scatter(las.xyz[:, 0], las.xyz[:, 1], c=las.raw_classification == 5, s=0.1, label='vegetation')

    plt.legend()
    # plt.show()
    plt.savefig('zad1.jpg')


def zad2():
    """
    została wybrana metoda interpolacji jako metoda najbliższych sąsiadów ze względu na swoją prostotę oraz nie długi czas oczekiwania na wynik
    """

    in_interpolation = f'swidwie_dense_cloud_crop.las'
    out_interpolation = f'swidwie_budynek_interpolation.tif'

    wbt = WhiteboxTools()
    wbt.work_dir = WORK_DIR
    wbt.lidar_nearest_neighbour_gridding(in_interpolation, out_interpolation, resolution=0.1)


if __name__ == '__main__':
    zad2()
