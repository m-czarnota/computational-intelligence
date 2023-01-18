import rasterio
from matplotlib import pyplot as plt
import earthpy.plot as ep
import earthpy.spatial as es
import numpy as np
from rasterio.plot import show as rasterio_show
from rasterio.warp import calculate_default_transform
from shapely.geometry import box

IMAGES_DIR = './images'


def zad1():
    image = rasterio.open(f'{IMAGES_DIR}/my_village.tiff')
    raster = image.read(1)  # read first layer
    raster[raster <= -9999] = np.nan

    # numeric model
    ep.plot_bands(raster, figsize=(20, 10), cmap="gist_earth", title='Numeric model of Lesięcin and Kąkolewice')
    plt.show()

    # shading
    hillshade = es.hillshade(raster)
    ep.plot_bands(hillshade, cbar=False, title="Cieniowanie", figsize=(20, 10))
    plt.show()

    # model of terrain
    fig, ax = plt.subplots(figsize=(20, 10))
    ep.plot_bands(raster, ax=ax, cmap="terrain", title="Terrain model of Lesięcin and Kąkolewice")
    ax.imshow(hillshade, cmap="Greys", alpha=0.5)
    plt.show()


def zad2():
    dst_crs = 'EPSG:4326'

    with rasterio.open(f'{IMAGES_DIR}/szmaragdowe_geotif.tif', 'r+') as src_read:
        transform, width, height = calculate_default_transform(src_read.crs, dst_crs, src_read.width, src_read.height,
                                                               *src_read.bounds)
        kwargs = src_read.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        lake = src_read.read()
        lake[0, 50:100, 50:100] = 0

    with rasterio.open(f'{IMAGES_DIR}/szmaragdowe_geotif.tif', 'w', **kwargs) as src_write:
        src_write.write(lake)

    with rasterio.open(f'{IMAGES_DIR}/szmaragdowe_geotif.tif', 'r+') as src_read:
        raster = src_read.read(1)

        ep.plot_bands(raster, figsize=(20, 10), cmap="gist_earth")
        plt.show()


def zad3():
    image_path = f'{IMAGES_DIR}/bukowe.tif'
    image_classified_path = f'{IMAGES_DIR}/bukowe_classified.tif'

    with rasterio.open(image_path) as raster:
        image = raster.read(1)

    image_copy = np.copy(image)
    image_copy[image > 20] = 4
    image_copy[image <= 20] = 3
    image_copy[image <= 10] = 2
    image_copy[image < 5] = 1

    with rasterio.open(image_classified_path, 'w', driver='GTiff', height=image.shape[0], width=image.shape[1], count=1, dtype=str('float64')) as writer:
        writer.write(image_copy, 1)

    with rasterio.open(image_classified_path, 'r+') as image:
        figure, ax = plt.subplots(1, figsize=(12, 10))
        rasterio_show((image, 1), cmap='Greys_r', interpolation='nearest', ax=ax)
        rasterio_show((image, 1), contour=True, ax=ax)
        plt.show()

    area_of_class = {label: np.sum(np.where(image_copy == label)) for label in range(1, 5)}
    print(f'area of classes: {area_of_class}')


if __name__ == '__main__':
    zad3()
