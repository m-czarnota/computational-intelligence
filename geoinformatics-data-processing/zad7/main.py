from osgeo import gdal
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.plot import show
from matplotlib import colors, cm
import pylas
import whitebox
import laspy as lp


def zad1() -> None:

    dem = gdal.Open('puszcza_bukowa_szmaragdowe.asc')
    band = dem.GetRasterBand(1)
    height = band.ReadAsArray()
    min_val = np.min(height)

    gt = dem.GetGeoTransform()
    ulx = gt[0]
    uly = gt[3]
    res = gt[1]

    x_size = dem.RasterXSize
    y_size = dem.RasterYSize

    lrx = ulx + x_size * res
    lry = uly - y_size * res

    dem = None

    raster = gdal.Grid(
        'zad1.tif',
        'puszcza_bukowa_szmaragdowa_lowose_punkty.shp',
        zfield='VALUE',
        algorithm=f'invdist:radius1=40:radius2=40:smoothing=5:nodata={min_val}',
        outputBounds=[ulx, uly, lrx, lry],
        width=x_size,
        height=y_size,
    )
    raster = None

    src = rasterio.open('zad1.tif')
    data = src.read()

    fig, ax = plt.subplots(1, figsize=(20, 10))
    cmap = plt.get_cmap('rainbow')

    show(data, transform=src.transform, ax=ax, cmap=cmap)
    fig.colorbar(cm.ScalarMappable(
        norm=colors.Normalize(
            vmin=np.nanmin(data),
            vmax=np.nanmax(data),
        ),
        cmap=cmap,
    ), ax=ax)
    src = None


if __name__ == '__main__':
    zad1()
