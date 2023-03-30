import os
import subprocess
import rasterio
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy as np
from matplotlib import pyplot as plt
from pyproj import Transformer
import geopandas as gpd

wilderness = './puszcza_bukowa_szmaragdowe.asc'


def zad1() -> None:
    gdal.GetDriverByName('AAIGrid')
    nmt = gdal.Open(wilderness)
    band = nmt.GetRasterBand(1)
    data = band.ReadAsArray()
    data = np.delete(data, np.where((data >= -9999) & (data <= -10))[0], axis=0)
    print(data)

    data_min = np.min(data)
    data_max = np.max(data)

    # classification every 5 m
    for i in np.arange(data_min, data_max, 5):
        data[np.where((i <= data) & (data < i + 5))] = (i // 5) - 7

    output_file = 'mapa1.tif'
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_file, nmt.RasterXSize, nmt.RasterYSize, 1, gdal.GDT_Int16)
    output_ds.SetProjection('EPSG:2180')
    output_ds.GetRasterBand(1).WriteArray(data)
    output_ds.FlushCache()

    plt.figure()
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.gist_earth)
    plt.colorbar()
    plt.show()


def zad2_a() -> None:
    file_path = 'mapa1.tif'
    ds = gdal.Open(file_path)
    data = ds.ReadAsArray()
    geotransform = ds.GetGeoTransform()
    proj = ds.GetProjection()

    rows = ds.RasterYSize
    cols = ds.RasterXSize

    x_min = geotransform[0]
    y_max = geotransform[3]

    x_max = x_min + geotransform[1] * cols
    y_min = y_max + geotransform[5] * rows

    x_res = (x_max - x_min) / cols
    y_res = (y_max - y_min) / rows

    x_coords = np.arange(x_min + x_res / 2, x_max, x_res)
    y_coords = np.arange(y_max - y_res / 2, y_min, -y_res)

    xx, yy = np.meshgrid(x_coords, y_coords)
    transformer = Transformer.from_crs(proj, 'EPSG:32633')
    lon, lat = transformer.transform(xx, yy)

    driver = ogr.GetDriverByName("GPKG")
    output_file = 'szmaragdowe.gpkg'

    if os.path.exists(output_file):
        driver.DeleteDataSource(output_file)

    ds_out = driver.CreateDataSource(output_file)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32633)

    layer_name = "points"
    layer_out = ds_out.CreateLayer(layer_name, srs, ogr.wkbPoint)
    field_name = ogr.FieldDefn("id", ogr.OFTInteger)
    layer_out.CreateField(field_name)

    for i in range(rows):
        for j in range(cols):
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(lon[i, j], lat[i, j])
            feature = ogr.Feature(layer_out.GetLayerDefn())
            feature.SetGeometry(point)
            feature.SetField("id", i * cols + j)
            layer_out.CreateFeature(feature)


def zad2_b() -> None:
    gdal.GetDriverByName('GTiff')
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(3857)

    nmt = gdal.Open("mapa1.tif")
    band = nmt.GetRasterBand(1)

    data = band.ReadAsArray()
    max_val = np.amax(data)
    min_val = np.amin(data)

    ogr_ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource('contours.shp')  # driver OGR
    contour_shp = ogr_ds.CreateLayer('contour', sr)

    field_defn = ogr.FieldDefn("ID", ogr.OFTInteger)
    contour_shp.CreateField(field_defn)
    field_defn = ogr.FieldDefn("elev", ogr.OFTInteger)
    contour_shp.CreateField(field_defn)

    # Generate Contourlines
    intervals = list(range(int(min_val), int(max_val), 2))
    gdal.ContourGenerate(
        nmt.GetRasterBand(1),  # Band srcBand
        1,  # double contourInterval - This defines contour intervals
        0,  # double contourBase
        intervals,  # int fixedLevelCount
        -9999,  # int useNoData
        -9999,  # double noDataValue
        contour_shp,  # Layer dstLayer
        0,  # int idField
        1,  # int elevField
    )
    ogr_ds = None  # must be in code
    del ogr_ds  # must be in code

    contours = gpd.read_file('contours.shp')
    contours.plot()


def zad3() -> None:
    src = "puszcza_bukowa_szmaragdowe.asc"
    gdaldem = "gdaldem.exe slope"
    dst = "szmaragdowe_geotif_slope.tif"

    subprocess.run(gdaldem + " -of Gtiff -b 1 -s 1.0 " + src + " " + dst)
    image = rasterio.open(dst)
    data = image.read(1)

    data_copy = np.copy(data)
    data_copy[data <= 5] = 1
    data_copy[data > 5] = 2
    data_copy[data > 10] = 3
    data_copy[data > 20] = 4

    area_pixels = np.prod(image.res)
    area_of_classes = np.array({f'class {area}': np.sum(data_copy == area) * area_pixels / 10000 for area in np.unique(data_copy).astype(int)})
    print('Area for specify class in hectares:\n', area_of_classes)

    plt.figure()
    plt.imshow(data_copy, interpolation='nearest', vmin=1)
    plt.colorbar()
    plt.savefig('zad3.png')
    plt.show()


if __name__ == '__main__':
    zad2_b()
