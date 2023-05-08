import pylas
from osgeo import gdal
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from rasterio.plot import show
from whitebox import whitebox

WORK_DIR = r'D:\Programming\Python\computational-intelligence\geoinformatics-data-processing\zad8\\'


def get_geo_data_from_band(filename: str) -> dict:
    dem_file = gdal.Open(filename)
    band = dem_file.GetRasterBand(1)

    height = band.ReadAsArray()
    gt = dem_file.GetGeoTransform()

    ulx = gt[0]
    uly = gt[3]
    res = gt[1]

    x_size = dem_file.RasterXSize
    y_size = dem_file.RasterYSize

    return {
        'min_val': np.min(height),
        'ulx': ulx,
        'uly': uly,
        'res': res,
        'x_size': x_size,
        'y_size': y_size,
        'lrx': ulx + x_size * res,
        'lry': uly - y_size * res,
    }


def zad1() -> None:
    """
    zastosowano metodę odwrotnych odległości, ponieważ uwzględnia ona wszystkie punkty danych
    w porównaniu do najbliższych sąsiadów, który uwzględnia jedynie najbliższych sąsiadów
    """

    geo_data = get_geo_data_from_band('puszcza_bukowa_szmaragdowe.asc')

    gdal.Grid(
        "zad1.tif",
        "puszcza_bukowa_szmaragdowe_losowe_punkty.shp",
        zfield="VALUE",
        algorithm=f"invdist:radius1=40:radius2=40:smoothing=5:nodata={geo_data['min_val']}",
        outputBounds=[geo_data['ulx'], geo_data['uly'], geo_data['lrx'], geo_data['lry']],
        width=geo_data['x_size'],
        height=geo_data['y_size'],
    )

    with rasterio.open("zad1.tif") as src:
        data = src.read()

        fig, ax = plt.subplots(1, figsize=(20, 10))
        cmap = plt.get_cmap("magma")

        fig.colorbar(
            cm.ScalarMappable(
                norm=colors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data)),
                cmap=cmap
            ),
            ax=ax,
        )

        show(data, transform=src.transform, ax=ax, cmap=cmap)
        plt.savefig('zad1_result.png')

        src.close()
        plt.close(fig)


def zad2() -> None:
    """
    zastosowane metody i różnice między nimi w porównaniu do pliku referencyjnego:
        - najbliższy sąsiad: resultat wygląda jak mozaika. wynika to z tego, że uwzględnia on jedynie najbliższych sąsiadów, przez co dany fragment może obejmować swoim kolorem więcej niż powinien. widoczne to jest na dole jeziora
        - liniowa: wygląda jak uproszczona wersja jeziora, szczególnie jego otoczka. na górze jeziora przy jego ujściu fragment jest zbyt wypełniony, co obrazuje jakby tam także znajdowało się w takim samym stanie jezioro jak na środku
        - odwrotne odległości: najwierniejsze wypełnienie kolorami. chociaż na górze po prawej fragment został rozmyty
    """

    geo_data = get_geo_data_from_band('puszcza_bukowa_szmaragdowe.asc')

    methods = {
        'Linear': f"linear:nodata{geo_data['min_val']}",
        'Inverse distance': f"invdist:radius1=40:radius2=40:smoothing=5:nodata={geo_data['min_val']}",
        'Nearest': f"nearest:nodata={geo_data['min_val']}"
    }
    results = {'Reference': 'puszcza_bukowa_szmaragdowe.asc'}

    for method in methods:
        filename = f'zad2_{method}.tif'
        results[method] = filename

        gdal.Grid(
            filename,
            "puszcza_bukowa_szmaragdowe_losowe_punkty.shp",
            zfield="VALUE",
            algorithm=methods[method],
            outputBounds=[geo_data['ulx'], geo_data['uly'], geo_data['lrx'], geo_data['lry']],
            width=geo_data['x_size'],
            height=geo_data['y_size']
        )

    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    cmap = plt.get_cmap('rainbow')

    for i, method in enumerate(results):
        with rasterio.open(results[method]) as src:
            data = src.read()

            x = i // 2
            y = i % 2

            fig.colorbar(
                cm.ScalarMappable(
                    norm=colors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data)),
                    cmap=cmap
                ),
                ax=ax[x, y]
            )
            ax[x, y].set_title(method)

            show(data, transform=src.transform, ax=ax[x, y], cmap=cmap)

    plt.savefig('zad2_result.png')
    plt.close(fig)


def zad3() -> None:
    """
    zastosowana została metoda odwrotnej ważonej odległości, ponieważ:
        - pozwala ona na sporą elastyczność w dostosowywaniu wpływu odległości pomiędzy punktami danych
        - dobrze radzi sobie z interpolacją w obszarach, gdzie punkty są rozproszone
        - dobrze radzi sobie z punktami skrajnymi, ponieważ uwzględnia wszystkie dane
    """

    las = pylas.read('lidar.las')
    las.points = las.points[las.classification == 2]

    points2_filename = 'ground.las'
    las.write(points2_filename)

    wbt = whitebox.WhiteboxTools()
    wbt.work_dir = WORK_DIR

    interpolation_ground_filename = 'zad3.tif'
    wbt.lidar_idw_interpolation(
        points2_filename,
        output=interpolation_ground_filename,
        resolution=1.0,
        weight=2.0,
        radius=10.0,
    )

    with rasterio.open(interpolation_ground_filename) as src:
        data = src.read()
        data_max = np.nanmax(data)

        fig, ax = plt.subplots(1, figsize=(20, 10))
        cmap = plt.get_cmap('rainbow')

        show(data, transform=src.transform, ax=ax, cmap=cmap, vmin=0, vmax=data_max)
        plt.savefig('zad3.png')

        src.close()
        plt.close(fig)


if __name__ == '__main__':
    zad2()
