import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely import wkt
from shapely.geometry import Point
import pyproj
from shapely.ops import transform, unary_union

IMAGES_DIR = './images'


def load_data() -> pd.DataFrame:
    return pd.read_csv('budynki_multi.csv', sep='\t')


def map_data_to_polygons(data: pd.DataFrame) -> list:
    return [wkt.loads(row['wkt_geom']) for row_iter, row in data.iterrows()]


def transform_from_2180_to_4326(data):
    wgs84 = pyproj.CRS('EPSG:4326')
    pl = pyproj.CRS('EPSG:2180')

    project = pyproj.Transformer.from_crs(pl, wgs84, always_xy=True).transform

    return [transform(project, polygon) for polygon in data]


def calc_summary_area_in_classes(data: pd.DataFrame):
    classes = data['typ_budynku'].unique()
    summary_area = {label: 0 for label in classes}

    for label in classes:
        summary_area[label] = data[data['typ_budynku'] == label]['wkt_geom'].map(lambda val: wkt.loads(val).area).sum()

    return summary_area


def visualize_buildings(data: pd.DataFrame):
    polygons = map_data_to_polygons(data)
    labels = data['typ_budynku'].unique()
    label_colors = {label: np.random.rand(3,) for label in labels}

    polygons_bounds = pd.DataFrame([polygon.bounds for polygon in polygons], columns=['min_x', 'min_y', 'max_x', 'max_y'])
    random_points_x = np.random.randint(polygons_bounds['min_x'].min(), polygons_bounds['max_x'].max(), 100)
    random_points_y = np.random.randint(polygons_bounds['min_y'].min(), polygons_bounds['max_y'].max(), 100)
    points = [Point(rand_x, rand_y) for rand_x, rand_y in zip(random_points_x, random_points_y)]

    shape = unary_union(polygons)

    points_are_in_polygons = pd.DataFrame({f'polygon{polygon_iter + 1}': {f'point{point_iter}': point.within(polygon) for point_iter, point in enumerate(points)} for polygon_iter, polygon in enumerate(polygons)})

    fig, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')

    for geom_iter, geom in enumerate(shape.geoms):
        xs, ys = geom.exterior.xy
        label = data['typ_budynku'][geom_iter]
        axs.fill(xs, ys, fc=label_colors[label], ec='none', alpha=0.5)

    for point_iter, point in enumerate(points):
        plt.scatter(point.x, point.y, c='g' if points_are_in_polygons.loc[f'point{point_iter}'].any() else 'r')

    plt.legend(label_colors)
    # plt.show()
    plt.savefig(f'{IMAGES_DIR}/zad1.png')


def zad1() -> None:
    data = load_data()

    polygons = map_data_to_polygons(data)
    transformed = transform_from_2180_to_4326(polygons)

    summary_area_for_class = calc_summary_area_in_classes(data)
    print(summary_area_for_class)

    with open('zad1.json', 'w') as outfile:
        json.dump(data.to_json(), outfile)

    visualize_buildings(data)


def zad2() -> None:
    data = load_data()
    polygons = map_data_to_polygons(data)

    polygons_bounds = pd.DataFrame([polygon.bounds for polygon in polygons],
                                   columns=['min_x', 'min_y', 'max_x', 'max_y'])
    random_points_x = np.random.randint(polygons_bounds['min_x'].min(), polygons_bounds['max_x'].max(), 200)
    random_points_y = np.random.randint(polygons_bounds['min_y'].min(), polygons_bounds['max_y'].max(), 200)
    points = [Point(rand_x, rand_y) for rand_x, rand_y in zip(random_points_x, random_points_y)]

    points_are_in_polygons = pd.DataFrame({
        f'polygon{polygon_iter + 1}': {
            f'point{point_iter}': point.within(polygon) for point_iter, point in enumerate(points)
        } for polygon_iter, polygon in enumerate(polygons)
    })

    data['points'] = [{} for i in range(len(data))]
    for data_point_iter, data_point in enumerate(data.iterrows()):
        for point_iter, point in enumerate(points):
            data.loc[data_point_iter]['points'][f'x_{point.x}_y_{point.y}'] = points_are_in_polygons[f'polygon{data_point_iter + 1}'][f'point{point_iter}']

    with open('zad2.json', 'w') as outfile:
        json.dump(data.to_json(), outfile)


if __name__ == '__main__':
    zad2()

    """
    unary_union
    exterior
    """





