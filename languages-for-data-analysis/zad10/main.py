import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely import wkt, geometry
from shapely.geometry import Point
import pyproj
from shapely.ops import transform, unary_union


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

    min_max_polygon_values = pd.DataFrame([polygon.bounds for polygon in polygons], columns=['min_x', 'min_y', 'max_x', 'max_y'])
    random_points_x = np.random.randint(min_max_polygon_values['min_x'].min(), min_max_polygon_values['max_x'].max(), 100)
    random_points_y = np.random.randint(min_max_polygon_values['min_y'].min(), min_max_polygon_values['max_y'].max(), 100)
    points = [Point(random_points_x[iteration], random_points_y[iteration]) for iteration in range(random_points_x.size)]

    shape = unary_union(polygons)

    points_are_in_polygons = pd.DataFrame({f'polygon{polygon_iter + 1}': {f'point{point_iter}': polygon.within(point) for point_iter, point in enumerate(points)} for polygon_iter, polygon in enumerate(polygons)})
    print(points_are_in_polygons)

    fig, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')

    for geom_iter, geom in enumerate(shape.geoms):
        xs, ys = geom.exterior.xy
        label = data['typ_budynku'][geom_iter]
        axs.fill(xs, ys, fc=label_colors[label], ec='none', alpha=0.5)

    plt.legend(label_colors)
    plt.show()


if __name__ == '__main__':
    data = load_data()

    polygons = map_data_to_polygons(data)
    transformed = transform_from_2180_to_4326(polygons)

    summary_area_for_class = calc_summary_area_in_classes(data)
    print(summary_area_for_class)

    visualize_buildings(data)

    """
    unary_union
    exterior
    """





