import os
import fiona
import numpy as np
import pprint
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import json


path = r""
file = "powiaty.gpkg"
powiaty = os.path.join(path, file)


def zad1() -> None:
    powiaty_gdf = gpd.read_file(powiaty, layer='powiaty')

    powiaty_4326 = powiaty_gdf.to_crs("EPSG:4326")
    powiaty_4326.to_file("powiaty_4326.shp")

    powiaty_2180 = powiaty_gdf.to_crs("EPSG:2180")
    powiaty_2180.to_file("powiaty_2180.geojson", driver='GeoJSON')

    powiaty_3857 = powiaty_gdf.to_crs("EPSG:3857")
    powiaty_3857.to_file("powiaty_google_mercator_3857.geojson", layer='countries', driver="GPKG")


def zad2() -> None:
    nature = gpd.read_file(powiaty, layer='GDOS:SpecjalneObszaryOchrony')
    powiaty_gdf = gpd.read_file(powiaty, layer='powiaty')

    uk = powiaty_gdf.query("jpt_kod_je.str.startswith('32')")
    powiaty_intersection = gpd.overlay(uk, nature, how='intersection').to_crs('epsg:2180')

    jpt_kod_je_numpy = np.array(powiaty_intersection['jpt_kod_je'])
    intersection_area_numpy = np.array(powiaty_intersection.area)

    grouped_data = pd.DataFrame(
        np.c_[jpt_kod_je_numpy, intersection_area_numpy],
        columns=['jpt_kod_je', 'area_2000']
    )
    nature_2000 = grouped_data.groupby('jpt_kod_je').sum()
    powiaty_nature = uk.merge(nature_2000, on=('jpt_kod_je'))

    data = pd.DataFrame({
        'id_powiat': powiaty_nature['jpt_kod_je'],
        'nazwa_powiatu': powiaty_nature['jpt_nazwa_'],
        'powierzchnia_powiatu': powiaty_nature.area,
        'procent_powierzchni__zajmowany_przez_natura_2000': powiaty_nature['area_2000'] * 100 / powiaty_nature.area,
        'geometry': powiaty_nature['geometry'],
    }, index=None)
    data.to_csv('zad2.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    zad2()
