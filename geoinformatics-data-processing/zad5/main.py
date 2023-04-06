import csv

import pandas as pd
from owslib.fes import PropertyIsEqualTo
from owslib.wfs import WebFeatureService
from owslib.fes import Or as fil_or, etree
import geopandas as gpd
from owslib.wms import WebMapService
from osgeo import gdal
import numpy as np
import time


def zad1() -> None:
    wfs = WebFeatureService('http://sdi.gdos.gov.pl/wfs', version='1.1.0')
    filter_bialowieski = PropertyIsEqualTo(propertyname='kodinspire', literal='PL.ZIPOP.1393.PN.8')
    filter_tatrzanski = PropertyIsEqualTo(propertyname='kodinspire', literal='PL.ZIPOP.1393.PN.14')
    filter_draw_wol = fil_or([filter_tatrzanski, filter_bialowieski])
    filter_draw_wol_xml = etree.tostring(filter_draw_wol.toXML()).decode("utf-8")
    response = wfs.getfeature(typename='GDOS:ParkiNarodowe', filter=filter_draw_wol_xml)

    out = open('parki_narodowe.gml', 'wb')
    temp = str(response.read(), 'utf-8')
    out.write(bytes(temp, 'utf-8'))
    out.close()

    df = gpd.read_file('parki_narodowe.gml')
    result = pd.DataFrame({name: gpd.GeoSeries(area).area for name, area in zip(df['nazwa'], df['geometry'])})
    print(result.to_markdown())
    result.to_csv('parki_narodowe.csv')


def zad2() -> None:
    box = np.fromstring('659944.766407799, 486467.44840687, 660557.657032805, 486824.870281908', dtype=float, sep=',')
    tile_size = 200
    box1, box0, box3, box2 = (box[1], box[0], box[3], box[2])

    wms = WebMapService('https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow', version='1.3.0')

    response = wms.getfeatureinfo(
        layers=['kontury'],
        srs='EPSG:2180',
        bbox=(box1, box0, box3, box2),
        size=(tile_size, tile_size),
        format='image/png',
        # query_layers=['bvv:gmd_ex'],
        info_format="text/csv",
        xy=(250, 250),
    )

    xml_string = response.read()
    print(xml_string)
    etree.tostring(xml_string).decode("utf-8")

    out = open('zad2.csv', 'wb')
    out.write(response.read())
    out.close()

    file = open('zad2.csv', 'w')
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(response.read())
    file.close()


def zad3() -> None:
    box = np.fromstring('4500000, 5500000, 4505000, 5505000', dtype=float, sep=',')
    tile_size = 200
    box1, box0, box3, box2 = (box[1], box[0], box[3], box[2])

    wms = WebMapService('https://mapy.geoportal.gov.pl/wss/service/img/guest/TOPO/MapServer/WMSServer', version='1.3.0')

    i = 1
    while i < 10:
        img = wms.getmap(
            layers=['Raster'],
            styles=['default'],
            srs='EPSG:2180',
            bbox=(box1, box0, box3, box2),
            size=(tile_size, tile_size),
            format='image/png',
            # transparent=0
        )
        # print(img)

        out = open(f'zad3/tile_{str(i)}.png', 'wb')
        out.write(img.read())

        print(f'Progress: {i / (10 - 1) * 100}%')
        i += 1

        # print(box0)
        # print(box1)

        box0 += tile_size
        box1 += tile_size
        box2 += tile_size
        box3 += tile_size

        time.sleep(2)


if __name__ == '__main__':
    zad2()
