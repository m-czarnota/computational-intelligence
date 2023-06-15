import folium
from folium.plugins import FloatImage


def add_layer(layers: list, name: str) -> None:
    folium.raster_layers.WmsTileLayer(
        url='http://localhost:8080/geoserver/wms',
        layers=layers,
        transparent=True,
        control=True,
        fmt="image/png",
        name=name,
        overlay=True,
        show=True,
    ).add_to(m)


if __name__ == '__main__':
    m = folium.Map(location=[53.4327, 14.5483], control_scale=True)
    add_layer(['dzielnice_osiedla_szczecin'], 'dzielnice')
    add_layer(['OT_BUBD_A'], 'budynki')
    add_layer(['OT_SKJZ_L'], 'ulice')

    compass_rose = folium.FeatureGroup('compass rose')
    FloatImage(
        'https://upload.wikimedia.org/wikipedia/commons/9/99/Compass_rose_simple.svg',
        bottom=4,
        left=4
    ).add_to(compass_rose)
    compass_rose.add_to(m)

    folium.LayerControl().add_to(m)
    m.save("map.html")
