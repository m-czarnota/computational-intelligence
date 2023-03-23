import pandas as pd
from geopy.geocoders import Nominatim
import folium
from folium import plugins
import geopandas as gpd


def zad1() -> None:
    data = pd.read_csv('points.csv', index_col=[0])
    geolocator = Nominatim(user_agent='sip')
    lang = 'pl'

    addresses = [[values[1], values[0], geolocator.reverse(', '.join(values.astype(str)[::-1]), language=lang)] for values in data.values]
    results = pd.DataFrame(addresses, columns=['X', 'Y', 'Decoded address'])
    results.to_csv('zad1.csv', index=False)

    folium_map = folium.Map()
    plugins.MarkerCluster(results.iloc[:, :2]).add_to(folium_map)

    folium_map.save('zad1.html')
    folium_map.show_in_browser()


def zad2() -> None:
    countries = gpd.read_file('country.geojson')
    gmi_cntry = 'GMI_CNTRY'
    date_to_filter = '2021-03-13'
    covid_data = pd.read_csv('owid-covid-data.csv').rename(columns={'iso_code': gmi_cntry})

    covid_countries = countries.merge(covid_data.query(f'date == "{date_to_filter}"'), on=gmi_cntry)
    folium_map = folium.Map()
    bins = list(pd.to_numeric(covid_countries['total_cases']).quantile([0, 0.25, 0.5, 0.75, 1]))

    folium.Choropleth(
        geo_data=covid_countries,
        name='Count of residents',
        data=covid_countries,
        columns=[gmi_cntry, 'total_cases'],
        key_on=f'feature.properties.{gmi_cntry}',
        fill_color='PuRd',
        fill_opacity=0.7,
        line_opacity=0.1,
        legend_name='Count of residents',
        highlight=True,
        reset=True,
        bins=bins,
    ).add_to(folium_map)

    folium.GeoJson(
        data=covid_countries,
        name='Count of infected',
        smooth_factor=2,
        style_function=lambda x: {'color': 'black', 'fillColor': 'transparent', 'weight': 2},
        tooltip=folium.GeoJsonTooltip(
            fields=['total_cases'],
            labels=False,
            sticky=False,
        ),
        highlight_function=lambda x: {'weight': 3, 'fillColor': 'grey'}
    ).add_to(folium_map)

    folium.LayerControl().add_to(folium_map)
    folium.Popup('outline Popup on GeoJSON').add_to(folium_map)
    plugins.MiniMap().add_to(folium_map)

    folium_map.save('zad2.html')
    folium_map.show_in_browser()


if __name__ == '__main__':
    zad2()
