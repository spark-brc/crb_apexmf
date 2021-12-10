import streamlit as st
import json
import geopandas as gpd
import pyproj
import plotly.graph_objs as go
import os
import logging
import plotly.express as px


wd = r"D:\Projects\Tools\crb_apexmf\crb_apexmf_git\resources\watershed\Dolores"

def t1(wd):
    # reading in the polygon shapefile
    polygon = gpd.read_file(os.path.join(wd, "subs1.shp"))
    # project GeoPandas dataframe
    map_df = polygon 
    map_df.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)

    points = gpd.read_file(os.path.join(wd, "stf_gages.shp"))
    # project GeoPandas dataframe
    points.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    # define lat, long for points
    lat = points.geometry.y
    lon = points.geometry.x


    # set GeoJSON file path
    gj = os.path.join(wd, "geojson.json")
    # # write GeoJSON to file
    map_df.to_file(gj, driver = "GeoJSON")
    with open(os.path.join(wd, "geojson.json")) as geofile:
        j_file = json.load(geofile)
    # index geojson
    i=1
    for feature in j_file["features"]:
        feature['id'] = str(i).zfill(2)
        i += 1
    # mapbox token
    mapboxt = 'MapBox Token'
    
    # define layers and plot map
    choro = go.Choroplethmapbox(
        z=map_df['Subbasin'],
        locations=map_df.index, colorscale = 'Viridis',
        geojson=j_file,
        text = map_df['Subbasin'], marker_line_width=0.1
        ) 
    scatt = go.Scattermapbox(lat=lat, lon=lon,mode='markers+text',    
            below='False', marker=dict( size=12, color ='rgb(56, 44, 100)'))
    layout = go.Layout(title_text ='USA Cities', title_x =0.5,  
            width=950, height=700,mapbox = dict(center= dict(lat=37,  
            lon=-95),accesstoken= mapboxt, zoom=4,style="stamen-terrain"))
    # streamlit multiselect widget
    layer1 = st.multiselect('Layer Selection', [choro, scatt], 
            format_func=lambda x: 'Polygon' if x==choro else 'Points')
    # assign Graph Objects figure
    fig = go.Figure(data=layer1, layout=layout)
    # display streamlit map
    # st.plotly_chart(fig)
    return fig


def t2(area):
    subdf = gpd.read_file(os.path.join("./resources/watershed", area, 'subs1.shp'))
    subdf.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    subdf = subdf[['Subbasin', 'geometry']]
    # subdf['geometry'] = subdf['geometry'].convex_hull
    lat = subdf.centroid.y[0]
    lon = subdf.centroid.x[0]
    st.write(lat)
    subdf.index = subdf.Subbasin
    mfig = go.Figure()

    with open(os.path.join("./resources/watershed", area, "geojson.json")) as geofile:
        j_file = json.load(geofile)
    # index geojson
    i=1
    for feature in j_file["features"]:
        feature['id'] = str(i).zfill(2)
        i += 1
    # mapbox token


    mfig.add_trace(
        go.Choroplethmapbox(
            z=subdf['Subbasin'],
            geojson=j_file,
            locations=subdf.index,
            colorscale = 'Viridis',
            # color=sel_yr,
            # mapbox_style="open-street-map",
            # zoom=8,
            # center = {"lat":subdf.centroid.y[0], "lon": subdf.centroid.x[0]},
            # range_color=(dfmin, dfmax),
            # opacity=0.8,
            # marker_color='#FFFFFF',
            text = subdf['Subbasin'],
            showscale=False,
            marker_line_color='#FFFFFF',
            marker_opacity=0.5,
            marker_line_width=1,
            hovertemplate = '<b>Sub</b>: <b>%{text}</b><extra></extra>',
            # marker=dict(color='rgb(0, 0, 0)')
            ))

    stf_gages = gpd.read_file(os.path.join("./resources/watershed", area, 'stf_gages.shp'))
    stf_gages.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    mfig.add_trace(
        go.Scattermapbox(
                lat=stf_gages.geometry.y, lon=stf_gages.geometry.x,
                mode='markers', 
                text = stf_gages['HydroID'],
                # fillcolor='#FFFFFF',
                hovertemplate = '<b>Sub</b>: <b>%{text}</b><extra></extra>',
                below='False',
                marker = go.scattermapbox.Marker(size=10, symbol='circle', color ='red'),
                name='Streamflow Gages'
                # marker=dict(
                #     size=100, color ='rgb(56, 44, 100)',symbol='airport',
                #     ),

                # marker=dict(
                #     size=100, color ='rgb(56, 44, 100)',symbol='airport',
                #     ),
                # opacity=0.8
                ))
    gw_gages = gpd.read_file(os.path.join("./resources/watershed", area, 'gw_gages.shp'))
    gw_gages.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    mfig.add_trace(
        go.Scattermapbox(
                lat=gw_gages.geometry.y, lon=gw_gages.geometry.x,
                mode='markers', 
                text = [gw_gages['Site_name'][i] + '<br>' + gw_gages['cali_name'][i] for i in range(gw_gages.shape[0])],
                # gw_name = gw_gages['cali_name'],
                # fillcolor='#FFFFFF',
                hovertemplate = '<b>%{text}</b><extra></extra>',
                below='False',
                marker = go.scattermapbox.Marker(size=10, symbol='circle', color ='blue'),
                name='Groundwater Monitoring Wells'
                # marker=dict(
                #     size=100, color ='rgb(56, 44, 100)',symbol='airport',
                #     ),

                # marker=dict(
                #     size=100, color ='rgb(56, 44, 100)',symbol='airport',
                #     ),
                # opacity=0.8
                ))  
    mapboxt = 'MapBox Token'
    mfig.update_geos(fitbounds="locations", visible=False)
    # mfig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=1000)
    mfig.update_layout(
            title_text ='USA Cities', title_x=0.5,  
            width=950, height=700,
            mapbox=dict(
                center= dict(lat=lat, lon=lon),
                accesstoken=mapboxt, zoom=8, style="stamen-terrain"),
            legend=dict(
                yanchor="top",
                y=1.05,
                xanchor="center",
                x=0.5,
                orientation="h",
                title='',)            
                )



    # stf_gages = gpd.read_file(os.path.join("./resources/watershed", area, 'stf_gages.shp'))
    # stf_gages.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    # # stf_gages = stf_gages[['subids', 'geometry']]
    # stf_gages.index = stf_gages.Subbasin
    # mfig = go.Figure()

    # mfig.add_trace(go.Scattermapbox(
    #                     # stf_gages,
    #                     lat=stf_gages.geometry.y,
    #                     lon=stf_gages.geometry.x,
    #                     text = 'Milano',
    #                     marker_color='red',
    #                     below='False',
    #                     marker_size=8 ))


    # gfig = px.scatter_geo(
    #                 stf_gages,
    #                 # geojson=subdf.geometry,
    #                 lat=stf_gages.geometry.y,
    #                 lon=stf_gages.geometry.x,
    #                 # opacity=0.3,
    #                 )
    # mfig.add_traces(gfig)

     # gfig = px.scatter(
    #                 stf_gages,
    #                 y=stf_gages.geometry.y,
    #                 x=stf_gages.geometry.x,

    #                 geojson=stf_gages.geometry,
    #                 locations=stf_gages.index,
    #                 # color=sel_yr,
    #                 # mapbox_style="open-street-map",
    #                 zoom=8,
    #                 center = {"lat":subdf.centroid.y[0], "lon": subdf.centroid.x[0]},
    #                 # range_color=(dfmin, dfmax),
    #                 opacity=0.3,
    #                 )
    # mfig.add_traces(gfig)
    # ffig = [mfig, gfig]
    # fig.add_trace(gfig)
    # for i, frame in enumerate(mfig.frames):
    #     mfig.frames[i].data += (gfig.frames[i].data[0],)
    # fig.show()

    return mfig







if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)
    fig = t1(wd)
    mfig = t2("Dolores")
    st.plotly_chart(fig)
    st.plotly_chart(mfig)