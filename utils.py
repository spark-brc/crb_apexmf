import glob
import os
import pandas as pd
import numpy as np
import streamlit as st
import random
import geopandas as gpd
import pyproj
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as offline
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
from hydroeval import evaluator, nse, rmse, pbias
import base64
from pathlib import Path
import datetime
import statsmodels.api as sm


def define_sim_period(wd):
    if os.path.isfile(os.path.join("./resources/watershed", wd, "APEXCONT.DAT")):
        with open(os.path.join("./resources/watershed", wd, 'APEXCONT.DAT'), "r") as f:
            data = [x.strip().split() for x in f if x.strip()]
        numyr = int(data[0][0])
        styr = int(data[0][1])
        stmon = int(data[0][2])
        stday = int(data[0][3])
        ptcode = int(data[0][4])
        edyr = styr + numyr -1
        stdate = datetime.datetime(styr, stmon, 1) + datetime.timedelta(stday - 1)
        eddate = datetime.datetime(edyr, 12, 31) 
        duration = (eddate - stdate).days

        ##### 
        start_month = stdate.strftime("%b")
        start_day = stdate.strftime("%d")
        start_year = stdate.strftime("%Y")
        end_month = eddate.strftime("%b")
        end_day = eddate.strftime("%d") 
        end_year = eddate.strftime("%Y")

        # NOTE: This is later when we are handling model with different time steps
        # # Check IPRINT option
        # if ptcode == 3 or ptcode == 4 or ptcode == 5:  # month
        #     self.dlg.comboBox_SD_timeStep.clear()
        #     self.dlg.comboBox_SD_timeStep.addItems(['Monthly', 'Annual'])
        #     self.dlg.radioButton_month.setChecked(1)
        #     self.dlg.radioButton_month.setEnabled(True)
        #     self.dlg.radioButton_day.setEnabled(False)
        #     self.dlg.radioButton_year.setEnabled(False)
        # elif ptcode == 6 or ptcode == 7 or ptcode == 8 or ptcode == 9:
        #     self.dlg.comboBox_SD_timeStep.clear()
        #     self.dlg.comboBox_SD_timeStep.addItems(['Daily', 'Monthly', 'Annual'])
        #     self.dlg.radioButton_day.setChecked(1)
        #     self.dlg.radioButton_day.setEnabled(True)
        #     self.dlg.radioButton_month.setEnabled(False)
        #     self.dlg.radioButton_year.setEnabled(False)
        # elif ptcode == 0 or ptcode == 1 or ptcode == 2:
        #     self.dlg.comboBox_SD_timeStep.clear()
        #     self.dlg.comboBox_SD_timeStep.addItems(['Annual'])
        #     self.dlg.radioButton_year.setChecked(1)
        #     self.dlg.radioButton_year.setEnabled(True)
        #     self.dlg.radioButton_day.setEnabled(False)
        #     self.dlg.radioButton_month.setEnabled(False)
        return stdate, eddate, start_year, end_year


def get_val_info(wd):
    if os.path.isfile(os.path.join("./resources/watershed", wd, "interactive.dat")):
        with open(os.path.join("./resources/watershed", wd, 'interactive.dat'), "r") as f:
            data = [x.strip().split() for x in f if x.strip()]
        calstyr = int(data[0][0])
        caledyr = int(data[0][1])
        sims = [int(x) for x in data[1]]
        obds = [x for x in data[2]]
        gw_sims = [int(x) for x in data[3]]
        gw_obds = [x for x in data[4]]
        return calstyr, caledyr, sims, obds, gw_sims, gw_obds


def get_sim_obd(area, stdate, time_step, sims, obds, caldate, eddate):
    rch_file = read_rch_files(os.path.join("./resources/watershed", area))
    df = pd.read_csv(
                    os.path.join("./resources/watershed", area, rch_file[0]),
                    delim_whitespace=True,
                    skiprows=9,
                    usecols=[0, 1, 8],
                    names=['idx', 'rchid', 'sim'],
                    index_col=0)
    df = df.loc["REACH"]
    str_obd = pd.read_csv(
                        os.path.join("./resources/watershed", area, 'stf_mon.obd'),
                        sep=r'\s+', index_col=0, header=0,
                        parse_dates=True, delimiter="\t",
                        na_values=[-999, ""]
                        )
    str_obd = str_obd[obds]
    tot_sim = pd.DataFrame()
    for i in sims:
        df2 = df.loc[df['rchid'] == int(i)]
        df2.index = pd.date_range(stdate, periods=len(df2.sim), freq=time_step)
        df2.rename(columns = {'sim':'sim_{:03d}'.format(i)}, inplace = True)
        tot_sim = pd.concat([tot_sim, df2.loc[:, 'sim_{:03d}'.format(i)]], axis=1,
            # sort=False
            )
    tot_df = pd.concat([tot_sim, str_obd], axis=1)
    tot_df.index = pd.to_datetime(tot_df.index).normalize()
    tot_df = tot_df[caldate:eddate]
    return tot_df


def get_plot(df, sims):
    fig = go.Figure()
    colors = (get_matplotlib_cmap('tab10', bins=8))
    for i in range(len(sims)):
        fig.add_trace(go.Scatter(
            x=df.index, y=df.iloc[:, i], name='Reach {}'.format(sims[i]),
            line=dict(color=colors[i], width=2),
            legendgroup='Reach {}'.format(sims[i])
            ))
    for i in range(len(sims)):
        fig.add_trace(go.Scatter(
            x=df.index, y=df.iloc[:, i+len(sims)], mode='markers', name='Observed {}'.format(sims[i]),
            marker=dict(color=colors[i]),
            legendgroup='Reach {}'.format(sims[i]),
            showlegend=False
            ))
    # line_fig = px.line(df, height=500, width=1200)
    fig.update_layout(
        # showlegend=False,
        plot_bgcolor='white',
        height=600,
        # width=1200
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title='')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title='Monthly Average Stream Discharge (m<sup>3</sup>/s)')
    fig.update_layout(
            legend=dict(
                yanchor="top",
                y=1.0,
                xanchor="center",
                x=0.5,
                orientation="h",
                title='',),
            hovermode= "x unified"
            )
    fig.update_traces(marker=dict(size=10, opacity=0.5,
                                line=dict(width=1,
                                            color='white')
                                            ),
                    selector=dict(mode='markers'),
                    # hovertemplate=None
                    )
    return fig


def get_fdcplot(df, sims, yscale):
    fig = go.Figure()
    colors = (get_matplotlib_cmap('tab10', bins=8))
    for i in range(len(sims)):
        sort = np.sort(df.iloc[:, i])[::-1]
        exceedence = np.arange(1.,len(sort)+1) / len(sort)

        fig.add_trace(go.Scatter(
            x=exceedence*100, y=sort, name='Reach {}'.format(sims[i]),
            line=dict(color=colors[i], width=2),
            legendgroup='Reach {}'.format(sims[i])
            ))

    for i in range(len(sims)):
        sort = np.sort(df.iloc[:, i+len(sims)])[::-1]
        exceedence = np.arange(1.,len(sort)+1) / len(sort)
        fig.add_trace(go.Scatter(
            x=exceedence*100, y=sort, mode='markers', name='Observed {}'.format(sims[i]),
            marker=dict(color=colors[i]),
            legendgroup='Reach {}'.format(sims[i]),
            showlegend=False
            ))
    fig.update_layout(
        # showlegend=False,
        plot_bgcolor='white',
        height=700,
        # width=1200
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title='Exceedance Probability (%)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title='Monthly Average Stream Discharge (m<sup>3</sup>/s)')
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="center",
            x=0.5,
            orientation="h",
            title='',
            ),
        hovermode= "x unified")
    fig.update_traces(marker=dict(size=10, opacity=0.5,
                                line=dict(width=1,
                                            color='white')
                                            ),
                    selector=dict(mode='markers'))
    if yscale == 'Logarithmic':
        fig.update_yaxes(type="log")
    return fig


def get_stats(df):
    df_stat = df.dropna()
    sim = df_stat.iloc[:, 0].to_numpy()
    obd = df_stat.iloc[:, 1].to_numpy()
    df_nse = evaluator(nse, sim, obd)[0]
    df_rmse = evaluator(rmse, sim, obd)[0]
    df_pibas = evaluator(pbias, sim, obd)[0]
    r_squared = (
        ((sum((obd - obd.mean())*(sim-sim.mean())))**2)/
        ((sum((obd - obd.mean())**2)* (sum((sim-sim.mean())**2))))
        )
    return df_nse, df_rmse, df_pibas, r_squared


def get_stats_df(df, sims):
    stats_df = pd.DataFrame()
    for i in range(len(sims)):
        exdf = df.iloc[:, [i, i+len(sims)]]
        df_list = get_stats(exdf)
        stat_series = pd.Series(['{:.3f}'.format(x) for x in df_list], name='Reach {}'.format(sims[i]))
        stats_df = pd.concat([stats_df, stat_series], axis=1)
    stats_df.index = ['NSE', 'RMSE', 'PBIAS', 'R-squared']
    return stats_df


def get_watershed_list():
    ws_nams = [f.name for f in os.scandir("./resources/watershed") if f.is_dir()]
    ws_nams.sort()
    full_paths = [f.path for f in os.scandir("./resources/watershed") if f.is_dir()]
    return ws_nams, full_paths


def read_rch_files(wd):
    rch_files = []
    for filename in glob.glob(wd+"/*.RCH"):
        rch_files.append(os.path.basename(filename))
    return rch_files


def get_variables(wd, rch_file):
    columns = pd.read_csv(
                        os.path.join(wd, rch_file),
                        delim_whitespace=True,
                        skiprows=8,
                        nrows=1,
                        header=None
                        )

    col_lst = columns.iloc[0].tolist()
    col_lst.insert(2, 'YEAR')
    col_dic = dict((i, j) for i, j in enumerate(col_lst))
    keys = [x for x in range(0, 39)]
    col_dic_f = {k: col_dic[k] for k in keys}
    rch_vars = list(col_dic_f.values())
    rch_vars = list(col_dic_f.values())
    return rch_vars


# def get_subnums(wd, rch_file)
def get_matplotlib_cmap(cmap_name, bins, alpha=1):
    if bins is None:
        bins = 10
    cmap = cm.get_cmap(cmap_name)
    h = 1.0 / bins
    contour_colour_list = []

    for k in range(bins):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        contour_colour_list.append('rgba' + str((C[0], C[1], C[2], alpha)))

    C = list(map(np.uint8, np.array(cmap(bins * h)[:3]) * 255))
    contour_colour_list.append('rgba' + str((C[0], C[1], C[2], alpha)))
    return contour_colour_list

def filedownload(df):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="dataframe.csv">Download CSV File</a>'
    return href


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

    # st.markdown('<style>' + open('icons.css').read() + '</style>', unsafe_allow_html=True)

def viz_biomap():
    subdf = gpd.read_file("./resources/subs1.shp")
    subdf.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    subdf = subdf[['Subbasin', 'geometry']]
    subdf['geometry'] = subdf['geometry'].convex_hull
    tt = gpd.GeoDataFrame()
    for i in subdf.index:
        df = gpd.GeoDataFrame()
        df['time']= [str(x)[:-9] for x in pd.date_range('1/1/2000', periods=12, freq='M')]
        # df['time'] = [str(x) for x in range(2000, 2012)]
        df['Subbasin'] = 'Sub{:03d}'.format(i+1)
        df['geometry'] = subdf.loc[i, 'geometry']
        df['value'] = [random.randint(0,12) for i in range(12)]
        tt = pd.concat([tt, df], axis=0)   
    tt.index = tt.Subbasin 
    mfig = px.choropleth(tt,
                    geojson=tt.geometry,
                    locations=tt.index,
                    color="value",
                    #    projection="mercator"
                    animation_frame="time",
                    range_color=(0, 12),
                    )
    mfig.update_geos(fitbounds="locations", visible=False)
    mfig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=600)
    # offline.plot(mfig, auto_open=True, image = 'png', image_filename="map_us_crime_slider" ,image_width=2000, image_height=1000, 
    #             filename='tt.html', validate=True)
    # fig.update_layout(
    #     # showlegend=False,
    #     plot_bgcolor='white',
    #     height=600,
    #     # width=1200
    # )
    return mfig


def viz_perfomance_map(area):
    subdf = gpd.read_file(os.path.join("./resources/watershed", area, 'sub_dis.shp'))
    subdf.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    subdf = subdf[['Id', 'geometry']]
    st.write(subdf.centroid.y[0])
    subdf.index = subdf.Id
    # fig = go.Figure()
    mfig = px.choropleth_mapbox(subdf,
                    geojson=subdf.geometry,
                    locations=subdf.index,
                    # color=sel_yr,
                    mapbox_style="open-street-map",
                    zoom=8,
                    center = {"lat":subdf.centroid.y[0], "lon": subdf.centroid.x[0]},
                    # range_color=(dfmin, dfmax),
                    # opacity=0.3,
                    )
    mfig.update_geos(fitbounds="locations", visible=False)
    mfig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=1000)

    stf_gages = gpd.read_file(os.path.join("./resources/watershed", area, 'stf_gages.shp'))
    stf_gages.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    # stf_gages = stf_gages[['subids', 'geometry']]
    stf_gages.index = stf_gages.subids

    gfig = px.scatter(
                    stf_gages,
                    y=stf_gages.geometry.y,
                    x=stf_gages.geometry.x,

                    # geojson=stf_gages.geometry,
                    # locations=stf_gages.index
                    # color=sel_yr,
                    # mapbox_style="open-street-map",
                    # zoom=8,
                    # center = {"lat":subdf.centroid.y[0], "lon": subdf.centroid.x[0]},
                    # range_color=(dfmin, dfmax),
                    # opacity=0.3,
                    )
    mfig.add_traces(gfig.data)
    # ffig = [mfig, gfig]
    # fig.add_trace(gfig)
    # for i, frame in enumerate(mfig.frames):
    #     mfig.frames[i].data += (gfig.frames[i].data[0],)
    # fig.show()
    st.plotly_chart(gfig, use_container_width=True)


# def get_gw_info():


def wt_df(wd, start_date, grid_id, obd_nam):
    mf_obs = pd.read_csv(
                        os.path.join("./resources/watershed", wd, "modflow.obs"),
                        delim_whitespace=True,
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        os.path.join("./resources/watershed", wd, "dtw_day.obd"),
                        sep='\s+',
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""],
                        delimiter="\t")

    grid_id_lst = mf_obs.index.astype(str).values.tolist()

    output_wt = pd.read_csv(
                        os.path.join("./resources/watershed", wd, "apexmf_out_MF_obs"),
                        delim_whitespace=True,
                        skiprows = 1,
                        names = grid_id_lst,)
    output_wt = output_wt[str(grid_id)] - float(mf_obs.loc[int(grid_id)])
    output_wt.index = pd.date_range(start_date, periods=len(output_wt))

    
    output_wt = pd.concat([output_wt, mfobd_df[obd_nam]], axis=1)
    output_wt = output_wt[output_wt[str(grid_id)].notna()]
    return output_wt


def tot_wt(area, stdate, cal_start, cal_end, grid_ids, obd_nams, time_step=None):
    """combine all groundwater outputs to provide a dataframe for 1 to 1 plot

    Args:
        start_date (str): simulation start date 
        grid_ids (list): list of grid ids used for plot
        obd_nams (list): list of column names in observed data and in accordance with grid ids
        time_step (str, optional): simulation time step (day, month, annual). Defaults to None.

    Returns:
        dataframe: dataframe for all simulated groundwater levels and observed data
    """
    if time_step is None:
        time_step = "D"
        mfobd_file = "dtw_day.obd"
    else:
        time_step = "M"
        mfobd_file = "modflow_month.obd."
    # read obs and obd files to get grid ids, elev, and observed values
    mf_obs = pd.read_csv(
                        os.path.join("./resources/watershed", area, "modflow.obs"),
                        delim_whitespace=True,
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        os.path.join("./resources/watershed", area, mfobd_file),
                        sep='\s+',
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""],
                        delimiter="\t")
    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    # read simulated water elevation
    output_wt = pd.read_csv(
                        os.path.join("./resources/watershed", area, "apexmf_out_MF_obs"),
                        delim_whitespace=True,
                        skiprows = 1,
                        names = grid_id_lst,)
    # append data to big dataframe

    tot_df = pd.DataFrame()
    for grid_id, obd_nam in zip(grid_ids, obd_nams):
        # df = output_wt[str(grid_id)] - float(mf_obs.loc[int(grid_id)]) # calculate depth to water
        df = output_wt[str(grid_id)]
        df.index = pd.date_range(stdate, periods=len(df))
        df = df[cal_start:cal_end]
        if time_step == 'M':
            df = df.resample('M').mean()
        # mfobd_df = float(mf_obs.loc[int(grid_id)]) + mfobd_df[obd_nam]
        mfobd_dff = mfobd_df[obd_nam] + float(mf_obs.loc[int(grid_id)])
        df = pd.concat([df, mfobd_dff], axis=1) # concat sim with obd
        df = df.dropna() # drop nan
        new_cols ={x:y for x, y in zip(df.columns, ['sim', 'obd'])} #replace col nams with new nams
        df['grid_id'] = str(grid_id)
        tot_df = tot_df.append(df.rename(columns=new_cols))
    return tot_df

def gw_scatter(gwdf):

    fig = go.Figure()
    colors = (get_matplotlib_cmap('tab10', bins=8))
    # fig = px.scatter(gwdf, x="sim", y="obd", color="grid_id",trendline="ols")
    fig = px.scatter(gwdf, x="sim", y="obd", color="grid_id", trendline="ols", trendline_scope="overall")

    # # linear regression
    # regline = sm.OLS(gwdf["sim"], sm.add_constant(gwdf["obd"])).fit().fittedvalues

    # # add linear regression line for whole sample
    # fig.add_traces(go.Scatter(x=gwdf["sim"], y=regline,
    #                         mode = 'lines',
    #                         # marker_color='black',
    #                         name='trend all',
    #                         # trendline_scope="overall"
    #                         )
    #                         )

    fig.update_layout(
        # showlegend=False,
        plot_bgcolor='white',
        height=700,
        # width=1200
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title='Simulated Groundwater Level (meters)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title='Observed Groundwater Level (meters)')
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="center",
            x=0.5,
            orientation="h",
            title='',
            ),
        hovermode= "x unified")

    return fig

def gw_scatter2(gwdf):

    fig = go.Figure()
    colors = (get_matplotlib_cmap('tab10', bins=8))
    fig = px.scatter(gwdf, x="sim", y="obd", color="grid_id",trendline="ols")
    # fig = px.scatter(gwdf, x="sim", y="obd", color="grid_id", trendline="ols", trendline_scope="overall")

    # # linear regression
    # regline = sm.OLS(gwdf["sim"], sm.add_constant(gwdf["obd"])).fit().fittedvalues

    # # add linear regression line for whole sample
    # fig.add_traces(go.Scatter(x=gwdf["sim"], y=regline,
    #                         mode = 'lines',
    #                         # marker_color='black',
    #                         name='trend all',
    #                         # trendline_scope="overall"
    #                         )
    #                         )

    fig.update_layout(
        # showlegend=False,
        plot_bgcolor='white',
        height=700,
        # width=1200
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title='Simulated Groundwater Level (meters)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', title='Observed Groundwater Level (meters)')
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="center",
            x=0.5,
            orientation="h",
            title='',
            ),
        hovermode= "x unified")

    return fig