import logging
import plotly.express as px
import streamlit as st
import os
import pandas as pd
import base64
import numpy as np
from utils import (
    read_rch_files, get_variables, viz_biomap, get_sim_obd, get_plot, get_stats_df,
    filedownload
)
import glob
import utils

pd.options.mode.chained_assignment = None

LINE = """<style>
.vl {
  border-left: 2px solid black;
  height: 100px;
  position: absolute;
  left: 50%;
  margin-left: -3px;
  top: 0;
}
</style>

<div class="vl"></div>"""


st.set_page_config(layout="wide", initial_sidebar_state='collapsed')
st.title('CRB APEX-MODFLOW model performance')
st.markdown("""
This app helps analyze CRB APEX-MODFLOW model performance.
- **Main Python libraries:** base64, pandas, streamlit, plotly, geopandas
- **Source code:** [github.com/spark-brc/crb_apexmf](https://github.com/spark-brc/crb_apexmf)
""")

ws_nams, full_paths = utils.get_watershed_list()

col1, line, col2, col3, col4 = st.beta_columns([0.2, 0.05, 0.1, 0.15, 0.2])
area = col1.selectbox(
    "Select Watershed", ws_nams
    )
with line:
    st.markdown(LINE, unsafe_allow_html=True)
with col2:
    # st.markdown(
    #     "<h3 style='text-align: center;'>Simulation period</h3>",
    #     unsafe_allow_html=True)
    st.markdown(
        """
        ### Simulation period
        """)
with col3:
    st.markdown(
        """
        ### Streamgage station (Reach ID)
        """)
if area == ws_nams[0]:
    with col2:
        st.markdown(
            """
            ## 1992 - 2001
            """)
        st.markdown("---")
    with col3:
        st.markdown(
            """
            ## 12, 57, 75
            """)
        st.markdown("---")
if area == ws_nams[1]:
    with col2:
        st.markdown(
            """
            ## 2000 - 2012
            """)
    with col3:
        st.markdown(
            """
            ## 9, 96, 199
            """)
if area == ws_nams[2]:
    with col2:
        st.markdown(
            """
            ## 2010 - 2019
            """)
    with col3:
        # st.markdown(
        #     "<h4 style='text-align: center;'>66, 102, 133</h4>",
        #     unsafe_allow_html=True)
        st.markdown(
            """
            ## 66, 102, 133
            """)


def main(df, sims):
    tdf = st.beta_expander('Simulated and Observed Streamflow for {} APEX-MODFLOW Watershed Model'.format(area))
    tdf.dataframe(df, height=500)
    tdf.markdown(utils.filedownload(df), unsafe_allow_html=True)
    st.plotly_chart(utils.get_plot(df, sims), use_container_width=True)
    stats_df = utils.get_stats_df(df, sims)

    with col4:
        st.markdown(
            """
            ### Objective Functions
            """)
        st.dataframe(stats_df.T)
    wtdf = st.beta_expander('Simulated and Observed Groundwater Level for {} APEX-MODFLOW Watershed Model'.format(area))
    mddf = st.beta_expander('{} Model Description'.format(area))
    # tdf.dataframe(df, height=500)
        
@st.cache
def load_data():
    if area == ws_nams[0]:
        sims = [12, 57, 75]
        obds = ['str_012', 'str_057', 'str_075']
        stdate = '1/1/1980'
        time_step ='M'
        caldate = '1/1/1992'
        eddate = '12/31/2001'
    if area == ws_nams[1]:
        sims = [9, 96, 199]
        obds = ['sub009', 'sub096', 'sub199']
        stdate = '1/1/1990'
        time_step ='M'
        caldate = '1/1/2000'
        eddate = '12/31/2012'        
    if area == ws_nams[2]:
        sims = [66, 102, 133]
        obds = ['sub_66','sub_102','sub_133']
        stdate = '1/1/2000'
        time_step ='M'
        caldate = '1/1/2010'
        eddate = '12/31/2019'        
    df = get_sim_obd(area, stdate, time_step, sims, obds, caldate, eddate)
    
    return df, sims


if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)
    df, sims = load_data()
    main(df, sims)
