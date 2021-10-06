import logging
from hydralit.app_template import HydraHeadApp
import plotly.express as px
import streamlit as st
import os
import pandas as pd
import base64
import numpy as np
import glob
import utils

pd.options.mode.chained_assignment = None


LINE = """<style>
.vl {
  border-left: 2px solid #797A7D;
  height: 400px;
  position: absolute;
  left: 50%;
  margin-left: -3px;
  top: 0;
}
</style>
<div class="vl"></div>"""

class ModelPerform(HydraHeadApp):
    def __init__(self, area):
        self.area = area


    def run(self):

        st.title('CRB APEX-MODFLOW model performance')
        st.markdown("""
        This app helps analyze CRB APEX-MODFLOW model performance.
        - **Main Python libraries:** base64, pandas, streamlit, plotly, geopandas
        - **Source code:** [github.com/spark-brc/crb_apexmf](https://github.com/spark-brc/crb_apexmf)
        """)

        ws_nams, full_paths = utils.get_watershed_list()
        # col1, line, col2, col3, col4 = st.columns([0.2, 0.05, 0.1, 0.15, 0.2])
        col1, line, col2= st.columns([0.5, 0.05, 0.45])


        # area = col2.selectbox(
        #     "Select Watershed", ws_nams
        #     )
        stdate, eddate, start_year, end_year = utils.define_sim_period(self.area)
        calstyr, caledyr, sims, obds, gw_sims, gw_obds = utils.get_val_info(self.area)
        

        with col1:
            st.plotly_chart(utils.loc_map(self.area), use_container_width=True)

        with line:
            st.markdown(LINE, unsafe_allow_html=True)
        with col2:
            # st.markdown(
            #     "<h3 style='text-align: center;'>Simulation period</h3>",
            #     unsafe_allow_html=True)
            st.markdown(
                '**Simulation period**: &nbsp;{} - {}&emsp;|&emsp;**Calibrated ** from {} - {}&nbsp;'.format(start_year, end_year, calstyr, caledyr))
            st.markdown('---')

        with col2:
            st.markdown('**Streamgage station** (Reach ID):&nbsp; {}'.format(", ".join([str(x) for x in sims])))
            st.markdown('---')

        with col2:
            sim_range = st.slider(
                "Set Analysis Period:",
                min_value=int(start_year),
                max_value=int(end_year), value=(int(calstyr),int(caledyr)))

        def main(df, sims, gwdf):
            tdf = st.expander('{} Dataframe for Simulated and Observed Stream Discharge'.format(self.area))
            tdf.dataframe(df, height=500)
            tdf.markdown(utils.filedownload(df), unsafe_allow_html=True)
            # utils.viz_perfomance_map(self.area)

            st.markdown("## Hydrographs for stream discharge")
            
            st.plotly_chart(utils.get_plot(df, sims), use_container_width=True)
            stats_df = utils.get_stats_df(df, sims)

            with col2:
                st.markdown(
                    """
                    ### Objective Functions
                    """)
                st.dataframe(stats_df.T)

            tcol1, tcol2 = st.columns([0.55, 0.45])
            tcol1.markdown("## Flow Duration Curve")

            
            pcol1, pcol2= st.columns([0.1, 0.9])
            yscale = pcol1.radio("Select Y-axis scale", ["Linear", "Logarithmic"])
            pcol2.plotly_chart(utils.get_fdcplot(df, sims, yscale), use_container_width=True)
            # pcol3.image('tenor.gif')
            st.markdown("## Groundwater Levels (Depth to water)")
            gwcol1, gwspace, gwcol2= st.columns([0.45, 0.1, 0.45])
            gwcol1.plotly_chart(utils.gw_scatter(gwdf),  use_container_width=True)
            gwcol2.dataframe(gwdf, height=600)
            gwcol2.markdown(utils.gwfiledownload(gwdf), unsafe_allow_html=True)
            


        @st.cache
        def load_data():
            time_step = 'M'
            caldate = '1/1/{}'.format(sim_range[0])
            eddate = '12/31/{}'.format(sim_range[1])

            df = utils.get_sim_obd(self.area, stdate, time_step, sims, obds, caldate, eddate)
            # mfig = utils.viz_biomap()
            gwdf = utils.tot_dtw(self.area, stdate, caldate, eddate, gw_sims, gw_obds, time_step=None)
            return df, sims, gwdf


        logging.basicConfig(level=logging.CRITICAL)
        df, sims, gwdf = load_data()
        main(df, sims, gwdf)

