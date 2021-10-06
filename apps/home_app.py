from hydralit.app_template import HydraHeadApp
import streamlit as st
import os
import pandas as pd
import base64
import numpy as np
import glob
import utils

pd.options.mode.chained_assignment = None


class HomeApp(HydraHeadApp):

    def __init__(self, area):
        self.area = area

    def run(self):
        st.title('CRB APEX-MODFLOW model performance')
        st.markdown("""
        This app helps analyze CRB APEX-MODFLOW model performance.
        - **Main Python libraries:** base64, pandas, streamlit, plotly, geopandas
        - **Source code:** [github.com/spark-brc/crb_apexmf](https://github.com/spark-brc/crb_apexmf)
        """)
        # st.markdown("<h1 style='text-align:center;padding: 0px 0px;color:black;font-size:200%;'>Home2</h1>",unsafe_allow_html=True)

        mdcoll, mdmain, mdcolr = st.columns([0.1, 0.8, 0.1])
        # mdmain = mdmain.expander('{} Model Description'.format(self.area), expanded=True)
        # tdf.dataframe(df, height=500)

        intro_markdown = utils.read_markdown_file(
            os.path.join("./resources/watershed", "Animas/description", "Animas APEX-MODFLOW.md")
        )
        mdmain.markdown(intro_markdown, unsafe_allow_html=True)