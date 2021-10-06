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
        # st.markdown("<h1 style='text-align:center;padding: 0px 0px;color:black;font-size:200%;'>Home2</h1>",unsafe_allow_html=True)

        mdcoll, mdmain, mdcolr = st.columns([0.1, 0.8, 0.1])
        # mdmain = mdmain.expander('{} Model Description'.format(self.area), expanded=True)
        # tdf.dataframe(df, height=500)

        intro_markdown = utils.read_markdown_file(
            os.path.join("./resources", "home.md")
        )
        mdmain.markdown(intro_markdown, unsafe_allow_html=True)