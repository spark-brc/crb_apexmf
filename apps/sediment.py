from hydralit.app_template import HydraHeadApp
import streamlit as st
import os
import pandas as pd
import base64
import numpy as np
import glob
import utils

pd.options.mode.chained_assignment = None


class Sed(HydraHeadApp):

    def __init__(self, area):
        self.area = area

    def run(self):
        st.markdown("<h2 style='text-align: center; color: red;'>Coming soon!</h2>", unsafe_allow_html=True)        
        test = f"""<p align="center"><img src="https://cdn.dribbble.com/users/2059463/screenshots/5432270/media/2a51fd1c009ac309a2156eb9b72b4f90.gif" width="500"></p>"""

        st.markdown(test, unsafe_allow_html=True)
        st.markdown("### Hydrograph for Sediment")
        st.markdown("### Sediment Yield Map")
