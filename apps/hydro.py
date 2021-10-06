import logging
from hydralit.app_template import HydraHeadApp
import plotly.express as px
import streamlit as st
import os
import pandas as pd
import numpy as np
import utils

pd.options.mode.chained_assignment = None


class Hydro(HydraHeadApp):

    def __init__(self, area):
        self.area = area


    def run(self):

        st.markdown("<h2 style='text-align: center; color: red;'>Coming soon!</h2>", unsafe_allow_html=True)        
        test = f"""<p align="center"><img src="https://cdn.dribbble.com/users/2059463/screenshots/5432270/media/2a51fd1c009ac309a2156eb9b72b4f90.gif" width="500"></p>"""

        st.markdown(test, unsafe_allow_html=True)
        st.markdown("### Water Balance")
        st.markdown("### Surface water and Groundwater Interaction")
        st.markdown('### Groundwater Level Map')



