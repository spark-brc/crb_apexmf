import streamlit as st
import numpy as np
import pandas as pd
# from data.create_data import create_table

#add an import to Hydralit
from hydralit import HydraHeadApp

#create a wrapper class
class MySmallApp(HydraHeadApp):

#wrap all your code in this method and you should be done
    def run(self):
        #-------------------existing untouched code------------------------------------------
        st.title('Small Application with a table and chart.')

        st.markdown("### Plot")
        # df = create_table()

        # st.line_chart(df)