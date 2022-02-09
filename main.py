import logging
import plotly.express as px
import streamlit as st
import os
import pandas as pd
import base64
import numpy as np
import glob
import utils
from hydralit import HydraApp
import apps

pd.options.mode.chained_assignment = None

st.set_page_config(
    layout="wide",
    initial_sidebar_state='auto',
    page_title='CRB APEX-MODFLOW',
    page_icon='icon2.png',
    )

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
display:inline;
bottom: 0;
width: 170px;
background-color: transparent;
color: black;
text-align: center;
padding-bottom:140px;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://www.heflin.dev/" target="_blank">Seonggyu Park</a></p>
</div>
"""

# st.sidebar.markdown(footer,unsafe_allow_html=True)


LOGO_IMAGE = "./resources/TAMUAgriLifeResearchLogo.png"
LOGO_IMAGE2 = "./resources/blm-logo.png"
st.sidebar.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        color: #f9a01b !important;
        padding-top: 75px !important;
    }
    .logo-img {
        z-index: 1;
        display:inline;
        position:fixed;
        bottom:0;
        padding-bottom:45px;
        width:120px;
        background-color: transparent;
        margin-left:25px;
    }
        .logo-img2 {
        z-index: 1;
        display:inline;
        position:fixed;
        bottom:0;
        margin-left:150px;
        padding-bottom:25px;
        width:90px;
        background-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if __name__ == '__main__':
    #---ONLY HERE TO SHOW OPTIONS WITH HYDRALIT - NOT REQUIRED, use Hydralit constructor parameters.

    ws_nams, full_paths = utils.get_watershed_list()
    # Create Radio Buttons

    area = st.sidebar.radio('Select Watershed', ws_nams)
    st.markdown(
        f'''
            <style>
                .sidebar .sidebar-content {{
                    width: 1px;
                }}
            </style>
        ''',
        unsafe_allow_html=True
)
    # st.sidebar.markdown("""<br><br><br><br>""", unsafe_allow_html=True)
    # st.sidebar.image('./resources/TAMUAgriLifeResearchLogo.png',width=200)
    # st.sidebar.image('./resources/blm-logo.png', width=200)
    # st.markdown(footer,unsafe_allow_html=True)

    over_theme = {'txc_inactive': '#FFFFFF'}
    
    #this is the host application, we add children to it and that's it!
    app = HydraApp(
        title='Secure Hydralit Data Explorer',
        favicon="🐙",
        # hide_streamlit_markers=True,
        ##add a nice banner, this banner has been defined as 5 sections with spacing defined by the banner_spacing array below.
        # use_banner_images=["./resources/hydra.png",None,{'header':"<h1 style='text-align:center;padding: 0px 0px;color:grey;font-size:200%;'>Secure Hydralit Explorer</h1><br>"},None,"./resources/lock.png"], 
        # banner_spacing=[5,30,60,30,5],
        use_navbar=True, 
        navbar_sticky=True,
        navbar_animation=True,
        navbar_theme=over_theme
    )

    app.add_app(
        'Home', icon='🏠', app=apps.HomeApp(area),
        is_home=True
        )
    
    app.add_app('Model Performance', icon="✔️", app=apps.ModelPerform(area))
    app.add_app('Hydrology', icon="🏞️", app=apps.Hydro(area))
    app.add_app('Sediment', icon="⏳", app=apps.Sed(area))
    app.add_app('Salt', icon="🧂", app=apps.Salt(area))
    app.add_app('Fire', icon="🔥", app=apps.Fire(area))
    app.add_app('Model Information', icon="ℹ️", app=apps.ModelInfo(area))    
    # app.add_app('Contact', icon="📞", app=apps.Contact(area))
    app.add_loader_app(apps.MyLoadingApp(delay=0))

    # #if we want to auto login a guest but still have a secure app, we can assign a guest account and go straight in
    # app.enable_guest_access()

    # #check user access level to determine what should be shown on the menu
    # user_access_level, username = app.check_access()

    # # If the menu is cluttered, just rearrange it into sections!
    # # completely optional, but if you have too many entries, you can make it nicer by using accordian menus
    # if user_access_level > 1:
    #     complex_nav = {
    #         'Home': ['Home'],
    #         'Loader Playground': ['Loader Playground'],
    #         'Intro 🏆': ['Cheat Sheet',"Solar Mach"],
    #         'Hotstepper 🔥': ["Sequency Denoising","Sequency (Secure)"],
    #         'Clustering': ["Uber Pickups"],
    #         'NLP': ["Spacy NLP"],
    #         'Cookie Cutter': ['Cookie Cutter']
    #     }
    # elif user_access_level == 1:
    #     complex_nav = {
    #         'Home': ['Home'],
    #         'Loader Playground': ['Loader Playground'],
    #         'Intro 🏆': ['Cheat Sheet',"Solar Mach"],
    #         'Hotstepper 🔥': ["Sequency Denoising"],
    #         'Clustering': ["Uber Pickups"],
    #         'NLP': ["Spacy NLP"],
    #         'Cookie Cutter': ['Cookie Cutter']
    #     }
    # else:
    #     complex_nav = {
    #         'Home': ['Home'],
    #     }

    st.sidebar.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <img class="logo-img2" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE2, "rb").read()).decode()}">
        </div>
        """,
        unsafe_allow_html=True
    )
    # complex_nav = {
    #         'Home': ['Home']
    # }
    # #and finally just the entire app and all the children.
    # app.run(complex_nav)
    app.run()

