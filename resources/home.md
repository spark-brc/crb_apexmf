# Interactive Dashboard for CRB Integrated model
This web-based interactive dashboard helps analyze APEX-MODFLOW models in the Colorado River Basin (CRB). *This dashboard under construction.*
- **Main Python libraries:** powerd by base64, pandas, streamlit, plotly, geopandas
- **Source code:** [github.com/spark-brc/crb_apexmf](https://github.com/spark-brc/crb_apexmf)

***INTRODUCTION***: Texas A&M AgriLife Research and the Bureau of Land Management (BLM) conduct an in-depth assessment of water resources, salt, fire and land management in the Colorado River Basin. We've enhanced the Agricultural Policy / Environmental eXtender (APEX) model, coupling it with USGS MODFLOW (groundwater model) and Reactive Transport Model (RT3D), and implementing salinity module. We've built 6 APEX-MODFLOW models: Animas, White, Price, Dolores, Upper Green, and Gunnison in CRB. Currently, model optimizations are in progress against stream discharge, groundwater level, and sediment yield. In addition, various researches are ongoing, including:
- Assessment of salinity transport
	- Salt consists of 8 major ions (SO4, Ca, Mg, Na, Cl, K, CO3, HCO3)
	- Salt ion concentration simulated for each APEX soil layer, aquifer grid cell, and APEX stream
	- Salt ion mass transported in surface runoff, erosion, lateral flow, and groundwater discharge
	- Salt ion chemistry (precipitation-dissolution, cation exchange) simulated in soil and aquifer
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/salt_flow.png?raw=true" width="1000"></p>
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/salt_results.png?raw=true" width="1000"></p>