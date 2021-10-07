# Interactive Dashboard for CRB Integrated Model

This web-based interactive dashboard helps analyze APEX-MODFLOW models in the Colorado River Basin (CRB). *This dashboard under construction.*

- **Main Python libraries:** powerd by base64, pandas, streamlit, plotly, geopandas
- **Source code:** [github.com/spark-brc/crb_apexmf](https://github.com/spark-brc/crb_apexmf)

***INTRODUCTION***: Texas A&M AgriLife Research and the Bureau of Land Management (BLM) conduct an in-depth assessment of water resources, salt, fire and land management in the Colorado River Basin. We've enhanced the Agricultural Policy / Environmental eXtender (APEX) model, coupling it with USGS MODFLOW (groundwater model) and Reactive Transport Model (RT3D), and implementing salinity module. We've built 6 APEX-MODFLOW models: Animas, White, Price, Dolores, Upper Green, and Gunnison in CRB. Currently, model optimizations are in progress against stream discharge, groundwater level, and sediment yield. In addition, various researches are ongoing and include [assessment of salinity transport](#assessment-of-salinity-transport), [wildland fire simulation](#wildland-fire-simulation), [assessment of water resources and fluxes](#assessment-of-water-resources-and-fluxes), [application of machine learning in CRB](#application-of-machine-learning-in-crb), [drought & flood assessment](#drought-and-flood-assessment), and [development of model supporting utilities](#developements-of-apexmod-and-apex-cute).
<br>

### Assessment of Salinity Transport
- Salt consists of 8 major ions (SO4, Ca, Mg, Na, Cl, K, CO3, HCO3)
- Salt ion concentration simulated for each APEX soil layer, aquifer grid cell, and APEX stream
- Salt ion mass transported in surface runoff, erosion, lateral flow, and groundwater discharge
- Salt ion chemistry (precipitation-dissolution, cation exchange) simulated in soil and aquifer

*Workflow of APEX-MODFLOW-Salt*
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/salt_flow.png?raw=true" width="100%"></p>

*Simulated Groundwater Salt Ion Concentration & Simulated Salt Ion In-Stream Loading (kg/day):*
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/salt_results.png?raw=true" width="100%"></p>

<br>

### Wildland Fire Simulation
- Wildfire alters biophysical and soil properties depending on the burn severity in a watershed
- Changes in a river salt and sediment yield are one of the major consequences of fire due to changes in surface runoff processes.
- To investigate this we selected five fire events across CRB at micro-watershed scale.

*Post-fire biophysical reduction as measured by the Leaf Area Index (LAI):*
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/fire_lai.png?raw=true" width="80%"></p>

*Fire event simulation with APEX model- the 2002 Missionary Ridge fire & effect*
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/fire_results.png?raw=true" width="100%"></p>

<br>

### Assessment of Water Resources and Fluxes
*Groundwater levels & Saturated Thickness | Groundwater recharge & SW-GW interctions (Animas watershed)*
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/water.png?raw=true" width="100%"></p>

<br>

### Application of Machine Learning in CRB
- Investigation of dominant wildfires-related factors
- Detecting areas with high potential for wildfires in advance
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/ml1.png?raw=true" width="50%"></p>

- Investigation of dominant salinity-related factors in
- Mapping the spatial distribution of salinity (monthly, seasonal, annual)

<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/ml2.png?raw=true" width="50%"></p>

<br>


### Drought and Flood Assessment
- Examination of the impacts of flash droughts and floods on water quality (e.g., salt and sediment) 
- Investigation of the major sources of pollution during extreme events
- Identification, categorization, and prediction of drought
- Investigation of the impacts of drought on wildfire
<br>

### Development of Reproducible Model Optimization Framework
- A reproducible model optimization framework is developed for decision supports and to assess sensitivity analysis, uncertainty quantification, and parameter estimation (PE).

*Reproducible workflow of model optimization*
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/opt1.png?raw=true" width="100%"></p>
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/opt2.png?raw=true" width="100%"></p>

<br>

### Developements of APEXMOD and APEX-CUTE
- #### [APEXMOD](https://github.com/spark-brc/APEXMOD) 
  - Link APEX and MODFLOW
  - Configure model settings
  - Visualize model results

*Interface of APEXMOD*
<p align="center"><img src="https://github.com/spark-brc/crb_apexmf/blob/main/resources/pics/apexmod.png?raw=true" width="100%"></p>

<br>

- #### APEX-CUTE
  - Perform sensitivity analysis
  - Perform parameter estimation

<br>

<br>

### Publication
- [Bailey, R.T., Tasdighi, A., Park, S., Tavakoli-Kivi, S., Abitew, T., Jeong, J., Green, C.H. and Worqlul, A.W., 2021. APEX-MODFLOW: A New Integrated Model to Simulate Hydrological Processes in Watershed Systems. Environmental Modelling & Software, p.105093.](https://doi.org/10.1016/j.envsoft.2021.105093)
- A. Worqlul, J. Jeong, C. Green, T. Abitew, Streamflow simulation in high topographic gradients and snowmelt-dominated watershed using the APEX model - Price Watershed, Utah (Under review)
- T. Abitew, J. Jeong, C. Green, Modeling landscape wind erosion processes on rangelands using the APEX model (Internal review)
- S. Kim, S. Kim, C.H. Green, J. Jeong, Multivariate Polynomial Regression Modeling of Total Dissolved-Solids in Rangeland Stormwater Runoff in the Western United, (Internal review)
- T. Abitew, S. Park, J. Jeong, C. Green, Understanding the effects of post-wildfire treatments on hydrological responses in Colorado River Basin (In preparation)
