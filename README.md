<div align="center">

# Urban Heat Island Mitigation Using Deep Learning, LiDAR, and Satellite Remote Sensing  

### Spatial CNN-Based Green Roof Cooling Assessment in Downtown Iowa City, Iowa

<p align="center">
  <a href="https://w75frp.csb.app/">
    <img src="https://img.shields.io/badge/Live-Web_App-FFCD00?style=for-the-badge&labelColor=111111" alt="Live Web App">
  </a>
  <a href="https://mirza9003.github.io/GreenRoof-Iowa-City-CNN/GreenRoof_Interactive_Map_Iowa_FINAL_NO_OVERLAP.html">
    <img src="https://img.shields.io/badge/Interactive-Web_Map-2F6B45?style=for-the-badge&labelColor=111111" alt="Interactive Web Map">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-black?style=for-the-badge" alt="MIT License">
  </a>
  <img src="https://img.shields.io/badge/Research-GeoAI-orange?style=for-the-badge" alt="GeoAI Research">
</p>

</div>

---

<div align="center">

## Live Interactive Research Platform

This repository presents a reproducible GeoAI framework for evaluating green roof cooling potential in **Downtown Iowa City, Iowa**. The workflow integrates airborne LiDAR-derived urban morphology, satellite remote sensing, and spatial deep learning to model daytime land surface temperature, identify thermal hotspots, and simulate green roof-based urban heat mitigation.

The current interactive interface presents the **July 20, 2023 Iowa City case study**. The full modelling workflow can be reproduced for additional dates by regenerating date-specific daytime land surface temperature, NDVI, NDBI, and solar-radiation rasters.

</div>

---

<div align="center">

## Live Web Application

**Live Web Application:**  
[https://w75frp.csb.app/](https://w75frp.csb.app/)

<img src="App.png" width="1000">

<em>Interactive Spatial CNN dashboard for green roof cooling simulation, model comparison, hotspot visualization, sensitivity analysis, and reproducibility guidance.</em>

</div>

### Dashboard Capabilities

- Interactive research interface for the green roof cooling pipeline  
- Spatial CNN model architecture summary  
- Machine learning and deep learning model comparison  
- Sensitivity analysis ranking  
- Green roof cooling scenario outputs  
- Precomputed demonstration results for the July 20, 2023 Iowa City case study  

---

<div align="center">

## Interactive Web Map

**Interactive Web Map:**  
[https://mirza9003.github.io/GreenRoof-Iowa-City-CNN/GreenRoof_Interactive_Map_Iowa_FINAL_NO_OVERLAP.html](https://mirza9003.github.io/GreenRoof-Iowa-City-CNN/GreenRoof_Interactive_Map_Iowa_FINAL_NO_OVERLAP.html)

<img src="InteractiveMapLive.png" width="1000">

<em>Interactive web GIS map showing daytime land surface temperature, thermal hotspots, green roof cooling potential, building-height effects, and the study boundary for Downtown Iowa City.</em>

</div>

### Map Capabilities

- Street and satellite basemaps  
- Daytime land surface temperature visualization for July 20, 2023  
- Green roof cooling overlay  
- Thermal hotspot markers  
- Building-height layer  
- Layer toggling  
- Study summary panel  
- Iowa black and gold dashboard styling  

---

<div align="center">

## Green Roof Cooling Simulation

<img src="Fig07_GreenRoof_Simulation.png" width="1000">

<em>Spatial CNN-based green roof cooling simulation showing observed daytime land surface temperature, simulated cooling, and building-height-related cooling potential in Downtown Iowa City.</em>

</div>

---

<div align="center">

## Input Parameters

<img src="Fig01_Input_Parameters.png" width="1000">

<em>Spatial distribution of major urban morphology and remote sensing predictors used in the Spatial CNN model.</em>

</div>

---

<div align="center">

## Model Performance

<img src="Fig02_Model_Performance.png" width="1000">

<em>Comparison of machine learning and deep learning models using training, test, and 10-fold cross-validation metrics.</em>

<br><br>

<img src="Fig03_Predicted_vs_Actual.png" width="1000">

<em>Predicted versus observed daytime land surface temperature comparison for all evaluated models.</em>

<br><br>

<img src="Fig04_10Fold_CV.png" width="1000">

<em>Ten-fold cross-validation performance across all six models.</em>

</div>

---

<div align="center">

## Explainability and Sensitivity Analysis

<img src="Fig05_Sensitivity_Analysis.png" width="1000">

<em>Sensitivity analysis of input predictors based on model performance degradation when each predictor is removed from the Spatial CNN model.</em>

<br><br>

<img src="Fig06_SHAP_Summary.png" width="1000">

<em>SHAP-based explainability analysis for the Spatial CNN model.</em>

</div>

---

<div align="center">

## Urban Morphology and Cooling Relationships

<img src="Fig08_Scatter_Relationships.png" width="1000">

<em>Relationships between urban morphological variables and simulated green roof cooling response.</em>

</div>

---

## Study Title

**Climate-Resilient Urban Cooling Using LiDAR-Derived Urban Morphology and Spatial Deep Learning: A GeoAI Framework for Green Roof Assessment in Downtown Iowa City, Iowa**

---

## Study Overview

Urban heat islands intensify thermal exposure in dense built environments, particularly where impervious surfaces, low vegetation cover, and unfavorable urban morphology increase daytime heat storage. This study develops a spatial deep learning framework to estimate daytime land surface temperature and simulate green roof cooling potential in Downtown Iowa City.

The framework combines:

- LiDAR-derived urban morphology  
- Landsat 9-derived daytime land surface temperature  
- Sentinel-2-derived vegetation and built-up indices  
- Water-body proximity  
- Spatial CNN-based temperature prediction  
- Green roof scenario simulation  

The final model identifies thermal hotspots and estimates potential land surface temperature reduction under a green roof implementation scenario.

---

## Key Results

| Metric | Value |
|---|---:|
| Best model | Spatial CNN |
| Test R² | 0.982 |
| Test RMSE | 0.732 °F |
| Mean green roof cooling | 7.07 °F |
| Maximum green roof cooling | 12.18 °F |
| Hotspot threshold | Top 15% DLST pixels |
| Hotspot pixels | 125 pixels |
| Hotspot pixels with >1°F cooling | 98.4% |
| Most influential predictor | NDBI |
| Second most influential predictor | WBD, distance to Iowa River |

---

## Models Evaluated

Six machine learning and deep learning models were evaluated:

| Model | Type |
|---|---|
| Artificial Neural Network | Machine learning |
| Random Forest | Machine learning |
| XGBoost | Machine learning |
| Spatial CNN | Deep learning |
| CNN-LSTM | Deep learning |
| Vision Transformer | Deep learning |

The **Spatial CNN** achieved the strongest predictive performance because it captures local spatial context using 5 × 5 raster patches rather than relying only on center-pixel tabular predictors.

---

## Input Variables

The model uses one target raster and eight predictor rasters.

### Target Variable

| Variable | Description |
|---|---|
| DLST | Daytime land surface temperature derived from Landsat 9 thermal data |

### Predictor Variables

| Variable | Description | Source |
|---|---|---|
| BH | Building height | LiDAR |
| BRI | Building roof index | Building footprint / polygon raster |
| BVD | Building volume density | LiDAR-derived building height and roof area |
| SVF | Sky view factor | LiDAR-derived urban geometry |
| SR | Solar radiation | ArcGIS Pro area solar radiation |
| NDVI | Normalized Difference Vegetation Index | Sentinel-2 |
| NDBI | Normalized Difference Built-up Index | Sentinel-2 |
| WBD | Water body distance | Multi-index water mask and distance calculation |

---

## Methodological Workflow

### 1. LiDAR and Urban Morphology Processing

1. Process LiDAR point clouds  
2. Generate DSM and DTM  
3. Derive normalized building height  
4. Extract building footprint and roof-related metrics  
5. Compute building height, building roof index, building volume density, sky view factor, and solar radiation  

### 2. Satellite Remote Sensing Processing

6. Retrieve Landsat 9 thermal data  
7. Derive daytime land surface temperature  
8. Retrieve Sentinel-2 multispectral imagery  
9. Compute NDVI and NDBI  
10. Derive water mask and water-body distance  

### 3. GeoAI Modelling

11. Harmonize all rasters to 30 m resolution  
12. Build 5 × 5 spatial patches around each valid pixel  
13. Train machine learning and deep learning models  
14. Evaluate model performance using test metrics and 10-fold cross-validation  
15. Select the best-performing Spatial CNN model  

### 4. Green Roof Scenario Simulation

16. Identify thermal hotspots using the 85th percentile DLST threshold  
17. Assign green roof scenario values to NDVI and NDBI  
18. Predict post-intervention daytime land surface temperature  
19. Compute DLST reduction  
20. Export GeoTIFFs, tables, figures, and interactive web maps  

---

## Reproducibility Notes

The web application is designed as an interactive research interface and visual demonstration. It presents precomputed outputs for the July 20, 2023 Iowa City case study.

The full reproducible computation is implemented in:

```text
GreenRoof_FinalPipeline.py
GreenRoof_FinalPipeline.ipynb
