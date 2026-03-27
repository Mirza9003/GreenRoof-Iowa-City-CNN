# Urban Heat Island Mitigation Using Deep Learning, LiDAR, and Satellite Remote Sensing  
### Spatial CNN-Based Green Roof Cooling Assessment in Downtown Iowa City

<p align="center">

[![Live Web App](https://img.shields.io/badge/Live-Web_App-blue?style=for-the-badge)](https://w75frp.csb.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-green?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-black?style=for-the-badge)]()
[![Research](https://img.shields.io/badge/Research-GeoAI-orange?style=for-the-badge)]()

</p>

---

# Live Interactive Web Application

Explore the fully interactive Spatial CNN green-roof cooling simulation:

🌐 **Live Web App:**  
https://w75frp.csb.app/

The interactive dashboard allows users to:

- Visualize Urban Heat Island mitigation  
- Explore Spatial CNN cooling predictions  
- View hotspot reduction maps  
- Toggle model layers  
- Inspect building-level cooling potential  
- Reproduce study results  
- Explore spatial temperature distributions  

---

# Study Area

**Downtown Iowa City, Iowa, USA**

The study evaluates the cooling potential of green roof implementation using:

- LiDAR-derived urban morphology  
- Satellite remote sensing  
- Deep learning models  
- Spatial CNN architecture  

---

<p align="center">
  <img src="GR_Simulation_Maps.png" width="1000">
</p>

<p align="center">
  <em>Simulated daytime land surface temperature reduction under green roof implementation in Downtown Iowa City, Iowa.</em>
</p>

---

# Overview

This repository contains the reproducible codebase for a green roof urban heat island mitigation study in Downtown Iowa City, Iowa. The workflow integrates LiDAR-derived urban morphological variables, satellite remote sensing products, and machine learning/deep learning models to predict daytime land surface temperature (DLST) and simulate the cooling potential of green roof implementation.

The modelling framework evaluates six predictive models:

- Artificial Neural Network (ANN)  
- Random Forest  
- XGBoost  
- Spatial CNN  
- CNN-LSTM  
- Vision Transformer  

Among them, the **Spatial CNN** achieved the best predictive performance.

---

# Study Title

**Exploring the Cooling Potential of Green Roofs for Mitigating Urban Heat Islands Using LiDAR, Satellite Remote Sensing, and Spatial Convolutional Neural Networks: A Case Study of Downtown Iowa City, Iowa**

**Author**  
Mirza Md Tasnim Mukarram  
University of Iowa (2026)

---

# Key Results

| Metric | Value |
|--------|-------|
| Best model | Spatial CNN |
| Test R² | 0.974 |
| RMSE | 0.842 °F |
| K-Fold R² | 0.962 ± 0.007 |
| Mean green-roof cooling | 6.37 °F |
| Maximum green-roof cooling | 10.27 °F |
| Hotspot coverage | 100% pixels > 1 °F |

---

# Web Application Features

The interactive web application includes:

- Interactive cooling maps  
- Hotspot detection  
- Spatial CNN predictions  
- Feature importance analysis  
- DLST simulation  
- Building-level cooling analysis  
- Layer toggling  
- Satellite basemap visualization  

---

# Methodology

## Data Sources

### LiDAR Derived Variables

- Building Height (BH)  
- Building Density (BVD)  
- Sky View Factor (SVF)  
- Surface Roughness (SR)  
- Building Roof Index (BRI)  

### Satellite Variables

- NDVI  
- NDBI  
- Water Body Distance (WBD)  

---

# Models Evaluated

| Model | Type |
|------|------|
| ANN | Deep Learning |
| Random Forest | Machine Learning |
| XGBoost | Gradient Boosting |
| Spatial CNN | Deep Learning |
| CNN-LSTM | Hybrid Deep Learning |
| Vision Transformer | Transformer |

---

# Best Model

**Spatial CNN**

Performance:

- R² = 0.974  
- RMSE = 0.842°F  
- Highest spatial accuracy  
- Best generalization performance  

---

# Repository Structure

```text
├── ModelCode.py                    # Main modelling pipeline
├── GreenRoof_Pipeline.jsx          # Interactive app
├── GreenRoof_Interactive_Map_v2.html  # Interactive map
├── Fig10_GR_Simulation_Maps.png    # Main figure
├── README.md
├── LICENSE
