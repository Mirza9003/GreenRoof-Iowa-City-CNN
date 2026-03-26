# GreenRoof-Iowa-City-CNN
Spatial CNN for green roof UHI mitigation — Downtown Iowa City
# Green Roof Urban Heat Island Mitigation — Downtown Iowa City

## Overview
This repository contains the complete reproducible codebase for:

**"Exploring the Cooling Potential of Green Roofs for Mitigating 
Urban Heat Islands Using LiDAR, Satellite Remote Sensing, and 
Spatial Convolutional Neural Networks: A Case Study of 
Downtown Iowa City, Iowa"**

*Mirza Md Tasnim Mukarram, University of Iowa (2026)*

---

## Key Results
| Metric | Value |
|--------|-------|
| Best model | Spatial CNN |
| Test R² | 0.974 |
| RMSE | 0.842°F |
| K-Fold R² | 0.962 ± 0.007 |
| Mean GR cooling | 6.37°F |
| Max GR cooling | 10.27°F |
| Hotspot coverage | 100% pixels > 1°F |

---

## Repository Structure
```
├── GreenRoof_Publication_Final.py   # Main pipeline — all 6 models,
│                                    # 11 figures, 5 tables
├── GreenRoof_Pipeline.jsx           # Interactive reproducibility app
├── GreenRoof_Interactive_Map.html   # Folium web map (open in browser)
└── README.md
```

---

## Requirements
```bash
pip install numpy pandas rasterio matplotlib tensorflow==2.21 
            xgboost shap scikit-learn folium branca geopandas
```

Python 3.11 | TensorFlow 2.21 | scikit-learn 1.6 | SHAP 0.45

---

## Data Access
Input rasters (LiDAR-derived + satellite) are available at:
**[Add your Google Drive or Zenodo link here]**

| File | Source | Resolution |
|------|--------|------------|
| DLST_IowaCity_20230720.tif | Landsat 9 / GEE | 30m |
| NDVI_July20_IowaCity.tif | Sentinel-2 / GEE | 30m |
| NDBI_July20_IowaCity.tif | Sentinel-2 / GEE | 30m |
| WBD_30m.tif | Sentinel-2 / GEE | 30m |
| BH_30m.tif | LiDAR / ArcGIS Pro | 30m |
| BRI_30m.tif | LiDAR / ArcGIS Pro | 30m |
| BVD_30m.tif | LiDAR / ArcGIS Pro | 30m |
| SR_30m.tif | LiDAR / ArcGIS Pro | 30m |
| SVF_30m.tif | LiDAR / ArcGIS Pro | 30m |

---

## Citation
If you use this code, please cite:

Mukarram, M. M. T. (2026). Exploring the cooling potential of green 
roofs for mitigating urban heat islands using LiDAR, satellite remote 
sensing, and spatial convolutional neural networks: A case study of 
Downtown Iowa City, Iowa. *Sustainable Cities and Society*.

---

## License
MIT License — free to use with attribution.
