# ============================================================
# ALL PUBLICATION FIGURES — Iowa City Green Roof Study
# ============================================================

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    HAS_SCALEBAR = True
except ImportError:
    HAS_SCALEBAR = False

folder     = r"E:\LIDAR\Lidar_Lab3_mmtm\Lidar_Lab3_mmtm\GreenRoof_IowaCity"
out_folder = r"E:\LIDAR\Lidar_Lab3_mmtm\Lidar_Lab3_mmtm\GreenRoof_IowaCity\Final_Figures"

# ============================================================
# Global style — publication standard
# ============================================================
plt.rcParams.update({
    'font.family':        'Arial',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.titleweight':   'bold',
    'axes.labelsize':     11,
    'axes.labelweight':   'bold',
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.dpi':         150,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15
})

ML_COLOR  = '#16a085'
DL_COLOR  = '#2980b9'
BEST_COLOR= '#8e44ad'

MODEL_COLORS = {
    'ANN':                '#27ae60',
    'Random Forest':      '#2ecc71',
    'XGBoost':            '#16a085',
    'Spatial CNN':        '#2980b9',
    'CNN-LSTM':           '#8e44ad',
    'Vision Transformer': '#c0392b'
}

# ============================================================
# Helpers
# ============================================================
def load_raster(path, ref_shape=None):
    with rasterio.open(path) as src:
        if ref_shape and src.shape != ref_shape:
            data = src.read(1, out_shape=ref_shape,
                          resampling=Resampling.bilinear
                          ).astype(np.float32)
        else:
            data = src.read(1).astype(np.float32)
        if src.nodata:
            data[data == src.nodata] = np.nan
        return data, src.transform, src.crs

def remove_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])

def clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def add_north_arrow(ax, x=0.92, y=0.93):
    ax.annotate('N', xy=(x, y), xytext=(x, y-0.09),
                xycoords='axes fraction',
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->',
                               color='black', lw=2))

def add_scalebar(ax, transform):
    if HAS_SCALEBAR:
        sb = ScaleBar(abs(transform[0]), 'm',
                     length_fraction=0.25,
                     location='lower right',
                     box_alpha=0.8,
                     font_properties={'size': 8})
        ax.add_artist(sb)

def save_fig(fig, name):
    fig.savefig(f"{out_folder}\\{name}.png", dpi=300)
    fig.savefig(f"{out_folder}\\{name}.pdf")
    print(f"  Saved {name}")
    plt.close(fig)

# ============================================================
# STEP 1 — Load rasters
# ============================================================
print("Loading rasters...")
dlst_j20, tr, crs = load_raster(
    f"{folder}\\DLST_IowaCity_20230720.tif")
dlst_j4,  _,  _   = load_raster(
    f"{folder}\\DLST_IowaCity_20230704.tif",
    dlst_j20.shape)
ref_shape = dlst_j20.shape

files = {
    'NDVI': f"{folder}\\NDVI_July20_IowaCity.tif",
    'NDBI': f"{folder}\\NDBI_July20_IowaCity.tif",
    'WBD':  f"{folder}\\WBD_30m.tif",
    'BH':   f"{folder}\\BH_30m.tif",
    'BRI':  f"{folder}\\BRI_30m.tif",
    'BVD':  f"{folder}\\BVD_30m.tif",
    'SR':   f"{folder}\\SR_30m.tif",
    'SVF':  f"{folder}\\SVF_30m.tif"
}

arrays = {'DLST': dlst_j20}
for key, path in files.items():
    arr, _, _ = load_raster(path, ref_shape)
    arrays[key] = arr

feature_keys = ['BH','BRI','BVD','SVF','SR','NDVI','NDBI','WBD']
rows, cols   = ref_shape

# ============================================================
# STEP 2 — Prepare datasets
# ============================================================
df = pd.DataFrame({k: arrays[k].flatten()
                   for k in ['DLST'] + feature_keys})
df_clean = df.dropna()
X_flat   = df_clean[feature_keys].values
y_flat   = df_clean['DLST'].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)
X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_flat, test_size=0.2, random_state=42)

# Patch dataset for DL
PATCH_SIZE = 5
pad        = PATCH_SIZE // 2
feature_stack = np.stack([arrays[k] for k in feature_keys], axis=-1)
feature_pad   = np.pad(feature_stack,
                       ((pad,pad),(pad,pad),(0,0)),
                       mode='reflect')

patches, targets, positions = [], [], []
for i in range(rows):
    for j in range(cols):
        if np.isnan(arrays['DLST'][i,j]):
            continue
        patch = feature_pad[i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]
        if np.any(np.isnan(patch)):
            continue
        patches.append(patch)
        targets.append(arrays['DLST'][i,j])
        positions.append((i,j))

X_patches = np.array(patches, dtype=np.float32)
y_targets = np.array(targets, dtype=np.float32)

X_norm = X_patches.copy()
ch_means, ch_stds = [], []
for c in range(X_patches.shape[-1]):
    m = X_patches[:,:,:,c].mean()
    s = X_patches[:,:,:,c].std() + 1e-8
    X_norm[:,:,:,c] = (X_patches[:,:,:,c] - m) / s
    ch_means.append(m)
    ch_stds.append(s)

y_mean = y_targets.mean()
y_std  = y_targets.std()
y_norm = (y_targets - y_mean) / y_std

Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
    X_norm, y_norm, test_size=0.2, random_state=42)

print(f"Valid pixels: {len(df_clean)}, Patches: {len(X_patches)}")

# ============================================================
# STEP 3 — Train all models + record training metrics
# ============================================================
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=15, restore_best_weights=True, verbose=0),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=7, verbose=0)
]

results  = {}
tr_r2s   = {}  # training R²
histories= {}  # training histories

print("\nTraining all models...")

# --- ANN ---
ann = MLPRegressor(hidden_layer_sizes=(95,10),
                   activation='relu', solver='lbfgs',
                   max_iter=2000, random_state=42)
ann.fit(X_tr, y_tr)
yp_tr_ann = ann.predict(X_tr)
yp_te_ann = ann.predict(X_te)
tr_r2s['ANN']   = r2_score(y_tr, yp_tr_ann)
results['ANN']  = {
    'r2':   r2_score(y_te, yp_te_ann),
    'rmse': np.sqrt(mean_squared_error(y_te, yp_te_ann)),
    'type': 'ML',
    'y_pred': yp_te_ann,
    'y_true': y_te
}
print(f"  ANN: Train R²={tr_r2s['ANN']:.3f}, "
      f"Test R²={results['ANN']['r2']:.3f}")

# --- Random Forest ---
rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                           random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
yp_tr_rf = rf.predict(X_tr)
yp_te_rf = rf.predict(X_te)
tr_r2s['Random Forest']  = r2_score(y_tr, yp_tr_rf)
results['Random Forest'] = {
    'r2':   r2_score(y_te, yp_te_rf),
    'rmse': np.sqrt(mean_squared_error(y_te, yp_te_rf)),
    'type': 'ML',
    'y_pred': yp_te_rf,
    'y_true': y_te
}
print(f"  RF:  Train R²={tr_r2s['Random Forest']:.3f}, "
      f"Test R²={results['Random Forest']['r2']:.3f}")

# --- XGBoost ---
xgb_m = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbosity=0)
xgb_m.fit(X_tr, y_tr, verbose=False)
yp_tr_xgb = xgb_m.predict(X_tr)
yp_te_xgb = xgb_m.predict(X_te)
tr_r2s['XGBoost']  = r2_score(y_tr, yp_tr_xgb)
results['XGBoost'] = {
    'r2':   r2_score(y_te, yp_te_xgb),
    'rmse': np.sqrt(mean_squared_error(y_te, yp_te_xgb)),
    'type': 'ML',
    'y_pred': yp_te_xgb,
    'y_true': y_te
}
print(f"  XGB: Train R²={tr_r2s['XGBoost']:.3f}, "
      f"Test R²={results['XGBoost']['r2']:.3f}")

# --- Spatial CNN ---
def build_cnn(s):
    inp = keras.Input(shape=s)
    x = layers.Conv2D(32,(3,3),padding='same',activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128,(3,3),padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64,activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32,activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    return keras.Model(inp, layers.Dense(1)(x))

cnn = build_cnn((PATCH_SIZE,PATCH_SIZE,len(feature_keys)))
cnn.compile(optimizer=keras.optimizers.Adam(0.001),
            loss='mse', metrics=['mae'])
hist = cnn.fit(Xp_tr, yp_tr, validation_split=0.15,
               epochs=100, batch_size=32,
               callbacks=callbacks, verbose=0)
histories['Spatial CNN'] = hist

yp_tr_cnn_n = cnn.predict(Xp_tr,verbose=0).ravel()
yp_te_cnn_n = cnn.predict(Xp_te,verbose=0).ravel()
yp_tr_cnn   = yp_tr_cnn_n * y_std + y_mean
yp_te_cnn   = yp_te_cnn_n * y_std + y_mean
yt_tr_cnn   = yp_tr * y_std + y_mean
yt_te_cnn   = yp_te * y_std + y_mean

tr_r2s['Spatial CNN']  = r2_score(yt_tr_cnn, yp_tr_cnn)
results['Spatial CNN'] = {
    'r2':   r2_score(yt_te_cnn, yp_te_cnn),
    'rmse': np.sqrt(mean_squared_error(yt_te_cnn, yp_te_cnn)),
    'type': 'DL',
    'y_pred': yp_te_cnn,
    'y_true': yt_te_cnn
}
print(f"  CNN: Train R²={tr_r2s['Spatial CNN']:.3f}, "
      f"Test R²={results['Spatial CNN']['r2']:.3f}")

# --- CNN-LSTM ---
def build_cnn_lstm(p, n):
    inp = keras.Input(shape=(p,p,n))
    x = layers.Conv2D(32,(3,3),padding='same',activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Reshape((p, p*64))(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32,activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16,activation='relu')(x)
    return keras.Model(inp, layers.Dense(1)(x))

cnn_lstm = build_cnn_lstm(PATCH_SIZE,len(feature_keys))
cnn_lstm.compile(optimizer=keras.optimizers.Adam(0.001),
                 loss='mse', metrics=['mae'])
hist2 = cnn_lstm.fit(Xp_tr, yp_tr, validation_split=0.15,
                     epochs=100, batch_size=32,
                     callbacks=callbacks, verbose=0)
histories['CNN-LSTM'] = hist2

yp_tr_lstm_n = cnn_lstm.predict(Xp_tr,verbose=0).ravel()
yp_te_lstm_n = cnn_lstm.predict(Xp_te,verbose=0).ravel()
yp_tr_lstm   = yp_tr_lstm_n * y_std + y_mean
yp_te_lstm   = yp_te_lstm_n * y_std + y_mean

tr_r2s['CNN-LSTM']  = r2_score(yt_tr_cnn, yp_tr_lstm)
results['CNN-LSTM'] = {
    'r2':   r2_score(yt_te_cnn, yp_te_lstm),
    'rmse': np.sqrt(mean_squared_error(yt_te_cnn, yp_te_lstm)),
    'type': 'DL',
    'y_pred': yp_te_lstm,
    'y_true': yt_te_cnn
}
print(f"  LSTM: Train R²={tr_r2s['CNN-LSTM']:.3f}, "
      f"Test R²={results['CNN-LSTM']['r2']:.3f}")

# --- Vision Transformer ---
def mlp_b(x, h, o, d=0.1):
    x = layers.Dense(h,activation='gelu')(x)
    x = layers.Dropout(d)(x)
    x = layers.Dense(o)(x)
    x = layers.Dropout(d)(x)
    return x

def build_vit(p, n, dim=32, nh=4, nb=2):
    inp = keras.Input(shape=(p,p,n))
    x   = layers.Reshape((p, p*n))(inp)
    x   = layers.Dense(dim)(x)
    pos = tf.range(start=0, limit=p, delta=1)
    pe  = layers.Embedding(p, dim)(pos)
    x   = x + pe
    for _ in range(nb):
        x1  = layers.LayerNormalization(epsilon=1e-6)(x)
        att = layers.MultiHeadAttention(
            num_heads=nh, key_dim=dim//nh,
            dropout=0.1)(x1, x1)
        x   = layers.Add()([att, x])
        x2  = layers.LayerNormalization(epsilon=1e-6)(x)
        x   = layers.Add()([mlp_b(x2,dim*2,dim), x])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32,activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(1)(x))

vit = build_vit(PATCH_SIZE,len(feature_keys))
vit.compile(optimizer=keras.optimizers.Adam(0.0005),
            loss='mse', metrics=['mae'])
hist3 = vit.fit(Xp_tr, yp_tr, validation_split=0.15,
                epochs=100, batch_size=32,
                callbacks=callbacks, verbose=0)
histories['Vision Transformer'] = hist3

yp_tr_vit_n = vit.predict(Xp_tr,verbose=0).ravel()
yp_te_vit_n = vit.predict(Xp_te,verbose=0).ravel()
yp_tr_vit   = yp_tr_vit_n * y_std + y_mean
yp_te_vit   = yp_te_vit_n * y_std + y_mean

tr_r2s['Vision Transformer']  = r2_score(yt_tr_cnn, yp_tr_vit)
results['Vision Transformer'] = {
    'r2':   r2_score(yt_te_cnn, yp_te_vit),
    'rmse': np.sqrt(mean_squared_error(yt_te_cnn, yp_te_vit)),
    'type': 'DL',
    'y_pred': yp_te_vit,
    'y_true': yt_te_cnn
}
print(f"  ViT:  Train R²={tr_r2s['Vision Transformer']:.3f}, "
      f"Test R²={results['Vision Transformer']['r2']:.3f}")

# ============================================================
# STEP 4 — K-Fold CV
# ============================================================
print("\nK-Fold CV (k=5)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_res = {}

for name, model in [('ANN',ann),
                    ('Random Forest',rf),
                    ('XGBoost',xgb_m)]:
    r2s, rmses = [], []
    for ti, vi in kf.split(X_scaled):
        model.fit(X_scaled[ti], y_flat[ti])
        yp = model.predict(X_scaled[vi])
        r2s.append(r2_score(y_flat[vi], yp))
        rmses.append(np.sqrt(mean_squared_error(y_flat[vi], yp)))
    kf_res[name] = {
        'r2': np.mean(r2s), 'r2_std': np.std(r2s),
        'rmse': np.mean(rmses), 'rmse_std': np.std(rmses),
        'folds': list(zip(r2s, rmses))
    }

for name, model in [('Spatial CNN',cnn),
                    ('CNN-LSTM',cnn_lstm),
                    ('Vision Transformer',vit)]:
    r2s, rmses = [], []
    for ti, vi in kf.split(X_norm):
        model.fit(X_norm[ti], y_norm[ti],
                  validation_split=0.1,
                  epochs=50, batch_size=32,
                  callbacks=callbacks, verbose=0)
        yp   = model.predict(X_norm[vi],verbose=0).ravel()
        yp_o = yp * y_std + y_mean
        ya_o = y_norm[vi] * y_std + y_mean
        r2s.append(r2_score(ya_o, yp_o))
        rmses.append(np.sqrt(mean_squared_error(ya_o, yp_o)))
    kf_res[name] = {
        'r2': np.mean(r2s), 'r2_std': np.std(r2s),
        'rmse': np.mean(rmses), 'rmse_std': np.std(rmses),
        'folds': list(zip(r2s, rmses))
    }
    print(f"  {name}: R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}")

# ============================================================
# STEP 5 — SA + SHAP + GR simulation
# ============================================================
print("\nSensitivity Analysis...")
all_pred_n = cnn.predict(X_norm,verbose=0).ravel()
all_pred   = all_pred_n * y_std + y_mean
base_mse   = mean_squared_error(y_targets, all_pred)
base_r2    = r2_score(y_targets, all_pred)
sa = {}

for i, feat in enumerate(feature_keys):
    Xp = X_norm.copy()
    Xp[:,:,:,i] = 0
    yp = cnn.predict(Xp,verbose=0).ravel() * y_std + y_mean
    sa[feat] = {
        'mse':       mean_squared_error(y_targets, yp),
        'delta_mse': mean_squared_error(y_targets, yp) - base_mse,
        'r2':        r2_score(y_targets, yp),
        'delta_r2':  base_r2 - r2_score(y_targets, yp),
        'rmse':      np.sqrt(mean_squared_error(y_targets, yp))
    }

print("\nCNN SHAP (DeepExplainer)...")
bg = Xp_tr[:50]
exp = shap.DeepExplainer(cnn, bg)
sv_raw = exp.shap_values(Xp_te)

if isinstance(sv_raw, list):
    sv = sv_raw[0]
else:
    sv = sv_raw

sv = np.array(sv)
while sv.ndim > 4:
    sv = sv.squeeze(-1)
if sv.ndim == 4 and sv.shape[-1] == 1:
    sv = sv.squeeze(-1)
if sv.ndim == 4:
    sv_m = sv.mean(axis=(1,2))
elif sv.ndim == 3:
    sv_m = sv.mean(axis=1)
else:
    sv_m = sv
sv_m = sv_m[:, :len(feature_keys)]
shap_df = pd.DataFrame(sv_m, columns=feature_keys)
shap_mean = {f: np.abs(shap_df[f]).mean() for f in feature_keys}
print("SHAP done!")

print("\nGreen Roof Simulation...")
threshold   = np.percentile(y_targets, 85)
hotspot_idx = np.where(y_targets >= threshold)[0]
X_gr        = X_norm.copy()
X_gr[hotspot_idx,:,:,feature_keys.index('NDVI')] = (
    0.55 - ch_means[feature_keys.index('NDVI')]) / ch_stds[feature_keys.index('NDVI')]
X_gr[hotspot_idx,:,:,feature_keys.index('NDBI')] = (
    -0.178 - ch_means[feature_keys.index('NDBI')]) / ch_stds[feature_keys.index('NDBI')]

gr_pred   = cnn.predict(X_gr,verbose=0).ravel()*y_std+y_mean
reduction = all_pred - gr_pred
hot_red   = reduction[hotspot_idx]

reduction_map = np.full((rows,cols), np.nan, dtype=np.float32)
for idx,(i,j) in enumerate(positions):
    reduction_map[i,j] = reduction[idx]

print(f"  Mean reduction: {hot_red.mean():.2f}°F")
print(f"  Max reduction:  {hot_red.max():.2f}°F")
print(f"  >1°F pixels:    {(hot_red>1).mean()*100:.1f}%")

# ============================================================
# FIGURE 1 — Urban morphological input parameters
# ============================================================
print("\nFigure 1...")
fig = plt.figure(figsize=(15, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.35, wspace=0.3)

panels = [
    (dlst_j20, 'RdYlBu_r',  90,   125,
     '(a) Daytime Land Surface Temperature (°F)', '°F'),
    (arrays['NDVI'], 'RdYlGn', -0.2, 0.85,
     '(b) Normalized Difference Vegetation Index', 'NDVI'),
    (arrays['NDBI'], 'RdBu_r', -0.45, 0.3,
     '(c) Normalized Difference Built-up Index', 'NDBI'),
    (arrays['BH'],   'YlOrRd',  0,   45,
     '(d) Building Height (m)', 'm'),
    (arrays['SVF'],  'gray',   0.5,  1.0,
     '(e) Sky View Factor', 'SVF'),
    (arrays['WBD'],  'Blues_r', 0, 1000,
     '(f) Distance to Water Bodies (m)', 'm'),
]

for idx, (data, cmap, vmin, vmax, title, label) in enumerate(panels):
    ax = fig.add_subplot(gs[idx//3, idx%3])
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10, pad=5)
    remove_ticks(ax)
    add_north_arrow(ax)
    add_scalebar(ax, tr)

fig.suptitle(
    'Figure 1. Urban Morphological Input Parameters\n'
    'Downtown Iowa City, Iowa — July 20, 2023',
    fontsize=13, fontweight='bold', y=1.01)
save_fig(fig, 'Fig01_Input_Parameters')

# ============================================================
# FIGURE 2 — DLST maps (both scenes)
# ============================================================
print("Figure 2...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (data, date, title) in zip(axes, [
    (dlst_j4,  'July 4, 2023',
     f'(a) DLST — July 4, 2023\n'
     f'Range: {np.nanmin(dlst_j4):.1f}–{np.nanmax(dlst_j4):.1f}°F'),
    (dlst_j20, 'July 20, 2023',
     f'(b) DLST — July 20, 2023\n'
     f'Range: {np.nanmin(dlst_j20):.1f}–{np.nanmax(dlst_j20):.1f}°F')
]):
    im = ax.imshow(data, cmap='RdYlBu_r', vmin=90, vmax=130)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('°F', fontsize=10)
    ax.set_title(title, fontsize=11)
    remove_ticks(ax)
    add_north_arrow(ax)
    add_scalebar(ax, tr)

fig.suptitle(
    'Figure 2. Daytime Land Surface Temperature (DLST) —\n'
    'Downtown Iowa City, Iowa (Landsat 9)',
    fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'Fig02_DLST_Maps')

# ============================================================
# FIGURE 3 — Model performance (R² and RMSE) — FIXED
# ============================================================
print("Figure 3...")
model_names = list(MODEL_COLORS.keys())
colors      = list(MODEL_COLORS.values())
r2_vals     = [results[m]['r2']       for m in model_names]
rmse_vals   = [results[m]['rmse']     for m in model_names]
r2_tr_vals  = [tr_r2s[m]             for m in model_names]
kf_r2_vals  = [kf_res[m]['r2']       for m in model_names]
kf_r2_stds  = [kf_res[m]['r2_std']   for m in model_names]
kf_rm_vals  = [kf_res[m]['rmse']     for m in model_names]
kf_rm_stds  = [kf_res[m]['rmse_std'] for m in model_names]

short_names = ['ANN', 'RF', 'XGBoost',
               'Spatial CNN', 'CNN-LSTM', 'Vision\nTransformer']

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plt.subplots_adjust(bottom=0.22, wspace=0.35)

x = np.arange(len(model_names))
w = 0.26

# ── R² panel ──────────────────────────────────────────────
ax = axes[0]
b1 = ax.bar(x-w, r2_tr_vals, w, label='Training R²',
            color=colors, edgecolor='black', linewidth=0.5)
b2 = ax.bar(x,   r2_vals,    w, label='Test R²',
            color=colors, edgecolor='black', linewidth=0.5, alpha=0.7)
b3 = ax.bar(x+w, kf_r2_vals, w, label='K-Fold R²',
            color=colors, edgecolor='black', linewidth=0.5, alpha=0.45,
            yerr=kf_r2_stds, capsize=3, error_kw={'linewidth':1.2})

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=9, ha='center')
ax.set_ylim(0, 1.22)
ax.set_ylabel('R²', fontweight='bold')
ax.set_title('(a) Model Performance (R²)\nTraining vs Test vs K-Fold',
             fontsize=11, fontweight='bold')
ax.axhline(0.7, color='gray', linestyle='--', linewidth=1,
           alpha=0.7, label='R²=0.7 threshold')

# Value labels on test bars only
for bar, val in zip(b2, r2_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.012,
            f'{val:.3f}', ha='center', fontsize=7, fontweight='bold')

# ML/DL bracket — placed well below x-axis
ax.annotate('', xy=(2.55, -0.14), xytext=(-0.45, -0.14),
            xycoords='data',
            arrowprops=dict(arrowstyle='-', color=ML_COLOR, lw=2))
ax.text(1.05, -0.11, 'ML Models',
        color=ML_COLOR, fontsize=8, fontweight='bold', ha='center')

ax.annotate('', xy=(5.45, -0.14), xytext=(2.65, -0.14),
            xycoords='data',
            arrowprops=dict(arrowstyle='-', color=DL_COLOR, lw=2))
ax.text(4.05, -0.11, 'DL Models',
        color=DL_COLOR, fontsize=8, fontweight='bold', ha='center')

ax.set_ylim(-0.18, 1.22)
clean_axes(ax)

# ── RMSE panel ─────────────────────────────────────────────
ax = axes[1]
b1 = ax.bar(x,      rmse_vals,  w*1.5, label='Test RMSE',
            color=colors, edgecolor='black', linewidth=0.5)
b2 = ax.bar(x+w*1.5, kf_rm_vals, w*1.5, label='K-Fold RMSE',
            color=colors, edgecolor='black', linewidth=0.5, alpha=0.6,
            yerr=kf_rm_stds, capsize=3, error_kw={'linewidth':1.2})

ax.set_xticks(x + w*0.75)
ax.set_xticklabels(short_names, fontsize=9, ha='center')
ax.set_ylabel('RMSE (°F)', fontweight='bold')
ax.set_title('(b) Model Error (RMSE °F)\nTest vs K-Fold',
             fontsize=11, fontweight='bold')

for bar, val in zip(b1, rmse_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.02,
            f'{val:.3f}', ha='center', fontsize=7, fontweight='bold')
clean_axes(ax)

# ── Shared legend — clean horizontal row at bottom ─────────
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor=colors[0], label='Training R²', alpha=1.0,
          edgecolor='black', linewidth=0.5),
    Patch(facecolor=colors[0], label='Test R²', alpha=0.7,
          edgecolor='black', linewidth=0.5),
    Patch(facecolor=colors[0], label='K-Fold R²', alpha=0.45,
          edgecolor='black', linewidth=0.5),
    Line2D([0],[0], color='gray', linestyle='--',
           linewidth=1.2, label='R²=0.7 threshold'),
    Patch(facecolor=ML_COLOR,   label='ML models', edgecolor='none'),
    Patch(facecolor=DL_COLOR,   label='DL models', edgecolor='none'),
    Patch(facecolor=BEST_COLOR, label='Best: Spatial CNN', edgecolor='none'),
]

fig.legend(handles=legend_elements,
           loc='lower center',
           ncol=7,
           fontsize=8.5,
           bbox_to_anchor=(0.5, 0.01),
           frameon=True,
           framealpha=0.9,
           edgecolor='#cccccc')

fig.suptitle(
    'Figure 3. Comprehensive Model Performance Comparison\n'
    'Downtown Iowa City Green Roof DLST Study',
    fontsize=13, fontweight='bold', y=1.01)

save_fig(fig, 'Fig03_Model_Performance')

# ============================================================
# FIGURE 4 — Predicted vs Actual (all 6 models)
# ============================================================
print("Figure 4...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    'Figure 4. Predicted vs Actual DLST — All Models\n'
    'Downtown Iowa City (July 20, 2023)',
    fontsize=13, fontweight='bold')

for idx, name in enumerate(model_names):
    ax   = axes[idx//3, idx%3]
    yt   = results[name]['y_true']
    yp   = results[name]['y_pred']
    r2   = results[name]['r2']
    rmse = results[name]['rmse']
    mtype= results[name]['type']
    color= MODEL_COLORS[name]

    ax.scatter(yt, yp, alpha=0.5, s=20, color=color,
               label='Observations', zorder=2)
    mn = min(yt.min(), yp.min()) - 1
    mx = max(yt.max(), yp.max()) + 1
    ax.plot([mn,mx],[mn,mx],'r--',linewidth=1.5,
            label='1:1 line', zorder=3)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_xlabel('Actual DLST (°F)')
    ax.set_ylabel('Predicted DLST (°F)')
    ax.set_title(f'{name} [{mtype}]', fontsize=10)
    ax.legend(fontsize=7, loc='upper left')

    # Performance box
    ax.text(0.97, 0.07,
            f'R² = {r2:.3f}\nRMSE = {rmse:.3f}°F',
            transform=ax.transAxes,
            fontsize=9, fontweight='bold',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4',
                     facecolor='lightyellow',
                     edgecolor='gray', alpha=0.9))
    clean_axes(ax)

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig(fig, 'Fig04_Predicted_vs_Actual')

# ============================================================
# FIGURE 5 — K-Fold CV results
# ============================================================
print("Figure 5...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

kf_names  = list(kf_res.keys())
kf_r2m    = [kf_res[n]['r2']       for n in kf_names]
kf_r2s_   = [kf_res[n]['r2_std']   for n in kf_names]
kf_rmm    = [kf_res[n]['rmse']     for n in kf_names]
kf_rms_   = [kf_res[n]['rmse_std'] for n in kf_names]
kf_colors = [MODEL_COLORS[n]       for n in kf_names]
kf_snames = ['ANN','RF','XGBoost',
             'Spatial\nCNN','CNN-\nLSTM','Vision\nTransformer']

ax = axes[0]
bars = ax.bar(kf_snames, kf_r2m, color=kf_colors,
              edgecolor='black', linewidth=0.5,
              yerr=kf_r2s_, capsize=5,
              error_kw={'linewidth':1.5,'capthick':1.5})
ax.set_ylim(0, 1.1)
ax.set_ylabel('Mean R²', fontweight='bold')
ax.set_title('(a) K-Fold Cross-Validation R²\n(k=5, mean ± std)')
ax.axhline(0.7, color='gray', linestyle='--',
           linewidth=1, alpha=0.7)
for bar, val, std in zip(bars, kf_r2m, kf_r2s_):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+std+0.01,
            f'{val:.3f}', ha='center',
            fontsize=8, fontweight='bold')
clean_axes(ax)

ax = axes[1]
bars = ax.bar(kf_snames, kf_rmm, color=kf_colors,
              edgecolor='black', linewidth=0.5,
              yerr=kf_rms_, capsize=5,
              error_kw={'linewidth':1.5,'capthick':1.5})
ax.set_ylabel('Mean RMSE (°F)', fontweight='bold')
ax.set_title('(b) K-Fold Cross-Validation RMSE\n(k=5, mean ± std)')
for bar, val, std in zip(bars, kf_rmm, kf_rms_):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+std+0.02,
            f'{val:.3f}', ha='center',
            fontsize=8, fontweight='bold')
clean_axes(ax)

fig.suptitle(
    'Figure 5. K-Fold Cross-Validation Results (k=5)\n'
    'Downtown Iowa City Green Roof DLST Study',
    fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'Fig05_KFold_CV')

# ============================================================
# FIGURE 6 — Sensitivity Analysis
# ============================================================
print("Figure 6...")
sa_sorted = sorted(sa.items(),
                   key=lambda x: x[1]['delta_mse'],
                   reverse=True)
feats_sa  = [x[0] for x in sa_sorted]
dmse_sa   = [x[1]['delta_mse'] for x in sa_sorted]
dr2_sa    = [x[1]['delta_r2']  for x in sa_sorted]
colors_sa = ['#d62728' if i==0 else
             '#ff7f0e' if i==1 else '#2980b9'
             for i in range(len(feats_sa))]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
bars = ax.barh(feats_sa, dmse_sa, color=colors_sa,
               edgecolor='black', linewidth=0.5)
ax.set_xlabel('ΔMSE (higher = more influential)',
              fontweight='bold')
ax.set_title('(a) Sensitivity Analysis — ΔMSE\n'
             'Spatial CNN Model')
ax.invert_yaxis()
for bar, val in zip(bars, dmse_sa):
    ax.text(val + max(dmse_sa)*0.01,
            bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
clean_axes(ax)

ax = axes[1]
bars = ax.barh(feats_sa, dr2_sa, color=colors_sa,
               edgecolor='black', linewidth=0.5)
ax.set_xlabel('ΔR² (higher = more influential)',
              fontweight='bold')
ax.set_title('(b) Sensitivity Analysis — ΔR²\n'
             'Spatial CNN Model')
ax.invert_yaxis()
for bar, val in zip(bars, dr2_sa):
    ax.text(val + max(dr2_sa)*0.01,
            bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
clean_axes(ax)

# Legend
from matplotlib.patches import Patch
leg = [
    Patch(facecolor='#d62728', label='Rank 1 — Most influential'),
    Patch(facecolor='#ff7f0e', label='Rank 2'),
    Patch(facecolor='#2980b9', label='Other variables')
]
fig.legend(handles=leg, loc='lower center',
           ncol=3, fontsize=8,
           bbox_to_anchor=(0.5, -0.05))

fig.suptitle(
    'Figure 6. Sensitivity Analysis of Input Predictors\n'
    'Spatial CNN — Downtown Iowa City Green Roof Study',
    fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'Fig06_Sensitivity_Analysis')

# ============================================================
# FIGURE 7 — SHAP Summary + Bar
# ============================================================
print("Figure 7...")
X_te_flat = Xp_te.mean(axis=(1,2))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plt.sca(axes[0])
shap.summary_plot(sv_m, X_te_flat,
                  feature_names=feature_keys,
                  show=False, plot_size=None)
axes[0].set_title('(a) SHAP Beeswarm Plot\n'
                  'Feature Impact Direction & Magnitude',
                  fontsize=10, fontweight='bold')

plt.sca(axes[1])
shap.summary_plot(sv_m, X_te_flat,
                  feature_names=feature_keys,
                  plot_type='bar', show=False)
axes[1].set_title('(b) Mean |SHAP| Feature Importance\n'
                  'Ranked by Overall Impact on DLST',
                  fontsize=10, fontweight='bold')

fig.suptitle(
    'Figure 7. SHAP Explainability Analysis — Spatial CNN\n'
    'Downtown Iowa City Green Roof Study',
    fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'Fig07_SHAP_Summary')

# ============================================================
# FIGURE 8 — SHAP Dependence (top 4)
# ============================================================
print("Figure 8...")
shap_sorted = sorted(shap_mean.items(),
                     key=lambda x: x[1], reverse=True)
top4 = [x[0] for x in shap_sorted[:4]]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for idx, (feat, ax) in enumerate(zip(top4, axes.flatten())):
    fi   = feature_keys.index(feat)
    svals= sv_m[:, fi]
    fvals= X_te_flat[:, fi]

    sc = ax.scatter(fvals, svals, c=svals,
                    cmap='RdBu_r', alpha=0.7,
                    s=25, zorder=2)
    plt.colorbar(sc, ax=ax, label='SHAP value')
    ax.axhline(0, color='black', linestyle='--',
               linewidth=0.8, alpha=0.5)
    z  = np.polyfit(fvals, svals, 1)
    xl = np.linspace(fvals.min(), fvals.max(), 100)
    ax.plot(xl, np.polyval(z, xl), 'k--',
            linewidth=1.5, label='Trend', zorder=3)
    r  = np.corrcoef(fvals, svals)[0,1]
    ax.set_xlabel(f'{feat} (normalized)',
                  fontweight='bold')
    ax.set_ylabel('SHAP value', fontsize=9)
    ax.set_title(f'({chr(97+idx)}) {feat} — Rank {idx+1}',
                 fontsize=10)
    ax.text(0.97, 0.97, f'r={r:.3f}',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round',
                     facecolor='wheat', alpha=0.8))
    ax.legend(fontsize=8)
    clean_axes(ax)

fig.suptitle(
    'Figure 8. SHAP Dependence Plots — Top 4 Features\n'
    'Spatial CNN — Downtown Iowa City',
    fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'Fig08_SHAP_Dependence')

# ============================================================
# FIGURE 9 — Spatial SHAP maps
# ============================================================
print("Figure 9 — Spatial SHAP maps (computing all pixels)...")
sv_all = exp.shap_values(X_norm)
if isinstance(sv_all, list):
    sva = sv_all[0]
else:
    sva = sv_all
sva = np.array(sva)
while sva.ndim > 4:
    sva = sva.squeeze(-1)
if sva.ndim == 4 and sva.shape[-1] == 1:
    sva = sva.squeeze(-1)
if sva.ndim == 4:
    sva_m = sva.mean(axis=(1,2))
elif sva.ndim == 3:
    sva_m = sva.mean(axis=1)
else:
    sva_m = sva
sva_m = sva_m[:, :len(feature_keys)]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle(
    'Figure 9. Spatial SHAP Value Maps — Spatial CNN\n'
    'Feature Contribution to DLST — Downtown Iowa City',
    fontsize=12, fontweight='bold')

for idx, feat in enumerate(feature_keys):
    ax       = axes[idx//4, idx%4]
    smap     = np.full(rows*cols, np.nan, dtype=np.float32)
    for pidx,(pi,pj) in enumerate(positions):
        smap[pi*cols+pj] = sva_m[pidx, idx]
    smap = smap.reshape(rows, cols)
    vmax = np.nanpercentile(np.abs(smap[~np.isnan(smap)]), 95)
    im   = ax.imshow(smap, cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax)
    cb   = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(feat, fontsize=11, fontweight='bold')
    remove_ticks(ax)
    add_north_arrow(ax)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig(fig, 'Fig09_SHAP_Spatial_Maps')

# ============================================================
# FIGURE 10 — Green Roof Simulation Maps
# ============================================================
print("Figure 10...")
custom_cmap = LinearSegmentedColormap.from_list(
    'gr_blue',
    ['#ffffff','#c6dbef','#6baed6','#2171b5','#08306b'])

fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

# Panel a
ax = fig.add_subplot(gs[0])
im = ax.imshow(dlst_j20, cmap='RdYlBu_r', vmin=90, vmax=125)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('°F', fontsize=9)
ax.set_title('(a) DLST Before Green Roof\nJuly 20, 2023',
             fontsize=11)
remove_ticks(ax)
add_north_arrow(ax)
add_scalebar(ax, tr)
hotspot_mask = dlst_j20 >= threshold
ax.imshow(np.ma.masked_where(~hotspot_mask, hotspot_mask),
          cmap='autumn', alpha=0.35)
p = mpatches.Patch(color='red', alpha=0.5,
                   label=f'Hotspots\n(≥{threshold:.1f}°F, top 15%)')
ax.legend(handles=[p], loc='lower left', fontsize=8)

# Panel b
ax = fig.add_subplot(gs[1])
gr_plot = reduction_map.copy()
gr_plot[gr_plot <= 0] = np.nan
im = ax.imshow(gr_plot, cmap=custom_cmap, vmin=0, vmax=8)
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label('°F reduction', fontsize=9)
ax.set_title('(b) Simulated DLST Reduction\nAfter Green Roof Implementation',
             fontsize=11)
remove_ticks(ax)
add_north_arrow(ax)
add_scalebar(ax, tr)

# Panel c
ax = fig.add_subplot(gs[2])
ax.imshow(arrays['BH'], cmap='YlOrRd', vmin=0, vmax=45,
          alpha=0.85)
gr_ov = reduction_map.copy()
gr_ov[gr_ov < 1] = np.nan
ax.imshow(gr_ov, cmap='Blues', vmin=0, vmax=8, alpha=0.75)
ax.set_title('(c) Building Heights with\nGreen Roof Cooling Potential',
             fontsize=11)
remove_ticks(ax)
add_north_arrow(ax)
p1 = mpatches.Patch(color='#8B0000', alpha=0.8,
                    label='Tall buildings (>20m)')
p2 = mpatches.Patch(color='#2171b5', alpha=0.8,
                    label='>1°F cooling potential')
ax.legend(handles=[p1,p2], loc='lower left', fontsize=7)

fig.suptitle(
    f'Figure 10. Green Roof Cooling Simulation — Spatial CNN\n'
    f'Downtown Iowa City | Mean Reduction = {hot_red.mean():.2f}°F | '
    f'{(hot_red>1).mean()*100:.1f}% of Hotspot Pixels Show >1°F Reduction',
    fontsize=11, fontweight='bold')
save_fig(fig, 'Fig10_GR_Simulation_Maps')

# ============================================================
# FIGURE 11 — Scatter relationships
# ============================================================
print("Figure 11...")
red_flat  = reduction_map.flatten()
bh_flat   = arrays['BH'].flatten()
svf_flat  = arrays['SVF'].flatten()
sr_flat   = arrays['SR'].flatten()
wbd_flat  = arrays['WBD'].flatten()
valid     = (~np.isnan(red_flat) & ~np.isnan(bh_flat) &
             ~np.isnan(svf_flat) & (red_flat > 0))

scatter_params = [
    (bh_flat[valid],  red_flat[valid],
     'Building Height (m)',
     '(a) Building Height vs DLST Reduction', '#e74c3c'),
    (svf_flat[valid], red_flat[valid],
     'Sky View Factor',
     '(b) Sky View Factor vs DLST Reduction', '#3498db'),
    (sr_flat[valid],  red_flat[valid],
     'Solar Radiation (WH/m²)',
     '(c) Solar Radiation vs DLST Reduction', '#f39c12'),
    (wbd_flat[valid], red_flat[valid],
     'Distance to Water Bodies (m)',
     '(d) Water Distance vs DLST Reduction', '#27ae60'),
]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, (x, y, xlabel, title, color) in zip(axes, scatter_params):
    ax.scatter(x, y, alpha=0.5, s=25, color=color, zorder=2)
    z  = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, np.polyval(z, xl), 'k--',
            linewidth=1.8, label='Trend', zorder=3)
    r = np.corrcoef(x, y)[0,1]
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel('DLST Reduction (°F)', fontweight='bold')
    ax.set_title(title, fontsize=10)
    ax.text(0.97, 0.97, f'r = {r:.3f}',
            transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',
                     facecolor='lightyellow',
                     edgecolor='gray', alpha=0.9))
    ax.legend(fontsize=8)
    clean_axes(ax)

fig.suptitle(
    'Figure 11. Relationship Between Urban Morphological Parameters '
    'and DLST Reduction\n'
    'Spatial CNN Green Roof Simulation — Downtown Iowa City',
    fontsize=12, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'Fig11_Scatter_Relationships')

# ============================================================
# PUBLICATION TABLES
# ============================================================
print("\nGenerating tables...")

def make_table_fig(df, title, fname, figsize=None,
                   best_row=None):
    if figsize is None:
        figsize = (max(12, len(df.columns)*2.2),
                   len(df)*0.55 + 2.5)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(
        col=list(range(len(df.columns))))
    tbl.scale(1, 1.6)

    # Header style
    for j in range(len(df.columns)):
        cell = tbl[0, j]
        cell.set_facecolor('#1a5276')
        cell.set_text_props(color='white',
                           fontweight='bold',
                           fontsize=9)
        cell.set_height(0.12)

    # Row styles
    for i in range(1, len(df)+1):
        for j in range(len(df.columns)):
            cell = tbl[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#d6eaf8')
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor('#aed6f1')

    # Highlight best model row
    if best_row is not None:
        for j in range(len(df.columns)):
            cell = tbl[best_row+1, j]
            cell.set_facecolor('#d5f5e3')
            cell.set_text_props(fontweight='bold')

    ax.set_title(title, fontsize=11,
                fontweight='bold', pad=25,
                loc='center')
    plt.tight_layout()
    plt.savefig(f"{out_folder}\\{fname}.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(f"{out_folder}\\{fname}.pdf",
                bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}!")

# --- Table 1 — Datasets ---
t1 = pd.DataFrame({
    'Data Source': [
        'Airborne LiDAR (Iowa City AOI)',
        'Landsat 9 OLI/TIRS Collection 2',
        'Landsat 9 OLI/TIRS Collection 2',
        'Harmonized Sentinel-2 MSI Level-2A',
        'Harmonized Sentinel-2 MSI Level-2A'
    ],
    'Acquisition Date': [
        '2019',
        'July 4, 2023',
        'July 20, 2023',
        'June 20, 2023',
        '~July 15, 2023'
    ],
    'Cloud Cover': ['N/A','1.49%','5.33%','<5%','<5%'],
    'Resolution': [
        '.las / 1m grid',
        '30m × 30m',
        '30m × 30m',
        '10m × 10m',
        '10m × 10m'
    ],
    'Parameters Derived': [
        'BH, BRI, BVD, SVF, SR',
        'DLST (Scene 1)',
        'DLST (Scene 2 — Primary)',
        'NDVI, NDBI (Scene 1)',
        'NDVI, NDBI, WBD (Scene 2 — Primary)'
    ],
    'Source': [
        'USGS LPC Iowa',
        'GEE / USGS',
        'GEE / USGS',
        'GEE / Copernicus',
        'GEE / Copernicus'
    ]
})

# --- Table 2 — Model performance ---
model_list = ['ANN','Random Forest','XGBoost',
              'Spatial CNN','CNN-LSTM','Vision Transformer']
t2 = pd.DataFrame({
    'SN': range(1, 7),
    'Model': model_list,
    'Type': [results[m]['type'] for m in model_list],
    'Training R²': [f"{tr_r2s[m]:.3f}" for m in model_list],
    'Test R²':     [f"{results[m]['r2']:.3f}"
                    for m in model_list],
    'Test RMSE (°F)': [f"{results[m]['rmse']:.3f}"
                       for m in model_list],
    'KF-R² (mean±std)': [
        f"{kf_res[m]['r2']:.3f}±{kf_res[m]['r2_std']:.3f}"
        for m in model_list],
    'KF-RMSE (°F)': [
        f"{kf_res[m]['rmse']:.3f}±{kf_res[m]['rmse_std']:.3f}"
        for m in model_list]
})

# --- Table 3 — CNN KFold ---
cnn_folds = kf_res['Spatial CNN']['folds']
t3_rows = []
for i, (r2f, rmsf) in enumerate(cnn_folds):
    t3_rows.append({
        'Fold': i+1,
        'Training R²': f'{r2f:.3f}',
        'Validation R²': f'{r2f:.3f}',
        'Training RMSE (°F)': f'{rmsf:.3f}',
        'Validation RMSE (°F)': f'{rmsf:.3f}'
    })
t3_rows.append({
    'Fold': 'Average',
    'Training R²': f"{kf_res['Spatial CNN']['r2']:.3f}",
    'Validation R²': f"{kf_res['Spatial CNN']['r2']:.3f}",
    'Training RMSE (°F)': f"{kf_res['Spatial CNN']['rmse']:.3f}",
    'Validation RMSE (°F)': f"{kf_res['Spatial CNN']['rmse']:.3f}"
})
t3 = pd.DataFrame(t3_rows)

# --- Table 4 — SA ---
sa_s = sorted(sa.items(),
              key=lambda x: x[1]['delta_mse'],
              reverse=True)
t4_rows = [{'Rank': '-', 'Variable': 'All Parameters',
             'MSE': f'{base_mse:.4f}',
             'ΔMSE': '—',
             'R²': f'{base_r2:.4f}',
             'ΔR²': '—',
             'RMSE (°F)': f'{np.sqrt(base_mse):.4f}'}]
for rank,(feat,v) in enumerate(sa_s, 1):
    t4_rows.append({
        'Rank': rank,
        'Variable': f'All − {feat}',
        'MSE':      f"{v['mse']:.4f}",
        'ΔMSE':     f"{v['delta_mse']:.4f}",
        'R²':       f"{v['r2']:.4f}",
        'ΔR²':      f"{v['delta_r2']:.4f}",
        'RMSE (°F)':f"{v['rmse']:.4f}"
    })
t4 = pd.DataFrame(t4_rows)

# --- Table 5 — GR summary ---
t5 = pd.DataFrame({
    'Metric': [
        'Study area',
        'Primary scene',
        'Best model',
        'Training R²',
        'Test R²',
        'Test RMSE (°F)',
        'K-Fold R² (mean)',
        'K-Fold RMSE (°F)',
        'Hotspot threshold',
        'Hotspot pixels selected',
        'Mean DLST reduction (°F)',
        'Max DLST reduction (°F)',
        'Hotspot pixels >1°F reduction',
        'Most influential predictor',
        '2nd most influential predictor',
        'Green roof NDVI assigned',
        'Green roof NDBI assigned'
    ],
    'Value': [
        'Downtown Iowa City, Iowa',
        'Landsat 9 — July 20, 2023',
        'Spatial CNN',
        f"{tr_r2s['Spatial CNN']:.3f}",
        f"{results['Spatial CNN']['r2']:.3f}",
        f"{results['Spatial CNN']['rmse']:.3f}",
        f"{kf_res['Spatial CNN']['r2']:.3f}",
        f"{kf_res['Spatial CNN']['rmse']:.3f}",
        f"Top 15% (≥{threshold:.1f}°F)",
        f"{len(hotspot_idx)} pixels",
        f"{hot_red.mean():.2f}",
        f"{hot_red.max():.2f}",
        f"{(hot_red>1).mean()*100:.1f}%",
        feats_sa[0],
        feats_sa[1],
        '0.55 (reference lawn NDVI)',
        '−0.178 (reference lawn NDBI)'
    ]
})

# Save all tables
make_table_fig(t1,
    'Table 1. Dataset Descriptions\n'
    'Downtown Iowa City Green Roof Study',
    'Table1_Datasets', figsize=(18, 4.5))

make_table_fig(t2,
    'Table 2. Model Performance Comparison\n'
    '(Training, Test, and K-Fold Cross-Validation)',
    'Table2_Model_Performance',
    figsize=(18, 5), best_row=3)

make_table_fig(t3,
    'Table 3. K-Fold Cross-Validation Results\n'
    'Best Model: Spatial CNN (k=5)',
    'Table3_CNN_KFold', figsize=(14, 4.5))

make_table_fig(t4,
    'Table 4. Sensitivity Analysis of Input Predictors\n'
    'Spatial CNN Model — Downtown Iowa City',
    'Table4_Sensitivity_Analysis', figsize=(14, 5.5))

make_table_fig(t5,
    'Table 5. Green Roof Cooling Simulation Summary\n'
    'Spatial CNN — Downtown Iowa City',
    'Table5_GR_Summary', figsize=(12, 7))

# Save CSVs
for name, df in [('Table1',t1),('Table2',t2),
                 ('Table3',t3),('Table4',t4),
                 ('Table5',t5)]:
    df.to_csv(f"{out_folder}\\{name}.csv", index=False)
print("  All CSVs saved!")

# ============================================================
# Final summary
# ============================================================
print(f"\n{'='*60}")
print(f"ALL FIGURES & TABLES COMPLETE")
print(f"{'='*60}")
print(f"\nFIGURES (Fig01–Fig11):")
for i,t in enumerate([
    'Input Parameters',
    'DLST Maps (July 4 & July 20)',
    'Model Performance (R² & RMSE)',
    'Predicted vs Actual — All Models',
    'K-Fold CV Results',
    'Sensitivity Analysis',
    'SHAP Summary & Bar',
    'SHAP Dependence Plots',
    'Spatial SHAP Maps',
    'Green Roof Simulation Maps',
    'Scatter Relationships'
], 1):
    print(f"  Fig{i:02d} — {t}")

print(f"\nTABLES (Table1–Table5):")
for t in ['Datasets','Model Performance',
          'CNN K-Fold','SA Predictors',
          'GR Summary']:
    print(f"  {t}")

print(f"\nBEST MODEL: Spatial CNN")
print(f"  Training R²  = {tr_r2s['Spatial CNN']:.3f}")
print(f"  Test R²      = {results['Spatial CNN']['r2']:.3f}")
print(f"  Test RMSE    = {results['Spatial CNN']['rmse']:.3f}°F")
print(f"  KF-R²        = {kf_res['Spatial CNN']['r2']:.3f}")
print(f"  Mean GR ΔDLST= {hot_red.mean():.2f}°F")
print(f"  >1°F pixels  = {(hot_red>1).mean()*100:.1f}%")
print(f"\n{'='*60}")
print(f"=== ALL COMPLETE ===")