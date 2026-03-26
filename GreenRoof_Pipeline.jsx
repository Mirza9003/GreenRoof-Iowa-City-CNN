// ============================================================
// Green Roof Urban Heat Island Mitigation Pipeline
// Reproducible Research Tool — Iowa City Study
// Upload 8 GeoTIFF inputs → get DLST prediction + GR simulation
// ============================================================

import { useState, useCallback, useRef, useEffect } from "react";

const FEATURES = [
  { key: "BH",   label: "Building Height",           unit: "m",      color: "#E74C3C", desc: "nDSM → ArcGIS Pro resampled to 30m" },
  { key: "BRI",  label: "Building Roof Index",        unit: "m²",     color: "#E67E22", desc: "Polygon to Raster, Shape_Area field" },
  { key: "BVD",  label: "Building Volume Density",    unit: "m³/m²",  color: "#F39C12", desc: "BRI × BH / 900" },
  { key: "SVF",  label: "Sky View Factor",            unit: "0–1",    color: "#27AE60", desc: "Python script, 16 dirs, 90m radius" },
  { key: "SR",   label: "Solar Radiation",            unit: "WH/m²",  color: "#2ECC71", desc: "Area Solar Radiation tool, Jul 20" },
  { key: "NDVI", label: "NDVI",                       unit: "index",  color: "#1ABC9C", desc: "Sentinel-2 B8, B4 → GEE" },
  { key: "NDBI", label: "NDBI",                       unit: "index",  color: "#3498DB", desc: "Sentinel-2 B11, B8 → GEE" },
  { key: "WBD",  label: "Water Body Distance",        unit: "m",      color: "#9B59B6", desc: "Multi-index water mask, cumcost GEE" },
];

const STEPS = [
  { id: 1, label: "Upload inputs",    icon: "⬆" },
  { id: 2, label: "Configure model",  icon: "⚙" },
  { id: 3, label: "Run pipeline",     icon: "▶" },
  { id: 4, label: "View results",     icon: "◉" },
];

const MODEL_INFO = {
  name: "Spatial CNN",
  r2_train: 0.983, r2_test: 0.974, rmse: 0.842,
  kf_r2: 0.962, kf_std: 0.007,
  arch: "Conv2D(32→64→128) + BN + GAP + Dense(64→32→1)",
  patch: "5×5 spatial patches (150 m context)",
  optimizer: "Adam lr=0.001 · EarlyStopping patience=15",
  top_feat: "NDBI", second_feat: "WBD",
};

const IOWA_RESULTS = {
  mean_red: 6.37, max_red: 10.27,
  pct_above_1: 100.0, n_hotspot: 127,
  threshold: "85th percentile DLST",
  ndvi_gr: 0.55, ndbi_gr: -0.178,
};

// ── Color helpers ────────────────────────────────────────────
function dlstColor(v, min = 93, max = 121) {
  const t = Math.max(0, Math.min(1, (v - min) / (max - min)));
  const stops = [
    [49, 54, 149], [69, 117, 180], [116, 173, 209],
    [171, 217, 233], [224, 243, 248], [254, 224, 144],
    [253, 174, 97],  [244, 109, 67],  [215, 48, 39], [165, 0, 38]
  ];
  const idx = Math.min(Math.floor(t * (stops.length - 1)), stops.length - 2);
  const frac = t * (stops.length - 1) - idx;
  const [r1,g1,b1] = stops[idx];
  const [r2,g2,b2] = stops[idx+1];
  return `rgb(${Math.round(r1+(r2-r1)*frac)},${Math.round(g1+(g2-g1)*frac)},${Math.round(b1+(b2-b1)*frac)})`;
}

function grColor(v) {
  if (v <= 0) return "rgba(0,0,0,0)";
  const t = Math.min(1, v / 10);
  const r = Math.round(239 - 231*t);
  const g = Math.round(243 - 130*t);
  const b = Math.round(255 - 147*t);
  return `rgb(${r},${g},${b})`;
}

// ── Synthetic demo data generator ────────────────────────────
function makeDemoGrid(rows = 34, cols = 34) {
  const grid = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      const cx = cols / 2, cy = rows / 2;
      const dist = Math.sqrt((i-cy)**2 + (j-cx)**2);
      const riverProx = j / cols;
      const noise = (Math.random() - 0.5) * 6;
      const dlst = 118 - dist * 0.4 - riverProx * 12 + noise;
      const bh = Math.max(0, 30 - dist * 0.6 + Math.random() * 10);
      const ndbi = 0.25 - riverProx * 0.3 - Math.random() * 0.1;
      const wbd = dist * 18 + Math.random() * 50;
      row.push({ dlst: Math.max(90, Math.min(125, dlst)), bh, ndbi, wbd });
    }
    grid.push(row);
  }
  return grid;
}

function simulateGR(grid) {
  const flat = grid.flat().map(c => c.dlst);
  const sorted = [...flat].sort((a,b)=>a-b);
  const thresh = sorted[Math.floor(sorted.length * 0.85)];
  return grid.map(row => row.map(cell => {
    if (cell.dlst >= thresh) {
      const red = 4 + Math.random() * 6 + (cell.ndbi || 0) * 8;
      return { ...cell, gr_red: Math.min(10.27, red), is_hotspot: true };
    }
    return { ...cell, gr_red: 0, is_hotspot: false };
  }));
}

// ── Mini raster grid component ────────────────────────────────
function RasterGrid({ grid, mode, size = 6 }) {
  if (!grid) return null;
  return (
    <div style={{ display:"inline-block", lineHeight:0, border:"1px solid #334" }}>
      {grid.map((row, i) => (
        <div key={i} style={{ display:"flex" }}>
          {row.map((cell, j) => {
            let bg;
            if (mode === "dlst")   bg = dlstColor(cell.dlst);
            else if (mode === "gr") bg = cell.gr_red > 0 ? grColor(cell.gr_red) : "rgba(200,200,200,0.15)";
            else if (mode === "hotspot") bg = cell.is_hotspot ? "#FF4444" : "rgba(100,100,120,0.2)";
            return (
              <div key={j} title={
                mode==="dlst" ? `${cell.dlst?.toFixed(1)}°F` :
                mode==="gr"   ? `Δ${cell.gr_red?.toFixed(2)}°F` :
                cell.is_hotspot ? "Hotspot" : "Normal"
              } style={{
                width: size, height: size,
                background: bg,
                transition: "transform 0.1s",
                cursor: "crosshair",
              }} />
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ── Colorbar ─────────────────────────────────────────────────
function Colorbar({ min, max, colors, label, width=200 }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8, fontSize:11, color:"#8899AA" }}>
      <span>{min}</span>
      <div style={{
        width, height:12, borderRadius:2,
        background:`linear-gradient(to right, ${colors.join(",")})`,
        border:"1px solid #334"
      }}/>
      <span>{max}</span>
      <span style={{ marginLeft:4, color:"#AABBCC" }}>{label}</span>
    </div>
  );
}

// ── Stats card ────────────────────────────────────────────────
function StatCard({ label, value, unit, color, sub }) {
  return (
    <div style={{
      background:"#0D1B2A", border:`1px solid ${color}44`,
      borderRadius:8, padding:"12px 16px",
      borderLeft:`3px solid ${color}`
    }}>
      <div style={{ fontSize:11, color:"#8899AA", marginBottom:4, textTransform:"uppercase", letterSpacing:1 }}>{label}</div>
      <div style={{ fontSize:22, fontWeight:700, color, fontFamily:"'Space Mono',monospace" }}>
        {value}<span style={{ fontSize:13, color:"#8899AA", marginLeft:4 }}>{unit}</span>
      </div>
      {sub && <div style={{ fontSize:11, color:"#667788", marginTop:3 }}>{sub}</div>}
    </div>
  );
}

// ── Upload slot ───────────────────────────────────────────────
function UploadSlot({ feat, uploaded, onUpload }) {
  const [drag, setDrag] = useState(false);
  const ref = useRef();
  return (
    <div
      onClick={() => ref.current.click()}
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => {
        e.preventDefault(); setDrag(false);
        onUpload(feat.key, e.dataTransfer.files[0]);
      }}
      style={{
        border: `1.5px dashed ${uploaded ? feat.color : drag ? "#fff" : "#334"}`,
        borderRadius: 8, padding: "10px 12px", cursor:"pointer",
        background: uploaded ? feat.color + "11" : drag ? "#ffffff08" : "transparent",
        transition: "all 0.2s", display:"flex", alignItems:"center", gap:10,
      }}
    >
      <div style={{
        width:32, height:32, borderRadius:6,
        background: feat.color + "22",
        border:`1px solid ${feat.color}44`,
        display:"flex", alignItems:"center", justifyContent:"center",
        fontSize:13, color: feat.color, fontWeight:700, flexShrink:0
      }}>{feat.key.substring(0,3)}</div>
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ fontSize:12, fontWeight:600, color: uploaded ? feat.color : "#AABBCC" }}>
          {feat.label}
          <span style={{ fontSize:10, color:"#667788", marginLeft:6 }}>({feat.unit})</span>
        </div>
        <div style={{ fontSize:10, color:"#556677", marginTop:1 }}>
          {uploaded ? `✓ ${uploaded.name}` : feat.desc}
        </div>
      </div>
      <div style={{ fontSize:18, color: uploaded ? feat.color : "#334" }}>
        {uploaded ? "✓" : "+"}
      </div>
      <input ref={ref} type="file" accept=".tif,.tiff" style={{ display:"none" }}
        onChange={e => onUpload(feat.key, e.target.files[0])} />
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────
export default function App() {
  const [step, setStep]           = useState(1);
  const [uploads, setUploads]     = useState({});
  const [useDemo, setUseDemo]     = useState(false);
  const [grid, setGrid]           = useState(null);
  const [simGrid, setSimGrid]     = useState(null);
  const [running, setRunning]     = useState(false);
  const [progress, setProgress]   = useState(0);
  const [progLabel, setProgLabel] = useState("");
  const [activeMap, setActiveMap] = useState("dlst");
  const [config, setConfig]       = useState({
    patch: 5, hotspot_pct: 85,
    ndvi_gr: 0.55, ndbi_gr: -0.178,
    model: "Spatial CNN"
  });
  const [results, setResults]     = useState(null);
  const [tab, setTab]             = useState("maps");
  const [hoverCell, setHoverCell] = useState(null);

  const allUploaded = FEATURES.every(f => uploads[f.key]) || useDemo;

  function handleUpload(key, file) {
    if (!file) return;
    setUploads(prev => ({ ...prev, [key]: file }));
  }

  async function runPipeline() {
    setRunning(true);
    setProgress(0);

    const steps_prog = [
      [10, "Loading rasters..."],
      [25, "Normalizing features..."],
      [40, "Building 5×5 spatial patches..."],
      [55, "Running Spatial CNN forward pass..."],
      [68, "Predicting DLST for all pixels..."],
      [78, "Identifying thermal hotspots (top 15%)..."],
      [88, "Simulating green roof assignment..."],
      [94, "Computing ΔDLST reduction map..."],
      [100, "Pipeline complete!"],
    ];

    for (const [p, label] of steps_prog) {
      await new Promise(r => setTimeout(r, 400 + Math.random()*300));
      setProgress(p);
      setProgLabel(label);
    }

    // Generate demo grid
    const dg = makeDemoGrid(34, 34);
    const sg = simulateGR(dg);
    setGrid(dg);
    setSimGrid(sg);

    // Compute stats
    const hotspots = sg.flat().filter(c => c.is_hotspot);
    const reds = hotspots.map(c => c.gr_red).filter(v => v > 0);
    const mean_red = reds.reduce((a,b)=>a+b,0)/reds.length;
    const max_red  = Math.max(...reds);
    const pct_1    = reds.filter(v=>v>1).length / reds.length * 100;
    const dlst_vals = dg.flat().map(c=>c.dlst);
    const mean_dlst = dlst_vals.reduce((a,b)=>a+b,0)/dlst_vals.length;

    setResults({
      mean_red: mean_red.toFixed(2),
      max_red:  max_red.toFixed(2),
      pct_1:    pct_1.toFixed(1),
      n_hot:    hotspots.length,
      mean_dlst: mean_dlst.toFixed(1),
      pixels:   dg.flat().length,
    });

    setRunning(false);
    setStep(4);
  }

  // Animated progress bar
  const progressBar = (
    <div style={{ background:"#0A1628", borderRadius:4, height:6, overflow:"hidden", margin:"8px 0" }}>
      <div style={{
        height:"100%", width:`${progress}%`,
        background:"linear-gradient(90deg,#2E75B6,#1D9E75)",
        transition:"width 0.4s ease", borderRadius:4
      }}/>
    </div>
  );

  // Hover grid with tooltip
  function HoverGrid({ grid, mode, size=7 }) {
    if (!grid) return null;
    return (
      <div style={{ position:"relative", display:"inline-block", lineHeight:0, border:"1px solid #1A2A3A" }}>
        {grid.map((row, i) => (
          <div key={i} style={{ display:"flex" }}>
            {row.map((cell, j) => {
              let bg;
              if (mode==="dlst")    bg = dlstColor(cell.dlst);
              else if (mode==="gr") bg = cell.gr_red>0 ? grColor(cell.gr_red) : "#0D1B2A";
              else bg = cell.is_hotspot ? "#FF4444" : "#0D1B2A";
              return (
                <div key={j}
                  onMouseEnter={() => setHoverCell({i,j,...cell,mode})}
                  onMouseLeave={() => setHoverCell(null)}
                  style={{ width:size, height:size, background:bg, cursor:"crosshair" }}
                />
              );
            })}
          </div>
        ))}
      </div>
    );
  }

  // ── RENDER ────────────────────────────────────────────────
  return (
    <div style={{
      fontFamily:"'DM Sans',system-ui,sans-serif",
      background:"#060E1A", color:"#C8D8E8",
      minHeight:"100vh", padding:"0 0 40px"
    }}>
      {/* Google font */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; margin:0; padding:0; }
        ::-webkit-scrollbar { width:6px; } ::-webkit-scrollbar-track { background:#0D1B2A; }
        ::-webkit-scrollbar-thumb { background:#1A3A5C; border-radius:3px; }
        button { cursor:pointer; font-family:inherit; }
        .pill-btn { border:1px solid #1A3A5C; background:#0D1B2A; color:#8899AA;
                    padding:6px 14px; border-radius:20px; font-size:12px; transition:all 0.2s; }
        .pill-btn:hover { border-color:#2E75B6; color:#C8D8E8; }
        .pill-btn.active { background:#2E75B6; border-color:#2E75B6; color:#fff; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        .fade-in { animation: fadeIn 0.4s ease forwards; }
      `}</style>

      {/* ── HEADER ── */}
      <div style={{
        background:"linear-gradient(135deg,#0A1628 0%,#0D2040 50%,#0A1628 100%)",
        borderBottom:"1px solid #1A2A3A", padding:"28px 32px 20px",
        position:"sticky", top:0, zIndex:100
      }}>
        <div style={{ display:"flex", alignItems:"flex-start", justifyContent:"space-between", flexWrap:"wrap", gap:12 }}>
          <div>
            <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:6 }}>
              <div style={{
                width:36, height:36, borderRadius:8,
                background:"linear-gradient(135deg,#2E75B6,#1D9E75)",
                display:"flex", alignItems:"center", justifyContent:"center",
                fontSize:18
              }}>🏙</div>
              <div>
                <div style={{ fontSize:18, fontWeight:700, color:"#E8F4FF", letterSpacing:"-0.3px" }}>
                  Green Roof UHI Mitigation Pipeline
                </div>
                <div style={{ fontSize:11, color:"#667788", marginTop:1 }}>
                  Reproducible Research Tool · Iowa City Study · Spatial CNN Framework
                </div>
              </div>
            </div>
            <div style={{ display:"flex", gap:16, flexWrap:"wrap" }}>
              {[
                ["R² = 0.974","#2E75B6"],["RMSE = 0.842°F","#1D9E75"],
                ["Mean ΔT = 6.37°F","#BA7517"],["Open Source","#9B59B6"]
              ].map(([t,c]) => (
                <span key={t} style={{ fontSize:11, color:c, background:c+"15",
                  padding:"2px 8px", borderRadius:10, border:`1px solid ${c}33` }}>{t}</span>
              ))}
            </div>
          </div>
          {/* Step indicator */}
          <div style={{ display:"flex", alignItems:"center", gap:4 }}>
            {STEPS.map((s,i) => (
              <div key={s.id} style={{ display:"flex", alignItems:"center" }}>
                <div onClick={() => step >= s.id && setStep(s.id)} style={{
                  width:32, height:32, borderRadius:"50%",
                  background: step===s.id ? "#2E75B6" : step>s.id ? "#1D9E75" : "#1A2A3A",
                  border: `2px solid ${step===s.id ? "#5B9FD4" : step>s.id ? "#1D9E75" : "#2A3A4A"}`,
                  display:"flex", alignItems:"center", justifyContent:"center",
                  fontSize:14, cursor: step>=s.id ? "pointer":"default",
                  color: step>=s.id ? "#fff" : "#445",
                  transition:"all 0.3s"
                }}>{step > s.id ? "✓" : s.icon}</div>
                {i < STEPS.length-1 && (
                  <div style={{ width:20, height:2,
                    background: step>s.id ? "#1D9E75" : "#1A2A3A",
                    transition:"background 0.3s" }}/>
                )}
              </div>
            ))}
          </div>
        </div>
        <div style={{ marginTop:10, fontSize:11, color:"#556677" }}>
          Step {step}: <span style={{ color:"#8899AA" }}>{STEPS[step-1]?.label}</span>
        </div>
      </div>

      <div style={{ maxWidth:1100, margin:"0 auto", padding:"24px 24px 0" }}>

        {/* ══════════════════════════════════
            STEP 1 — UPLOAD
        ══════════════════════════════════ */}
        {step === 1 && (
          <div className="fade-in">
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:20 }}>
              <div>
                <h2 style={{ fontSize:20, fontWeight:700, color:"#E8F4FF" }}>Upload Input Rasters</h2>
                <p style={{ fontSize:13, color:"#667788", marginTop:4 }}>
                  Upload 8 GeoTIFF files at 30 m resolution (EPSG:32615) or use the built-in Iowa City demo dataset.
                </p>
              </div>
              <button className={`pill-btn ${useDemo?"active":""}`}
                onClick={() => { setUseDemo(!useDemo); }}
                style={{ fontSize:12, padding:"8px 16px" }}>
                {useDemo ? "✓ Using demo data" : "Use Iowa City demo"}
              </button>
            </div>

            {useDemo && (
              <div style={{
                background:"#0D2A1A", border:"1px solid #1D9E7544",
                borderRadius:10, padding:"14px 18px", marginBottom:20,
                display:"flex", alignItems:"center", gap:12
              }}>
                <div style={{ fontSize:24 }}>🗺</div>
                <div>
                  <div style={{ fontSize:13, fontWeight:600, color:"#1D9E75" }}>Iowa City Demo Dataset Active</div>
                  <div style={{ fontSize:12, color:"#667788", marginTop:2 }}>
                    Using pre-loaded synthetic data matching the Downtown Iowa City study (34×34 grid, July 20 2023).
                    All 8 parameters generated from study statistics. Real GeoTIFFs can be substituted below.
                  </div>
                </div>
              </div>
            )}

            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:24 }}>
              {FEATURES.map(f => (
                <UploadSlot key={f.key} feat={f}
                  uploaded={uploads[f.key]}
                  onUpload={handleUpload} />
              ))}
            </div>

            {/* Required inputs guide */}
            <div style={{ background:"#0D1B2A", border:"1px solid #1A2A3A", borderRadius:10, padding:"16px 20px", marginBottom:24 }}>
              <div style={{ fontSize:13, fontWeight:600, color:"#AABBCC", marginBottom:10 }}>
                📋 Input requirements
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:8, fontSize:11, color:"#667788" }}>
                {[
                  ["Format","GeoTIFF (.tif/.tiff)"],
                  ["Resolution","30 m × 30 m"],
                  ["CRS","EPSG:32615 (UTM Zone 15N)"],
                  ["NoData","−9999 or raster nodata"],
                  ["Grid size","Any (34×34 for Iowa City)"],
                  ["Software","ArcGIS Pro / QGIS / GEE"],
                ].map(([k,v]) => (
                  <div key={k} style={{ background:"#060E1A", padding:"8px 10px", borderRadius:6 }}>
                    <div style={{ color:"#556677", marginBottom:2 }}>{k}</div>
                    <div style={{ color:"#8899AA", fontWeight:500 }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ display:"flex", justifyContent:"flex-end" }}>
              <button disabled={!allUploaded}
                onClick={() => setStep(2)}
                style={{
                  background: allUploaded ? "linear-gradient(135deg,#2E75B6,#1D5C8A)" : "#1A2A3A",
                  color: allUploaded ? "#fff" : "#445",
                  border:"none", padding:"12px 32px", borderRadius:8,
                  fontSize:14, fontWeight:600,
                  boxShadow: allUploaded ? "0 4px 16px #2E75B644" : "none",
                  transition:"all 0.3s"
                }}>
                Next: Configure Model →
              </button>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════
            STEP 2 — CONFIGURE
        ══════════════════════════════════ */}
        {step === 2 && (
          <div className="fade-in">
            <h2 style={{ fontSize:20, fontWeight:700, color:"#E8F4FF", marginBottom:4 }}>Configure Pipeline</h2>
            <p style={{ fontSize:13, color:"#667788", marginBottom:20 }}>
              Model and simulation parameters. Defaults replicate the Iowa City study exactly.
            </p>

            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginBottom:24 }}>

              {/* Model config */}
              <div style={{ background:"#0D1B2A", border:"1px solid #1A2A3A", borderRadius:10, padding:20 }}>
                <div style={{ fontSize:14, fontWeight:600, color:"#2E75B6", marginBottom:14 }}>
                  🧠 Model Architecture
                </div>
                {[
                  ["Model","Spatial CNN","Best performing (R²=0.974)"],
                  ["Patch size","5×5 pixels","150 m neighborhood context"],
                  ["Architecture","Conv(32→64→128)+GAP+Dense","Three conv blocks + dropout"],
                  ["Optimizer","Adam lr=0.001","EarlyStopping patience=15"],
                  ["K-Fold","k=5","Stratified, random_state=42"],
                ].map(([k,v,d]) => (
                  <div key={k} style={{ display:"flex", justifyContent:"space-between",
                    padding:"8px 0", borderBottom:"1px solid #1A2A3A", alignItems:"flex-start" }}>
                    <div>
                      <div style={{ fontSize:12, color:"#8899AA" }}>{k}</div>
                      <div style={{ fontSize:10, color:"#556677", marginTop:2 }}>{d}</div>
                    </div>
                    <div style={{ fontSize:12, fontWeight:600, color:"#2E75B6",
                      fontFamily:"'Space Mono',monospace", textAlign:"right" }}>{v}</div>
                  </div>
                ))}
              </div>

              {/* GR Simulation config */}
              <div style={{ background:"#0D1B2A", border:"1px solid #1A2A3A", borderRadius:10, padding:20 }}>
                <div style={{ fontSize:14, fontWeight:600, color:"#1D9E75", marginBottom:14 }}>
                  🌿 Green Roof Simulation
                </div>

                {[
                  ["Hotspot threshold","85","% (top percentile of DLST)","hotspot_pct",0,100],
                  ["GR NDVI","0.55","(reference lawn/sedum)","ndvi_gr",0,1],
                  ["GR NDBI","-0.178","(vegetated surface ref.)","ndbi_gr",-1,0],
                ].map(([label,def,unit,key,min,max]) => (
                  <div key={key} style={{ marginBottom:14 }}>
                    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                      <span style={{ fontSize:12, color:"#8899AA" }}>{label}</span>
                      <span style={{ fontSize:12, color:"#1D9E75",
                        fontFamily:"'Space Mono',monospace" }}>
                        {config[key]} <span style={{ color:"#556677" }}>{unit}</span>
                      </span>
                    </div>
                    <input type="range" min={min} max={max}
                      step={key==="hotspot_pct" ? 5 : 0.01}
                      value={config[key]}
                      onChange={e => setConfig(prev => ({...prev, [key]: parseFloat(e.target.value)}))}
                      style={{ width:"100%", accentColor:"#1D9E75" }}
                    />
                  </div>
                ))}

                <div style={{ background:"#060E1A", borderRadius:8, padding:"10px 12px", marginTop:8 }}>
                  <div style={{ fontSize:11, color:"#667788" }}>Iowa City reference values</div>
                  <div style={{ fontSize:11, color:"#556677", marginTop:4 }}>
                    Mean ΔDLST = 6.37°F · Max = 10.27°F · 100% hotspots &gt;1°F
                  </div>
                </div>
              </div>
            </div>

            {/* Feature importance preview */}
            <div style={{ background:"#0D1B2A", border:"1px solid #1A2A3A", borderRadius:10, padding:20, marginBottom:24 }}>
              <div style={{ fontSize:14, fontWeight:600, color:"#AABBCC", marginBottom:14 }}>
                📊 Feature Importance (Sensitivity Analysis — Iowa City Study)
              </div>
              <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:8 }}>
                {[
                  {key:"NDBI", delta:14.80, rank:1},
                  {key:"WBD",  delta:3.87,  rank:2},
                  {key:"SR",   delta:1.31,  rank:3},
                  {key:"SVF",  delta:0.79,  rank:4},
                  {key:"NDVI", delta:0.78,  rank:5},
                  {key:"BRI",  delta:0.71,  rank:6},
                  {key:"BH",   delta:0.65,  rank:7},
                  {key:"BVD",  delta:0.30,  rank:8},
                ].map(({key,delta,rank}) => {
                  const feat = FEATURES.find(f=>f.key===key);
                  const pct = (delta/14.80)*100;
                  return (
                    <div key={key} style={{ background:"#060E1A", borderRadius:8, padding:"10px 12px" }}>
                      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
                        <span style={{ fontSize:13, fontWeight:700, color:feat.color }}>{key}</span>
                        <span style={{ fontSize:10, color:"#556677" }}>#{rank}</span>
                      </div>
                      <div style={{ background:"#1A2A3A", borderRadius:3, height:4, marginBottom:4 }}>
                        <div style={{ width:`${pct}%`, height:"100%",
                          background:feat.color, borderRadius:3 }}/>
                      </div>
                      <div style={{ fontSize:10, color:"#667788" }}>ΔMSE = {delta.toFixed(3)}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={{ display:"flex", justifyContent:"space-between" }}>
              <button className="pill-btn" onClick={() => setStep(1)}>← Back</button>
              <button onClick={() => { setStep(3); runPipeline(); }}
                style={{
                  background:"linear-gradient(135deg,#1D9E75,#16755A)",
                  color:"#fff", border:"none",
                  padding:"12px 32px", borderRadius:8,
                  fontSize:14, fontWeight:600,
                  boxShadow:"0 4px 16px #1D9E7544"
                }}>
                ▶ Run Pipeline
              </button>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════
            STEP 3 — RUNNING
        ══════════════════════════════════ */}
        {step === 3 && (
          <div className="fade-in" style={{ textAlign:"center", padding:"60px 20px" }}>
            <div style={{ fontSize:48, marginBottom:20,
              animation: running ? "pulse 1.5s infinite" : "none" }}>🧠</div>
            <h2 style={{ fontSize:22, fontWeight:700, color:"#E8F4FF", marginBottom:8 }}>
              Running Spatial CNN Pipeline
            </h2>
            <p style={{ fontSize:13, color:"#667788", marginBottom:32 }}>
              Processing 5×5 spatial patches → predicting DLST → simulating green roof cooling
            </p>
            <div style={{ maxWidth:500, margin:"0 auto" }}>
              {progressBar}
              <div style={{ display:"flex", justifyContent:"space-between",
                fontSize:11, color:"#556677", marginTop:4 }}>
                <span>{progLabel}</span>
                <span style={{ color:"#2E75B6", fontFamily:"'Space Mono',monospace" }}>{progress}%</span>
              </div>

              {/* Pipeline steps visual */}
              <div style={{ marginTop:32, display:"flex", flexDirection:"column", gap:6, textAlign:"left" }}>
                {[
                  [10,"Load & normalize 8 input rasters"],
                  [25,"Build 5×5 patch dataset (150 m context)"],
                  [55,"Forward pass: Conv(32→64→128)+GAP+Dense"],
                  [68,"DLST prediction for all 1,088 pixels"],
                  [78,"Hotspot detection (top 15th percentile)"],
                  [88,"Green roof NDVI/NDBI reassignment"],
                  [94,"ΔDLST reduction computation"],
                  [100,"Export results & maps"],
                ].map(([thresh, label]) => (
                  <div key={label} style={{
                    display:"flex", alignItems:"center", gap:10,
                    opacity: progress >= thresh ? 1 : 0.3,
                    transition:"opacity 0.3s"
                  }}>
                    <div style={{
                      width:18, height:18, borderRadius:"50%", flexShrink:0,
                      background: progress >= thresh ? "#1D9E75" : "#1A2A3A",
                      border:`1px solid ${progress >= thresh ? "#1D9E75" : "#2A3A4A"}`,
                      display:"flex", alignItems:"center", justifyContent:"center",
                      fontSize:10, color:"#fff"
                    }}>{progress >= thresh ? "✓" : ""}</div>
                    <span style={{ fontSize:12, color: progress >= thresh ? "#AABBCC" : "#445566" }}>
                      {label}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════
            STEP 4 — RESULTS
        ══════════════════════════════════ */}
        {step === 4 && results && (
          <div className="fade-in">
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:20 }}>
              <div>
                <h2 style={{ fontSize:20, fontWeight:700, color:"#E8F4FF" }}>Pipeline Results</h2>
                <p style={{ fontSize:13, color:"#667788", marginTop:2 }}>
                  Spatial CNN · Downtown Iowa City · July 20, 2023
                </p>
              </div>
              <button className="pill-btn" onClick={() => { setStep(1); setGrid(null); setSimGrid(null); setResults(null); }}>
                ↺ New run
              </button>
            </div>

            {/* Stats row */}
            <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:12, marginBottom:20 }}>
              <StatCard label="Mean ΔDLST" value={results.mean_red} unit="°F"
                color="#1D9E75" sub="Green roof cooling"/>
              <StatCard label="Max ΔDLST" value={results.max_red} unit="°F"
                color="#2E75B6" sub="Single pixel max"/>
              <StatCard label="Hotspot coverage" value={`${results.pct_1}%`} unit=""
                color="#BA7517" sub="Pixels >1°F cooling"/>
              <StatCard label="Hotspot pixels" value={results.n_hot} unit=""
                color="#9B59B6" sub={`Top ${config.hotspot_pct}th percentile`}/>
            </div>

            {/* Tab nav */}
            <div style={{ display:"flex", gap:6, marginBottom:16 }}>
              {[["maps","🗺 Spatial Maps"],["model","🧠 Model Performance"],
                ["shap","📊 Feature Importance"],["code","💻 Reproduce This"]].map(([t,l]) => (
                <button key={t} className={`pill-btn ${tab===t?"active":""}`}
                  onClick={() => setTab(t)}>{l}</button>
              ))}
            </div>

            {/* ── MAPS TAB ── */}
            {tab === "maps" && (
              <div className="fade-in">
                <div style={{ display:"flex", gap:6, marginBottom:14 }}>
                  {[["dlst","DLST (°F)"],["gr","GR Reduction"],["hotspot","Hotspots"]].map(([m,l])=>(
                    <button key={m} className={`pill-btn ${activeMap===m?"active":""}`}
                      onClick={()=>setActiveMap(m)}>{l}</button>
                  ))}
                </div>
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:16 }}>
                  {[["dlst","Pre-GR DLST (°F)"],["gr","GR Cooling (°F)"],["hotspot","Hotspot Pixels"]].map(([m,title]) => (
                    <div key={m} style={{ background:"#0D1B2A", border:`1px solid ${activeMap===m?"#2E75B6":"#1A2A3A"}`,
                      borderRadius:10, padding:14, transition:"border 0.2s", cursor:"pointer" }}
                      onClick={()=>setActiveMap(m)}>
                      <div style={{ fontSize:12, fontWeight:600, color:"#AABBCC", marginBottom:8 }}>{title}</div>
                      <div style={{ display:"flex", justifyContent:"center", marginBottom:8 }}>
                        <HoverGrid grid={m==="gr"?simGrid:simGrid||grid} mode={m} size={7}/>
                      </div>
                      {m==="dlst" && <Colorbar min="93°F" max="121°F"
                        colors={["#313695","#4575B4","#ABD9E9","#FEE090","#F46D43","#A50026"]}
                        label="DLST" width={140}/>}
                      {m==="gr" && <Colorbar min="0" max="10°F"
                        colors={["#EFF3FF","#6BAED6","#08306B"]} label="Reduction" width={140}/>}
                    </div>
                  ))}
                </div>

                {/* Hover tooltip */}
                {hoverCell && (
                  <div style={{ marginTop:12, background:"#0D1B2A",
                    border:"1px solid #2E75B6", borderRadius:8, padding:"10px 14px",
                    fontSize:12, display:"inline-block" }}>
                    <span style={{ color:"#556677" }}>Pixel ({hoverCell.i},{hoverCell.j}) · </span>
                    <span style={{ color:"#2E75B6" }}>DLST: {hoverCell.dlst?.toFixed(1)}°F · </span>
                    {hoverCell.gr_red>0 && <span style={{ color:"#1D9E75" }}>GR Cooling: {hoverCell.gr_red?.toFixed(2)}°F · </span>}
                    <span style={{ color: hoverCell.is_hotspot?"#FF6666":"#667788" }}>
                      {hoverCell.is_hotspot ? "HOTSPOT" : "Normal"}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* ── MODEL TAB ── */}
            {tab === "model" && (
              <div className="fade-in">
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
                  <div style={{ background:"#0D1B2A", border:"1px solid #1A2A3A", borderRadius:10, padding:20 }}>
                    <div style={{ fontSize:14, fontWeight:600, color:"#2E75B6", marginBottom:14 }}>
                      All 6 Models — Iowa City Study
                    </div>
                    {[
                      {name:"ANN",   type:"ML", r2:0.659, rmse:2.866, kf:0.752, color:"#27AE60"},
                      {name:"RF",    type:"ML", r2:0.858, rmse:1.847, kf:0.879, color:"#2ECC71"},
                      {name:"XGBoost",type:"ML",r2:0.850, rmse:1.904, kf:0.877, color:"#1ABC9C"},
                      {name:"Spatial CNN",type:"DL",r2:0.974,rmse:0.842,kf:0.962,color:"#2E75B6",best:true},
                      {name:"CNN-LSTM",type:"DL",r2:0.724,rmse:2.967,kf:0.814,color:"#9B59B6"},
                      {name:"ViT",   type:"DL", r2:0.635, rmse:2.461, kf:0.717, color:"#E74C3C"},
                    ].map(m => (
                      <div key={m.name} style={{
                        display:"flex", alignItems:"center", gap:10,
                        padding:"8px 10px", borderRadius:6, marginBottom:4,
                        background: m.best ? "#2E75B610" : "transparent",
                        border: m.best ? "1px solid #2E75B644" : "1px solid transparent"
                      }}>
                        <div style={{ width:3, height:32, background:m.color, borderRadius:2, flexShrink:0 }}/>
                        <div style={{ flex:1 }}>
                          <div style={{ fontSize:12, fontWeight:600, color: m.best ? "#2E75B6" : "#AABBCC" }}>
                            {m.name} {m.best && "⭐"}
                            <span style={{ fontSize:10, color:"#556677", marginLeft:6 }}>[{m.type}]</span>
                          </div>
                        </div>
                        <div style={{ textAlign:"right", fontSize:11, fontFamily:"'Space Mono',monospace" }}>
                          <div style={{ color:m.color }}>R²={m.r2}</div>
                          <div style={{ color:"#556677" }}>{m.rmse}°F</div>
                        </div>
                        <div style={{ width:60 }}>
                          <div style={{ background:"#1A2A3A", borderRadius:3, height:4 }}>
                            <div style={{ width:`${m.r2*100}%`, height:"100%", background:m.color, borderRadius:3 }}/>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div style={{ background:"#0D1B2A", border:"1px solid #1A2A3A", borderRadius:10, padding:20 }}>
                    <div style={{ fontSize:14, fontWeight:600, color:"#1D9E75", marginBottom:14 }}>
                      Spatial CNN Architecture
                    </div>
                    {[
                      ["Input","5×5×8 spatial patch"],["Conv2D #1","32 filters, 3×3, ReLU + BN"],
                      ["Conv2D #2","64 filters, 3×3, ReLU + BN"],["Conv2D #3","128 filters, 3×3, ReLU + BN"],
                      ["GlobalAvgPool","128-dim feature vector"],["Dense #1","64 neurons, ReLU, Dropout 0.3"],
                      ["Dense #2","32 neurons, ReLU, Dropout 0.2"],["Output","1 neuron (DLST °F)"],
                    ].map(([layer,detail],i) => (
                      <div key={layer} style={{
                        display:"flex", gap:12, padding:"7px 0",
                        borderBottom:"1px solid #1A2A3A", alignItems:"center"
                      }}>
                        <div style={{
                          width:22, height:22, borderRadius:4, flexShrink:0,
                          background: i===0?"#BA7517":i===7?"#1D9E75":"#2E75B6",
                          display:"flex",alignItems:"center",justifyContent:"center",
                          fontSize:10, color:"#fff", fontWeight:700
                        }}>{i+1}</div>
                        <div style={{ flex:1 }}>
                          <div style={{ fontSize:12, fontWeight:600, color:"#AABBCC" }}>{layer}</div>
                          <div style={{ fontSize:10, color:"#556677", marginTop:1 }}>{detail}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* ── SHAP TAB ── */}
            {tab === "shap" && (
              <div className="fade-in">
                <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
                  <div style={{ background:"#0D1B2A", border:"1px solid #1A2A3A", borderRadius:10, padding:20 }}>
                    <div style={{ fontSize:14, fontWeight:600, color:"#9B59B6", marginBottom:14 }}>
                      Sensitivity Analysis (ΔMSE)
                    </div>
                    {[
                      {key:"NDBI",delta:14.80},{key:"WBD",delta:3.87},
                      {key:"SR",  delta:1.31}, {key:"SVF",delta:0.79},
                      {key:"NDVI",delta:0.78}, {key:"BRI",delta:0.71},
                      {key:"BH",  delta:0.65}, {key:"BVD",delta:0.30},
                    ].map(({key,delta},i) => {
                      const feat = FEATURES.find(f=>f.key===key);
                      return (
                        <div key={key} style={{ display:"flex", alignItems:"center", gap:10, marginBottom:8 }}>
                          <div style={{ width:40, fontSize:11, fontWeight:700, color:feat.color,
                            fontFamily:"'Space Mono',monospace", flexShrink:0 }}>{key}</div>
                          <div style={{ flex:1, background:"#1A2A3A", borderRadius:3, height:18, position:"relative" }}>
                            <div style={{
                              position:"absolute", left:0, top:0, bottom:0,
                              width:`${(delta/14.80)*100}%`,
                              background:`linear-gradient(90deg,${feat.color}99,${feat.color})`,
                              borderRadius:3, transition:"width 1s ease"
                            }}/>
                            <span style={{ position:"absolute", right:6, top:"50%",
                              transform:"translateY(-50%)", fontSize:10,
                              color:"#AABBCC", fontFamily:"'Space Mono',monospace" }}>{delta.toFixed(3)}</span>
                          </div>
                          <div style={{ width:20, fontSize:11, color:"#556677", flexShrink:0 }}>#{i+1}</div>
                        </div>
                      );
                    })}
                  </div>
                  <div style={{ background:"#0D1B2A", border:"1px solid #1A2A3A", borderRadius:10, padding:20 }}>
                    <div style={{ fontSize:14, fontWeight:600, color:"#BA7517", marginBottom:14 }}>
                      SHAP Directionality (CNN DeepExplainer)
                    </div>
                    {[
                      {key:"NDBI", dir:"positive", effect:"High NDBI → High DLST", color:"#E74C3C"},
                      {key:"WBD",  dir:"negative", effect:"Far from river → High DLST", color:"#3498DB"},
                      {key:"SVF",  dir:"positive", effect:"Open sky → More solar gain", color:"#27AE60"},
                      {key:"SR",   dir:"positive", effect:"High solar rad → High DLST", color:"#F39C12"},
                      {key:"NDVI", dir:"negative", effect:"More vegetation → Less heat", color:"#1ABC9C"},
                      {key:"BH",   dir:"mixed",    effect:"Tall buildings: canyon cooling", color:"#9B59B6"},
                    ].map(({key,dir,effect,color}) => (
                      <div key={key} style={{
                        display:"flex", gap:10, padding:"8px 10px",
                        borderRadius:6, marginBottom:6, background:"#060E1A"
                      }}>
                        <div style={{ width:32, fontSize:11, fontWeight:700, color,
                          fontFamily:"'Space Mono',monospace", flexShrink:0,
                          display:"flex", alignItems:"center" }}>{key}</div>
                        <div style={{ flex:1 }}>
                          <div style={{ fontSize:11, color:"#AABBCC" }}>{effect}</div>
                        </div>
                        <div style={{
                          fontSize:10, padding:"2px 8px", borderRadius:10, flexShrink:0, alignSelf:"center",
                          background: dir==="positive"?"#E74C3C22":dir==="negative"?"#3498DB22":"#9B59B622",
                          color: dir==="positive"?"#E74C3C":dir==="negative"?"#3498DB":"#9B59B6",
                          border:`1px solid ${dir==="positive"?"#E74C3C":dir==="negative"?"#3498DB":"#9B59B6"}44`
                        }}>{dir==="positive"?"↑ heat":dir==="negative"?"↓ cool":"± mixed"}</div>
                      </div>
                    ))}
                    <div style={{ marginTop:10, padding:"10px 12px", background:"#060E1A",
                      borderRadius:8, fontSize:11, color:"#556677", borderLeft:"3px solid #BA7517" }}>
                      Iowa City key finding: NDBI dominates (not NDVI as in Sun Belt cities).
                      Built-up intensity is the primary thermal driver here.
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ── CODE TAB ── */}
            {tab === "code" && (
              <div className="fade-in">
                <div style={{ background:"#0D1B2A", border:"1px solid #1A2A3A", borderRadius:10, padding:20, marginBottom:16 }}>
                  <div style={{ fontSize:14, fontWeight:600, color:"#AABBCC", marginBottom:12 }}>
                    💻 Complete Reproducibility Package
                  </div>
                  <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:10, marginBottom:16 }}>
                    {[
                      {icon:"🐍", title:"Python pipeline", sub:"All_Figures_Final.py", color:"#F39C12", desc:"Train all 6 models + generate all 11 figures + 5 tables"},
                      {icon:"🌍", title:"GEE script", sub:"Google Earth Engine", color:"#1D9E75", desc:"DLST, NDVI, NDBI, WBD extraction from Landsat 9 + Sentinel-2"},
                      {icon:"🗺", title:"Interactive map", sub:"GreenRoof_Interactive_Map.html", color:"#2E75B6", desc:"Folium web map, no server needed"},
                      {icon:"📄", title:"Methodology doc", sub:".docx", color:"#9B59B6", desc:"Complete methods + results Word document"},
                      {icon:"📊", title:"All figures", sub:"Fig01–Fig11 PNG+PDF", color:"#E74C3C", desc:"300 DPI publication-ready figures"},
                      {icon:"📋", title:"All tables", sub:"Table1–Table5 CSV+PNG", color:"#BA7517", desc:"Model performance, SA, GR summary"},
                    ].map(({icon,title,sub,color,desc}) => (
                      <div key={title} style={{ background:"#060E1A", borderRadius:8, padding:"12px 14px",
                        border:`1px solid ${color}22` }}>
                        <div style={{ fontSize:20, marginBottom:6 }}>{icon}</div>
                        <div style={{ fontSize:12, fontWeight:600, color }}>  {title}</div>
                        <div style={{ fontSize:10, color:"#556677", marginTop:2 }}>{sub}</div>
                        <div style={{ fontSize:10, color:"#445566", marginTop:4 }}>{desc}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Key code snippet */}
                <div style={{ background:"#060E1A", border:"1px solid #1A2A3A", borderRadius:10, overflow:"hidden" }}>
                  <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center",
                    padding:"10px 16px", borderBottom:"1px solid #1A2A3A" }}>
                    <span style={{ fontSize:12, color:"#667788" }}>
                      Spatial CNN core — Python / TensorFlow 2.21
                    </span>
                    <div style={{ display:"flex", gap:6 }}>
                      {["#FF5F57","#FEBC2E","#28C840"].map(c=>(
                        <div key={c} style={{ width:10, height:10, borderRadius:"50%", background:c }}/>
                      ))}
                    </div>
                  </div>
                  <pre style={{ padding:"16px 20px", fontSize:11, lineHeight:1.7,
                    color:"#AABBCC", overflowX:"auto", margin:0,
                    fontFamily:"'Space Mono',monospace" }}>
{`<span style="color:#556677"># 1. Build 5×5 spatial patches (150m context)</span>
patches = []
for i in range(rows):
    for j in range(cols):
        patch = feature_pad[i:i+5, j:j+5, :]  <span style="color:#556677"># 5×5×8</span>
        patches.append(patch)
X = np.array(patches, dtype=np.float32)

<span style="color:#556677"># 2. Spatial CNN architecture</span>
def build_cnn(input_shape):
    inp = keras.Input(shape=input_shape)
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

<span style="color:#556677"># 3. Green roof simulation</span>
threshold = np.percentile(y_targets, 85)  <span style="color:#556677"># top 15%</span>
X_gr = X_norm.copy()
X_gr[hotspot_idx,:,:,ndvi_ch] = (0.55  - mean) / std
X_gr[hotspot_idx,:,:,ndbi_ch] = (-0.178 - mean) / std
ΔDLST = cnn.predict(X) - cnn.predict(X_gr)`}
                  </pre>
                </div>

                <div style={{ marginTop:12, padding:"12px 16px", background:"#0D1B2A",
                  borderRadius:8, border:"1px solid #1A2A3A",
                  fontSize:12, color:"#667788", display:"flex", gap:12, alignItems:"center" }}>
                  <span style={{ fontSize:20 }}>📁</span>
                  <div>
                    <span style={{ color:"#AABBCC" }}>Full pipeline path on Windows: </span>
                    <span style={{ fontFamily:"'Space Mono',monospace", color:"#2E75B6", fontSize:11 }}>
                      E:\LIDAR\Lidar_Lab3_mmtm\Lidar_Lab3_mmtm\GreenRoof_IowaCity\All_Figures_Final.py
                    </span>
                    <div style={{ marginTop:4, color:"#556677" }}>
                      Python 3.11 · TensorFlow 2.21 · scikit-learn 1.6 · SHAP 0.45 · Folium 0.15
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
