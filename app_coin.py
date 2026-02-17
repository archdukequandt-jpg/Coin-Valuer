import math
import json
from typing import Optional, Dict

import pandas as pd
import streamlit as st

from coin_valuer import db, match, pricing, utils, nn

st.set_page_config(layout="wide", page_title="Coin Identifier & Valuer", page_icon="🪙")

COIN_LINKS = {
    "NGC Coin Explorer": "https://www.ngccoin.com/coin-explorer/",
    "PCGS CoinFacts": "https://www.pcgs.com/coinfacts",
    "Numista (global catalog)": "https://en.numista.com/",
    "US Mint — Coin Specifications": "https://www.usmint.gov/learn/coin-and-medal-programs/coin-specifications",
    "Royal Mint (UK) — Coinage": "https://www.royalmint.com/discover/uk-coins/",
}

CREATORS = [
    ("Ryan Childs", "ryanchilds10@gmail.com"),
    ("James Quandt", "archdukequandt@gmail.com"),
    ("James Belhund", "jamesbelhund@gmail.com"),
]

def _inject_css() -> None:
    st.markdown(
        """
<style>
section.main > div.block-container {max-width: 1500px; padding-left: 2rem; padding-right: 2rem}
.hero {background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #312e81 100%); color: #e5e7eb;
       padding: 26px 22px; border-radius: 18px; margin-bottom: 18px; border: 1px solid rgba(255,255,255,0.08);
       box-shadow: 0 6px 20px rgba(2, 6, 23, 0.45);}
.hero h1 {margin: 0 0 6px 0; font-size: 30px; letter-spacing: 0.2px}
.hero p  {margin: 2px 0; font-size: 14px; color: #cbd5e1}
.link-row {display:flex; flex-wrap: wrap; gap:10px; margin: 10px 0 0}
.link-pill {display:inline-flex; align-items:center; gap:8px; padding: 8px 12px; border-radius: 999px;
           border: 1px solid rgba(148,163,184,0.35); background: rgba(15,23,42,0.4); color: #e2e8f0 !important;
           text-decoration:none !important; font-size: 13px;}
.link-pill:hover {border-color: #93c5fd; background: rgba(2,132,199,0.15)}
.link-pill .dot {width:8px; height:8px; border-radius:50%; background:#60a5fa}
[data-testid="stSidebar"] .block-container {padding: 1rem 1rem}
</style>
        """,
        unsafe_allow_html=True,
    )

def _render_hero():
    creators_html = " • ".join(
        [f"<strong>{n}</strong> (<a href='mailto:{e}' style='color:#93c5fd'>{e}</a>)" for n, e in CREATORS]
    )
    links_html = "\n".join(
        [f"<a class='link-pill' href='{u}' target='_blank'><span class='dot'></span>{k}</a>" for k, u in COIN_LINKS.items()]
    )
    st.markdown(
        f"""
<div class="hero">
  <h1>Coin Identifier & Valuer</h1>
  <p>Created by: {creators_html}</p>
  <div class="link-row">{links_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def _load_db_df() -> pd.DataFrame:
    return db.current_db()

@st.cache_data(show_spinner=False)
def _spot_prices() -> Dict[str, Dict[str, float]]:
    return pricing.spot_prices()

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _estimate_scrap_value(metals: Dict[str, Dict[str, float]], mass_g: float, m1: str, m2: str, p2: float) -> float:
    if not mass_g or mass_g <= 0:
        return 0.0
    m1 = (m1 or "").strip().lower()
    m2 = (m2 or "").strip().lower()
    p2 = float(p2 or 0.0)
    p2 = min(max(p2, 0.0), 1.0)
    p1 = 1.0 - p2

    def per_g(m):
        info = metals.get(m, {})
        return float(info.get("per_g", 0.0))

    return p1 * mass_g * per_g(m1) + p2 * mass_g * per_g(m2)

def _sidebar_obs():
    st.sidebar.header("Observation")
    country = st.sidebar.text_input("Country (optional)", "")
    denom = st.sidebar.text_input("Denomination (optional)", "")
    year = st.sidebar.number_input("Year on coin (optional)", min_value=0, max_value=9999, value=0, step=1)
    color = st.sidebar.selectbox("Color / appearance", ["(unknown)", "coppery", "silvery", "gold-like"], index=0)
    st.sidebar.markdown("---")
    outer_mm = st.sidebar.number_input("Outer diameter (mm)", min_value=0.0, value=24.0, step=0.1)
    inner_mm = st.sidebar.number_input("Inner hole diameter (mm)", min_value=0.0, value=0.0, step=0.1)
    thickness_mm = st.sidebar.number_input("Thickness (mm)", min_value=0.0, value=1.8, step=0.1)
    mass_g = st.sidebar.number_input("Mass (g)", min_value=0.0, value=5.0, step=0.01)
    st.sidebar.markdown("---")
    run = st.sidebar.button("Identify Coin", type="primary")
    return {
        "country": country.strip() or None,
        "denomination": denom.strip() or None,
        "year": int(year) if year else None,
        "color_hint": None if color.startswith("(") else color,
        "diameter_mm": float(outer_mm),
        "hole_mm": float(inner_mm),
        "thickness_mm": float(thickness_mm),
        "mass_g": float(mass_g),
        "run": run,
    }

def _render_physics(obs):
    vol_mm3 = utils.volume_mm3(obs["diameter_mm"], obs["thickness_mm"], inner_mm=obs["hole_mm"])
    dens = utils.density_gcm3(obs["mass_g"], obs["diameter_mm"], obs["thickness_mm"], obs["hole_mm"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Volume (mm³)", f"{vol_mm3:,.2f}")
    c2.metric("Mass (g)", f"{obs['mass_g']:.3f}")
    c3.metric("Density (g/cm³)", f"{dens:.3f}" if dens and math.isfinite(dens) else "—")
    return dens

def _identification_tab(df: pd.DataFrame, obs: Dict):
    dens = _render_physics(obs)
    if not obs["run"]:
        st.info("Enter measurements on the left and click **Identify Coin**.")
        return

    if df is None or df.empty:
        st.error("Database is empty.")
        st.stop()

    obs_for_match = {
        "year": obs["year"],
        "diameter_mm": obs["diameter_mm"],
        "thickness_mm": obs["thickness_mm"],
        "mass_g": obs["mass_g"],
        "hole_mm": obs["hole_mm"],
        "density_gcm3": dens,
        "country": obs["country"],
        "denomination": obs["denomination"],
    }

    rows = df.to_dict("records")
    scored = match.rank_candidates(rows, obs_for_match, top_n=12)

    metals = _spot_prices()
    table = []
    for r, score in scored:
        m1 = r.get("metal") or r.get("metal_1") or "unknown"
        m2 = r.get("metal2") or r.get("metal_2") or "none"
        p2 = r.get("pct2") if r.get("pct2") is not None else r.get("pct_2")
        p2 = float(p2 or 0.0)
        scrap = _estimate_scrap_value(metals, float(obs["mass_g"] or 0.0), m1, m2, p2)

        table.append({
            "Country": r.get("country",""),
            "Denomination": r.get("denomination",""),
            "Years": f"{int(r.get('year_start') or 0)}–{int(r.get('year_end') or 0)}",
            "Diameter": r.get("diameter_mm",""),
            "Thickness": r.get("thickness_mm",""),
            "Metal_1": r.get("metal_1",""),
            "Pct_1": f"{100*float(r.get('pct_1') or 0):.1f}%",
            "Metal_2": r.get("metal_2",""),
            "Pct_2": f"{100*float(r.get('pct_2') or 0):.1f}%",
            "Metal_3": r.get("metal_3",""),
            "Pct_3": f"{100*float(r.get('pct_3') or 0):.1f}%",
            "Match": f"{score*100:.1f}%",
            "Scrap (USD)": f"${scrap:,.2f}",
        })
    st.subheader("Likely matches")
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

def _metal_ai_tab(df: pd.DataFrame):
    st.markdown("## Metal Composition AI (Keras)")
    st.caption("The model trains on the database composition columns: metal_1..metal_3 and pct_1..pct_3 (preferred).")

    with st.expander("Train / update model", expanded=False):
        epochs = st.slider("Epochs", 10, 300, 60, step=10, key="epochs")
        if st.button("Train model", type="primary", key="train"):
            try:
                train_df, metals = nn.build_training_frame(df)
                cfg = nn.TrainConfig(epochs=int(epochs), metals=metals)
                metrics = nn.train_model(train_df, cfg)
                st.success("Model trained and saved.")
                st.json(metrics)
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.markdown("### Predict")
    c1, c2 = st.columns(2)
    with c1:
        mass_g = st.number_input("Mass (g)", min_value=0.01, value=6.25, step=0.01)
        diameter_mm = st.number_input("Diameter (mm)", min_value=0.01, value=24.26, step=0.01)
        thickness_mm = st.number_input("Thickness (mm)", min_value=0.0, value=1.75, step=0.01)
        hole_mm = st.number_input("Hole diameter (mm)", min_value=0.0, value=0.0, step=0.01)
    with c2:
        color_hint = st.selectbox("Color hint", ["silver","gold","coppery","unknown"], index=0)
        denom = st.text_input("Denomination (optional)", "")
        year = st.number_input("Year (optional)", min_value=0, max_value=2100, value=0, step=1)

    if st.button("Predict", key="predict"):
        probs = nn.predict_metal_probs(
            mass_g=float(mass_g),
            diameter_mm=float(diameter_mm),
            thickness_mm=float(thickness_mm),
            hole_mm=float(hole_mm),
            denomination=denom,
            color_hint=color_hint,
            year=int(year) if int(year)>0 else None
        )
        items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        st.write(", ".join([f"**{m}** ({p*100:.1f}%)" for m,p in items[:5]]))
        st.download_button("Download probabilities (JSON)", json.dumps(probs, indent=2), "metal_probs.json", "application/json")

def _database_tab(df: pd.DataFrame):
    st.subheader("Database")
    st.caption("This view is loaded from data/coin_db_seed.csv + optional cache. The composition columns are used to train the AI.")
    st.dataframe(df, use_container_width=True)

def _resources_tab():
    st.subheader("Resources")
    for k,u in COIN_LINKS.items():
        st.markdown(f"- [{k}]({u})")
    st.markdown("---")
    st.markdown("**Creators / Contact**")
    for n,e in CREATORS:
        st.markdown(f"- {n}: [{e}](mailto:{e})")

def main():
    _inject_css()
    _render_hero()
    obs = _sidebar_obs()
    df = _load_db_df()

    tabs = st.tabs(["🔎 Identification", "🤖 Metal AI", "📚 Database", "🔗 Resources"])
    with tabs[0]:
        _identification_tab(df, obs)
    with tabs[1]:
        _metal_ai_tab(df)
    with tabs[2]:
        _database_tab(df)
    with tabs[3]:
        _resources_tab()

if __name__ == "__main__":
    main()
