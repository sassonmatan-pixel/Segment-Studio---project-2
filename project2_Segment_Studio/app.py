"""
Segment Studio
==============
An interactive Streamlit app for automatic customer/data segmentation.

Pipeline (matches the source notebook `project2_Segment_Studio.ipynb`):
  1. Upload & preview a CSV.
  2. Elbow method: WCSS + Silhouette score across a range of K.
  3. Fit K-Means for a chosen K, build a cluster summary table.
  4. Use an LLM (Ollama) to name & describe each cluster.
  5. Export the cluster summary table (cluster_id, count, name, description).
"""

import os
import io
import time
import random
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Page config + premium SaaS-style CSS (with a Dark/Light toggle)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Segment Studio",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Two hand-picked palettes for our own custom elements (hero, badges, cards,
# buttons, metrics). Streamlit's native widgets (inputs, dataframes) still
# follow the app's built-in theme set in .streamlit/config.toml — a full
# pixel-perfect swap of those requires Streamlit's own theme engine (the
# real toggle for those lives in the "⋮" menu → Settings → Theme).
_DARK = {
    "bg": "#0E1117",
    "bg_css": "linear-gradient(180deg, #1B1530 0%, #0E1117 55%)",
    "text": "#FAFAFA", "muted": "#B0B3B8",
    "surface": "#1E1B2E", "border": "rgba(190,160,255,0.22)",
    "metric_bg": "rgba(124,58,237,0.22)", "metric_border": "rgba(124,58,237,0.4)",
    "grid": "#363154", "accent": "#B18CFF",
    "shadow": "0 6px 20px rgba(0,0,0,0.45)",
}
_LIGHT = {
    "bg": "#F3EFFC",
    "bg_css": "linear-gradient(180deg, #ECE3FB 0%, #F6F3FD 55%)",
    "text": "#241F3D", "muted": "#655E82",
    "surface": "#FBF9FF", "border": "rgba(124,58,237,0.28)",
    "metric_bg": "rgba(124,58,237,0.10)", "metric_border": "rgba(124,58,237,0.28)",
    "grid": "#E4DDF7", "accent": "#7C3AED",
    "shadow": "0 6px 20px rgba(124,58,237,0.14)",
}
_P = _DARK if st.session_state.dark_mode else _LIGHT

CUSTOM_CSS = f"""
<style>
    .stApp {{ background: {_P['bg_css']}; }}
    .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown,
    h1, h2, h3, h4, h5, h6 {{ color: {_P['text']}; }}
    .main > div {{ padding-top: 1.5rem; }}

    /* Hero header */
    .ss-hero {{
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 60%, #C026D3 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white !important;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }}
    .ss-hero h1, .ss-hero p {{ color: white !important; }}
    .ss-hero h1 {{ margin: 0; font-size: 2rem; font-weight: 800; }}
    .ss-hero p {{ margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 1.02rem; }}

    /* Card container */
    .ss-card {{
        background: {_P['surface']};
        border: 1px solid {_P['border']};
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: {_P['shadow']};
    }}

    /* Metric tiles */
    div[data-testid="stMetric"] {{
        background: {_P['metric_bg']};
        border: 1px solid {_P['metric_border']};
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        box-shadow: {_P['shadow']};
    }}
    div[data-testid="stMetric"] * {{ color: {_P['text']} !important; }}

    /* Buttons — text lives in a nested <p>, which our own ambient text-color
       rule above (`.stApp p`) would otherwise repaint directly (a direct
       match always beats an inherited value, regardless of !important on
       the ancestor), so the button label color must be set on the
       descendants themselves, not just the <button>. */
    div.stButton > button, div.stDownloadButton > button {{
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.1rem;
        background: {_P['surface']};
        border: 1px solid {_P['border']};
        box-shadow: {_P['shadow']};
    }}
    div.stButton > button *, div.stDownloadButton > button * {{
        color: {_P['text']} !important;
    }}
    div.stButton > button[kind="primary"], div.stDownloadButton > button {{
        background: linear-gradient(135deg, #4F46E5, #C026D3);
        border: none;
    }}
    div.stButton > button[kind="primary"] *, div.stDownloadButton > button * {{
        color: white !important;
    }}
    div.stButton > button:disabled * {{
        color: {_P['muted']} !important;
    }}

    .stApp span.ss-badge {{
        display: inline-block;
        background: rgba(124, 58, 237, 0.16);
        color: {_P['accent']} !important;
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 0.5rem;
    }}

    /* File uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {{
        background: {_P['surface']} !important;
        border: 1px dashed {_P['border']} !important;
        border-radius: 12px;
    }}
    [data-testid="stFileUploaderDropzone"] * {{ color: {_P['text']} !important; }}
    [data-testid="stFileUploaderDropzone"] button {{
        background: {_P['surface']} !important;
        color: {_P['text']} !important;
        border: 1px solid {_P['border']} !important;
    }}
    [data-testid="stFileUploaderDropzoneInstructions"] svg {{ fill: {_P['muted']} !important; }}
    /* The "uploaded file" chip shown after a successful upload — has its
       own hardcoded dark background regardless of theme unless overridden. */
    [data-testid="stFileChip"] {{
        background: {_P['metric_bg']} !important;
        border: 1px solid {_P['metric_border']} !important;
        border-radius: 8px;
    }}
    [data-testid="stFileChip"] * {{ color: {_P['text']} !important; fill: {_P['text']} !important; }}

    /* Generic inputs */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] span {{
        background: {_P['surface']} !important;
        color: {_P['text']} !important;
        border-color: {_P['border']} !important;
    }}
    /* The +/- steppers live in their own container next to the input, with
       Streamlit's own hardcoded dark-theme background — left unstyled this
       shows up as a dark box glued to our light input. */
    [data-testid="stNumberInputContainer"] {{
        background: {_P['surface']} !important;
        border: 1px solid {_P['border']} !important;
        border-radius: 8px;
    }}
    [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{
        color: {_P['muted']} !important;
        background: transparent !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background: {_P['surface']};
        border: 1px solid {_P['border']};
        border-radius: 10px;
        box-shadow: {_P['shadow']};
    }}
    [data-testid="stExpander"] summary {{ color: {_P['text']} !important; }}

    /* Plain HTML tables (st.table) — these DO follow our palette fully,
       unlike st.dataframe's canvas-rendered grid. */
    [data-testid="stTable"] {{
        border-radius: 10px;
        overflow: hidden;
        box-shadow: {_P['shadow']};
    }}
    [data-testid="stTable"] table {{
        color: {_P['text']};
        border-collapse: collapse;
        width: 100%;
    }}
    [data-testid="stTable"] th {{
        background: {_P['metric_bg']};
        color: {_P['text']} !important;
        padding: 0.5rem 0.7rem;
        border-bottom: 1px solid {_P['border']};
    }}
    [data-testid="stTable"] td {{
        background: {_P['surface']};
        color: {_P['text']} !important;
        padding: 0.4rem 0.7rem;
        border-bottom: 1px solid {_P['border']};
    }}

    /* st.dataframe outer chrome (its internal grid is canvas-rendered and
       follows Streamlit's own base theme, not this CSS — see the note by
       the theme toggle). */
    [data-testid="stDataFrame"] {{
        border: 1px solid {_P['border']};
        border-radius: 10px;
    }}

    /* Streamlit's own top toolbar */
    [data-testid="stHeader"] {{ background: {_P['bg']} !important; }}
    [data-testid="stHeader"] {{ color: {_P['text']} !important; }}
    [data-testid="stHeader"] svg {{ color: {_P['text']} !important; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

hero_col, toggle_col = st.columns([7, 1])
with hero_col:
    st.markdown(
        """
        <div class="ss-hero">
            <h1>🧬 Segment Studio</h1>
            <p>Upload any CSV, discover natural customer segments with K-Means,
            and let an LLM name them for you — end to end, no code required.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with toggle_col:
    st.write("")
    toggle_label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
    if st.button(toggle_label, key="theme_toggle", width="stretch"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.caption("Everything on this page follows this toggle now.")

# --------------------------------------------------------------------------
# Core logic (ported from the notebook)
# --------------------------------------------------------------------------

def norma_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Prepare a raw DataFrame for K-Means.

    - Drops rows with missing values.
    - Drops the first column if it looks like an ID column.
    - One-hot encodes categorical (object) columns.
    - Scales everything with StandardScaler.
    """
    df = df_raw.dropna()

    first_col = df.columns[0]
    if "id" in str(first_col).lower():
        df = df.drop(columns=[first_col])

    for c in df.columns:
        # Treat both classic "object" columns and pandas' newer native
        # string dtype (default since pandas 3.0) as categorical text.
        if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
            df = pd.get_dummies(df, columns=[c], drop_first=True, dtype=int)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    df_norm = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return df_norm


def drop_leading_id_column(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Mirror norma_csv's ID-detection so the profile table stays consistent
    with the normalized data (a real feature in column 0 won't be dropped)."""
    first_col = df_raw.columns[0]
    if "id" in str(first_col).lower():
        return df_raw.drop(columns=[first_col])
    return df_raw


def remove_outliers_iqr(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Optional bonus: drop rows outside 1.5*IQR on any numeric column."""
    df = df_raw.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    mask = pd.Series(True, index=df.index)
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= df[col].between(low, high)
    return df[mask]


def compute_cluster_profile(df_with_cluster: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster mean of numeric features + most frequent (mode) value of
    categorical features, computed on the original (unscaled, non-one-hot)
    data — this is the profile sent to the LLM for naming each segment."""
    numeric_cols = [c for c in df_with_cluster.select_dtypes(include=np.number).columns if c != "cluster"]
    categorical_cols = [c for c in df_with_cluster.columns if c not in numeric_cols and c != "cluster"]

    grouped = df_with_cluster.groupby("cluster")
    profile = grouped[numeric_cols].mean().round(1) if numeric_cols else pd.DataFrame(index=grouped.groups.keys())

    for col in categorical_cols:
        profile[col] = grouped[col].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "")

    return profile


SILHOUETTE_TOLERANCE = 0.02  # see auto_select_k
MIN_MEANINGFUL_SILHOUETTE = 0.5  # Kaufman & Rousseeuw: >0.5 = "reasonable structure found"


def find_wcss_elbow_k(elbow_table: pd.DataFrame) -> int:
    """Find the elbow of the WCSS curve via max-distance-from-chord: draw a
    straight line from the first tested K to the last, and pick the K whose
    point bulges furthest from that line. This is what tells the difference
    between "still worth splitting further" and "diminishing returns" —
    Silhouette Score alone can't see it (see auto_select_k)."""
    table = elbow_table.sort_values("K").reset_index(drop=True)
    k = table["K"].to_numpy(dtype=float)
    wcss = table["WCSS"].to_numpy(dtype=float)
    if len(k) < 3:
        return int(k[0])
    x = (k - k.min()) / (k.max() - k.min())
    y = (wcss - wcss.min()) / (wcss.max() - wcss.min())
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec /= np.linalg.norm(line_vec)
    points = np.column_stack([x - x[0], y - y[0]])
    projection = np.outer(points @ line_vec, line_vec)
    distance_from_chord = np.linalg.norm(points - projection, axis=1)
    return int(table["K"].iloc[np.argmax(distance_from_chord)])


def auto_select_k(elbow_table: pd.DataFrame) -> int:
    """Pick a K from the WCSS + Silhouette Score table.

    Silhouette Score alone has a real blind spot: on data with hierarchical
    structure (e.g. a coarse Male/Female split that each further split into
    finer spending-behavior segments — common in real customer data), the
    *coarse* split can score higher on Silhouette than the finer one, simply
    because the coarse groups are more separated. Picking the highest score
    (or the smallest K within a small tolerance of it, which is all a plain
    tolerance rule can do) then recommends a K that's too small, even though
    a KMeans fit with n_init=10 makes each individual K's score itself fully
    reproducible — the run-to-run "inconsistency" this was reported as isn't
    randomness, it's this blind spot showing up on some datasets and not
    others.

    Fix: first find the WCSS elbow (find_wcss_elbow_k) — the point past
    which adding more clusters stops meaningfully reducing within-cluster
    variance — and only consider K's from there onward for the Silhouette
    comparison. That candidate is trusted only if its own Silhouette Score
    clears MIN_MEANINGFUL_SILHOUETTE (real structure, not a weak/artificial
    split); otherwise we fall back to the plain best-score rule across the
    whole range, since the WCSS elbow can't be at K_min itself (it needs a
    neighbor on both sides) and a weak elbow candidate is worse than trusting
    Silhouette outright.

    This does not — and, on data with real ambiguity, no purely statistical
    rule can — guarantee finding "the" true K. A dataset that truly has 2
    very well-separated top-level groups *and* 4+ finer sub-groups doesn't
    have one unambiguous right answer; this only fixes the specific
    "coarse split shadows the finer real one" failure pattern, verified
    against known-K synthetic data before shipping (see chat for the test).
    """
    table = elbow_table.sort_values("K").reset_index(drop=True)

    elbow_k = find_wcss_elbow_k(table)
    candidates = table[table["K"] >= elbow_k]
    best_candidate_score = candidates["Silhouette Score"].max()
    near_best_candidates = candidates[candidates["Silhouette Score"] >= best_candidate_score - SILHOUETTE_TOLERANCE]
    elbow_pick_k = int(near_best_candidates["K"].min())
    elbow_pick_score = table.loc[table["K"] == elbow_pick_k, "Silhouette Score"].iloc[0]

    if elbow_pick_score >= MIN_MEANINGFUL_SILHOUETTE:
        return elbow_pick_k

    best_score = table["Silhouette Score"].max()
    near_best = table[table["Silhouette Score"] >= best_score - SILHOUETTE_TOLERANCE]
    return int(near_best["K"].min())


def themed_matplotlib_figure(figsize):
    """A matplotlib figure/axes pair painted to match the current app palette
    (unlike st.dataframe, a plot we draw ourselves can fully follow the
    Dark/Light toggle)."""
    sns.set_theme(style="darkgrid" if st.session_state.dark_mode else "whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_P["bg"])
    ax.set_facecolor(_P["bg"])
    ax.tick_params(colors=_P["text"])
    ax.grid(color=_P["grid"])
    for spine in ax.spines.values():
        spine.set_color(_P["grid"])
    return fig, ax


# --------------------------------------------------------------------------
# Custom K-range slider component
# --------------------------------------------------------------------------
# Streamlit's own built-in st.slider has a confirmed rendering bug in this
# version (1.63.0): the handle's visual position is mirrored on the track
# (verified via direct DOM inspection — the reported value/label is correct,
# only the pixel position is wrong), and a pure-CSS attempt to fix that
# rendering also breaks the value-from-click/drag mapping (clicking a given
# spot then sets the mirrored value instead). This is a small hand-built
# Streamlit Component (two native <input type="range"> elements, talking to
# Python over Streamlit's documented postMessage component protocol) that
# sidesteps the buggy built-in widget entirely — verified end-to-end
# (drag-to-value mapping, min/max crossing prevention, and the Python-side
# round trip) before wiring it in here.
_K_RANGE_SLIDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "k_range_slider")
_k_range_slider_component = components.declare_component("k_range_slider", path=_K_RANGE_SLIDER_DIR)


def k_range_slider(min_value: int, max_value: int, default_min: int, default_max: int, key: str = None):
    result = _k_range_slider_component(
        min_value=min_value,
        max_value=max_value,
        default_min=default_min,
        default_max=default_max,
        dark_mode=st.session_state.dark_mode,
        key=key,
        default={"min": default_min, "max": default_max},
    )
    return result["min"], result["max"]


def ask_llm(prompt: str, max_retries: int = 4) -> str:
    """Call the Ollama cloud chat API. Raises on failure (caller handles it).

    Retries with growing backoff on HTTP 429 (rate limit) — running several
    clusters concurrently can burst past Ollama's per-second request limit,
    and a 429 there is transient, not a real failure.
    """
    api_key = st.secrets.get("OLLAMA_API_KEY", None) or os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Ollama API key configured. Add OLLAMA_API_KEY to "
            ".streamlit/secrets.toml (locally) or your host's secrets settings."
        )
    delay = 1.5
    for attempt in range(max_retries):
        response = requests.post(
            "https://ollama.com/api/chat",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-oss:120b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        if response.status_code == 429 and attempt < max_retries - 1:
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= 2
            continue
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


def generate_cluster_copy(profile_text: str) -> tuple[str, str]:
    """One combined LLM call for both the name and description, instead of
    two separate calls — halves the number of (slow) requests, which matters
    a lot once combined with running all clusters concurrently below."""
    prompt = (
        "Based on this customer segment's average feature values:\n"
        f"{profile_text}\n\n"
        "Reply with EXACTLY two lines and nothing else, no extra commentary:\n"
        "NAME: a short marketing name (1-3 words)\n"
        "DESCRIPTION: a one-sentence description (don't use the word 'Cluster')"
    )
    response = ask_llm(prompt)
    name, description = "Unnamed segment", response
    for line in response.splitlines():
        # Strip markdown noise models sometimes add (**bold**, bullets) so
        # "**NAME:** Foo" still matches the plain "NAME:" prefix.
        line = line.strip().strip("*#-• ").strip()
        if line.upper().startswith("NAME:"):
            name = line.split(":", 1)[1].strip().strip("*").strip()
        elif line.upper().startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip().strip("*").strip()
    return name, description


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
defaults = {
    "df": None,
    "original_filename": None,
    "remove_outliers": False,
    "df_normalized": None,
    "elbow_table": None,
    "k_min": 2,
    "k_max": 8,
    "chosen_k": None,
    "df_describe": None,
    "df_mean": None,
    "df_working": None,  # original (ID-checked) df used for the mean/profile table
    "current_step": 1,
    "uploaded_file_signature": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

STEP_LABELS = [
    "1️⃣ Upload Data",
    "2️⃣ Elbow Analysis",
    "3️⃣ Create Clusters",
    "4️⃣ Name Segments (LLM)",
    "5️⃣ Export",
]

nav_cols = st.columns(5)
for i, nav_col in enumerate(nav_cols, start=1):
    with nav_col:
        if st.button(
            STEP_LABELS[i - 1],
            key=f"nav_step_{i}",
            width="stretch",
            type="primary" if st.session_state.current_step == i else "secondary",
        ):
            st.session_state.current_step = i
            st.rerun()
st.divider()


def go_to_step(step: int):
    st.session_state.current_step = step
    st.rerun()


def step_nav_buttons(back_step: int = None, next_step: int = None, next_enabled: bool = True, next_label: str = "Next Step ➡️"):
    """Render Back/Next buttons at the bottom of a step."""
    st.write("")
    col1, col2 = st.columns([1, 1])
    with col1:
        if back_step is not None:
            if st.button("⬅️ Back", key=f"back_to_{back_step}", width="stretch"):
                go_to_step(back_step)
    with col2:
        if next_step is not None:
            if st.button(next_label, key=f"next_to_{next_step}", type="primary", width="stretch", disabled=not next_enabled):
                go_to_step(next_step)


# --------------------------------------------------------------------------
# STEP 1 — Upload & Preview
# --------------------------------------------------------------------------
if st.session_state.current_step == 1:
    st.markdown('<span class="ss-badge">STEP 1</span>', unsafe_allow_html=True)
    st.subheader("Upload & preview your data")

    uploaded_file = st.file_uploader(
        "Drop a CSV file here (up to 200MB)", type=["csv"]
    )

    if uploaded_file is not None:
        # Streamlit re-runs this whole script on every interaction anywhere in the
        # app (e.g. clicking a button in Step 3), and `uploaded_file` keeps
        # returning the same file across those re-runs. Only (re)load the CSV and
        # reset the downstream steps when the uploaded file has actually changed —
        # otherwise Step 2/3 results would get wiped out on every unrelated click.
        file_signature = getattr(uploaded_file, "file_id", None) or (uploaded_file.name, uploaded_file.size)
        if file_signature != st.session_state.uploaded_file_signature:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.original_filename = uploaded_file.name
                st.session_state.uploaded_file_signature = file_signature
                # Reset downstream state only because this is a genuinely new file
                for key in ("df_normalized", "elbow_table", "chosen_k", "df_describe", "df_mean", "df_working"):
                    st.session_state[key] = defaults[key]
            except Exception as e:
                st.error(f"Could not read this CSV: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Numeric columns", f"{df.select_dtypes(include=np.number).shape[1]:,}")

        # st.table is plain HTML, so it actually follows the Dark/Light
        # toggle (unlike st.dataframe's canvas-rendered grid). A preview
        # doesn't need every row, so it's capped to the first/last 30.
        EDGE_ROWS = 30
        if len(df) > EDGE_ROWS * 2:
            ellipsis_row = pd.DataFrame([["⋯"] * df.shape[1]], columns=df.columns, index=["⋯"])
            preview = pd.concat([df.head(EDGE_ROWS), ellipsis_row, df.tail(EDGE_ROWS)])
            st.table(preview)
            st.caption(f"Showing the first and last {EDGE_ROWS} of {len(df):,} rows.")
        else:
            st.table(df)
    else:
        st.info("Upload a CSV file to begin.")

    step_nav_buttons(back_step=None, next_step=2, next_enabled=st.session_state.df is not None)

# --------------------------------------------------------------------------
# STEP 2 — Elbow / Silhouette Analysis
# --------------------------------------------------------------------------
if st.session_state.current_step == 2:
    st.markdown('<span class="ss-badge">STEP 2</span>', unsafe_allow_html=True)
    st.subheader("Find the right number of clusters")

    if st.session_state.df is None:
        st.info("Upload a CSV in Step 1 first.")
    else:
        max_possible = max(3, len(st.session_state.df) - 1)
        col1, col2 = st.columns([1.6, 1.4])
        with col1:
            k_min, k_max = k_range_slider(
                min_value=2,
                max_value=max_possible,
                default_min=min(st.session_state.k_min, max_possible - 1),
                default_max=max(min(st.session_state.k_max, max_possible), st.session_state.k_min + 1),
                key="k_range",
            )
        with col2:
            st.session_state.remove_outliers = st.checkbox(
                "Remove outliers first (IQR, optional bonus)",
                value=st.session_state.remove_outliers,
            )
            run_clicked = st.button("🚀 Run Analysis", type="primary", width="stretch")

        if run_clicked:
            with st.spinner("Normalizing data and running K-Means..."):
                working_df = st.session_state.df.copy()
                if st.session_state.remove_outliers:
                    working_df = remove_outliers_iqr(working_df)

                df_normalized = norma_csv(working_df)
                st.session_state.df_normalized = df_normalized
                st.session_state.df_working = drop_leading_id_column(working_df.dropna())
                st.session_state.k_min, st.session_state.k_max = int(k_min), int(k_max)

                rows = []
                for k in range(int(k_min), int(k_max) + 1):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(df_normalized)
                    score = silhouette_score(df_normalized, km.labels_)
                    rows.append({"K": k, "WCSS": km.inertia_, "Silhouette Score": score})
                st.session_state.elbow_table = pd.DataFrame(rows)

        if st.session_state.elbow_table is not None:
            table = st.session_state.elbow_table
            if int(k_min) != st.session_state.k_min or int(k_max) != st.session_state.k_max:
                st.warning(
                    f"⚠️ You changed the K range to {int(k_min)}–{int(k_max)}, but the table and "
                    f"recommendation below still reflect the last run ({st.session_state.k_min}–"
                    f"{st.session_state.k_max}). Click **Run Analysis** to refresh them."
                )
            left, right = st.columns([1, 1.3])
            with left:
                st.table(table.style.format({"WCSS": "{:.1f}", "Silhouette Score": "{:.4f}"}))
                best_k = int(table.loc[table["Silhouette Score"].idxmax(), "K"])
                # Same value "Auto-select K" will pick in Step 3 — shown here
                # too so the two steps never contradict each other.
                suggested_k = auto_select_k(table)
                st.success(f"✅ Recommended K = {suggested_k} — this is what **Auto-select K** picks in Step 3.")
                if best_k == int(table["K"].max()):
                    st.warning(
                        f"⚠️ The raw top score is at K = {best_k}, right at your **Max K boundary** — it "
                        "was still climbing when the range ended, so a better K may exist beyond it. "
                        "Raise Max K and re-run to check."
                    )
                elif suggested_k != best_k:
                    st.caption(
                        f"K = {best_k} has the single highest Silhouette Score, but K = {suggested_k} "
                        "is recommended instead — it's past the point where adding clusters stops "
                        "meaningfully reducing WCSS (the elbow) while still scoring well, which a raw "
                        "top-score pick can miss on data with coarser sub-groupings (e.g. gender) "
                        "sitting on top of the real, finer segments."
                    )
            with right:
                fig, ax = themed_matplotlib_figure((6.5, 4.5))
                ax.plot(table["K"], table["WCSS"], "o-", linewidth=2.5, markersize=8, color=_P["accent"])
                ax.set_xticks(table["K"])
                ax.set_title("The Elbow Method for Optimal K", fontsize=13, fontweight="bold", color=_P["text"])
                ax.set_xlabel("Number of Clusters (K)", color=_P["text"])
                ax.set_ylabel("WCSS (Within-Cluster Sum of Squares)", color=_P["text"])
                st.pyplot(fig, width="stretch")
        else:
            st.info("Set a K range and click **Run Analysis**.")

    step_nav_buttons(back_step=1, next_step=3, next_enabled=st.session_state.elbow_table is not None)

# --------------------------------------------------------------------------
# STEP 3 — Create Clusters
# --------------------------------------------------------------------------
if st.session_state.current_step == 3:
    st.markdown('<span class="ss-badge">STEP 3</span>', unsafe_allow_html=True)
    st.subheader("Create your clusters")

    if st.session_state.df_normalized is None:
        st.info("Run the elbow analysis in Step 2 first.")
    else:
        k_range = list(range(st.session_state.k_min, st.session_state.k_max + 1))
        default_k = st.session_state.chosen_k or k_range[0]

        col1, col2, col3 = st.columns([1, 1, 1.4])
        with col1:
            chosen_k = st.selectbox("Final number of clusters (K)", k_range,
                                     index=k_range.index(default_k) if default_k in k_range else 0)
        with col2:
            if st.button("✨ Auto-select K", width="stretch"):
                chosen_k = auto_select_k(st.session_state.elbow_table)
                st.session_state.chosen_k = chosen_k
                st.rerun()
        with col3:
            create_clicked = st.button("🧩 Create Clusters", type="primary", width="stretch")

        st.session_state.chosen_k = chosen_k

        if create_clicked:
            with st.spinner("Running K-Means and building the cluster profile..."):
                df_normalized = st.session_state.df_normalized
                df_working = st.session_state.df_working

                kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(df_normalized)

                df_normalized = df_normalized.copy()
                df_normalized["cluster"] = labels
                df_working = df_working.copy()
                df_working["cluster"] = pd.Series(labels, index=df_normalized.index)

                df_describe = (
                    df_normalized.groupby("cluster")["cluster"]
                    .count()
                    .reset_index(name="count")
                    .rename(columns={"cluster": "cluster_id"})
                )
                df_describe["name"] = ""
                df_describe["description"] = ""

                df_mean = compute_cluster_profile(df_working)

                st.session_state.df_normalized = df_normalized
                st.session_state.df_working = df_working
                st.session_state.df_describe = df_describe
                st.session_state.df_mean = df_mean

        if st.session_state.df_describe is not None:
            st.markdown("**Cluster summary**")
            st.table(st.session_state.df_describe)
        else:
            st.info("Choose a K and click **Create Clusters**.")

    step_nav_buttons(back_step=2, next_step=4, next_enabled=st.session_state.df_describe is not None)

# --------------------------------------------------------------------------
# STEP 4 — LLM Interpretation
# --------------------------------------------------------------------------
if st.session_state.current_step == 4:
    st.markdown('<span class="ss-badge">STEP 4</span>', unsafe_allow_html=True)
    st.subheader("Let an LLM name your segments")

    if st.session_state.df_describe is None:
        st.info("Create clusters in Step 3 first.")
    else:
        # The generate button is the main call-to-action of this step, so it
        # comes first — front and center — rather than buried under details.
        st.markdown("👉 **This step is required** — Step 5 (Export) unlocks only after you generate names here.")
        generate_clicked = st.button(
            "🤖 Generate segment names with LLM", type="primary", width="stretch"
        )

        # A single placeholder we update in place, so the table is only ever
        # shown once — as the initial (unnamed) state, then live-updated
        # during generation, ending on the final named state.
        table_placeholder = st.empty()
        table_placeholder.table(st.session_state.df_describe)

        if generate_clicked and not (st.secrets.get("OLLAMA_API_KEY", None) or os.environ.get("OLLAMA_API_KEY")):
            # Check once up front — otherwise every one of the N concurrent
            # calls below fails independently and prints its own near-
            # identical error.
            st.error(
                "No Ollama API key configured. Add OLLAMA_API_KEY to "
                ".streamlit/secrets.toml (locally) or your host's secrets settings."
            )
        elif generate_clicked:
            df_describe = st.session_state.df_describe.copy()
            df_mean = st.session_state.df_mean
            status = st.status("Naming clusters (running in parallel)...", expanded=True)

            # One combined name+description call per cluster (see
            # generate_cluster_copy), fired concurrently across all clusters
            # instead of one-by-one — both changes cut the LLM's total wait
            # time down drastically compared to 2 sequential calls/cluster.
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(df_describe))) as executor:
                future_to_id = {
                    executor.submit(generate_cluster_copy, df_mean.loc[[row["cluster_id"]]].to_string()): row["cluster_id"]
                    for _, row in df_describe.iterrows()
                }
                done = 0
                for future in concurrent.futures.as_completed(future_to_id):
                    cluster_id = future_to_id[future]
                    done += 1
                    try:
                        name, description = future.result()
                    except Exception as e:
                        name, description = "Error", str(e)
                        st.error(f"Cluster {cluster_id}: could not generate a name ({e})")
                    df_describe.loc[df_describe["cluster_id"] == cluster_id, "name"] = name
                    df_describe.loc[df_describe["cluster_id"] == cluster_id, "description"] = description
                    status.write(f"Named cluster {cluster_id} ({done}/{len(df_describe)})")
                    table_placeholder.table(df_describe)

            status.update(label="Done naming clusters", state="complete")
            st.session_state.df_describe = df_describe

        names_generated = (st.session_state.df_describe["name"] != "").all()
        if not names_generated:
            st.warning("⚠️ Names haven't been generated yet — click the button above before moving on.")

        with st.expander("🔍 What the AI sees (feature averages per cluster)"):
            st.caption(
                "For each cluster: the mean of numeric features and the most common value of "
                "categorical ones. This is the only thing sent to the LLM to write the name and "
                "description above."
            )
            st.table(st.session_state.df_mean)

    step_nav_buttons(
        back_step=3,
        next_step=5,
        next_enabled=st.session_state.df_describe is not None and names_generated,
    )

# --------------------------------------------------------------------------
# STEP 5 — Export
# --------------------------------------------------------------------------
if st.session_state.current_step == 5:
    st.markdown('<span class="ss-badge">STEP 5</span>', unsafe_allow_html=True)
    st.subheader("Export your clustered data")

    if st.session_state.df_describe is None:
        st.info("Create clusters (and optionally name them) before exporting.")
    else:
        # Final deliverable: exactly the 4 required columns, nothing merged
        # back onto the original per-row CSV.
        export_df = st.session_state.df_describe[["cluster_id", "count", "name", "description"]]

        st.table(export_df)

        original_stem = os.path.splitext(st.session_state.original_filename or "data.csv")[0]
        out_filename = f"{original_stem}_clustered.csv"
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "⬇️ Download clustered CSV",
            data=csv_bytes,
            file_name=out_filename,
            mime="text/csv",
            width="stretch",
        )

    step_nav_buttons(back_step=4, next_step=None)
