# Segment Studio

Automatic customer segmentation, end to end: upload a CSV, discover natural
customer segments with K-Means, and let an LLM name and describe each one —
no code required from the user.

This repo has two parts:

- **[`project2_Segment_Studio.ipynb`](project2_Segment_Studio.ipynb)** — the
  original notebook prototype: normalization, the elbow/silhouette analysis,
  K-Means clustering, and the LLM naming step, run manually cell by cell.
- **[`project2_Segment_Studio/`](project2_Segment_Studio)** — the same logic
  turned into a full interactive **Streamlit web app**, so anyone can run the
  whole pipeline through a browser instead of editing a notebook.

## What the app does

1. **Upload & preview** any CSV.
2. **Elbow analysis** — runs K-Means across a chosen K range and reports
   WCSS + Silhouette Score for each K, with a plotted elbow curve.
3. **Create clusters** — fits the final K-Means model (with an "Auto-select
   K" option) and builds a per-cluster summary.
4. **Name the segments** — sends each cluster's feature profile to an LLM
   (via Ollama) to generate a short marketing name and description.
5. **Export** — download the final `cluster_id / count / name / description`
   table as CSV.

## Quick start

```bash
cd project2_Segment_Studio
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and add your own OLLAMA_API_KEY
streamlit run app.py
```

Full setup, deployment (Streamlit Community Cloud / Render), and
troubleshooting notes are in
[`project2_Segment_Studio/README.md`](project2_Segment_Studio/README.md).

## Tech stack

Python, [Streamlit](https://streamlit.io), pandas, scikit-learn (K-Means,
Silhouette Score), Matplotlib/Seaborn, and an LLM served through
[Ollama](https://ollama.com) for the segment naming step.
