# Segment Studio

**🔗 Live app:** https://segment-studio---project-2-ck8h7zyu5scytsxtt4ia58.streamlit.app/

Upload any CSV, run K-Means segmentation with an elbow/silhouette analysis,
auto-name the resulting segments with an LLM, and download the labeled file.

## Run locally

```bash
cd project2_Segment_Studio
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your real OLLAMA_API_KEY
streamlit run app.py
```

Step 4 (LLM naming) requires a valid `OLLAMA_API_KEY` (from https://ollama.com).
Steps 1-3 and 5 work without it.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (`.streamlit/secrets.toml` is gitignored — don't commit it).
2. Create a new app on https://share.streamlit.io pointing at `app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```
   OLLAMA_API_KEY = "your_real_key"
   ```

## Deploy on Render

1. Create a new **Web Service** from this repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add an environment variable `OLLAMA_API_KEY` with your real key
   (the app reads `st.secrets` first and falls back to the environment variable).

## Notes

- Max upload size is set to 200MB in `.streamlit/config.toml`. Some hosts cap
  request size below this at the proxy level — check your host's docs if a
  large upload fails.
- Rows with missing values are dropped during normalization and therefore
  excluded from the final exported CSV (they can't be assigned a cluster).
- The Min K / Max K sliders in Step 2 are a small hand-built Streamlit
  component (no extra pip package) living in `components/k_range_slider/`.
  It's a static `index.html`, not code that gets imported — make sure that
  folder is committed and deployed alongside `app.py`, or the sliders won't
  render.
