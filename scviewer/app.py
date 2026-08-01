#!/usr/bin/env python
"""
app.py — scviewer: a Python-native, Streamlit interactive viewer for AnnData
(.h5ad) single-cell objects.

Feature modules (map to the study feature-parity checklist):
  1. Embeddings + metadata coloring          -> "Embedding" tab
  2. Single/multi-gene expression overlay     -> "Embedding" + "Expression" tabs
  3. Violin/box grouped by metadata           -> "Expression" tab
  4. Marker gene / DE table browsing          -> "Markers / DE" tab
  5. Cross-dataset / cross-species comparison -> "Compare" tab
  6. No-code shareable deployment             -> this Streamlit app
  7. Native AnnData input (no Seurat)         -> loads .h5ad directly

Run:  streamlit run scviewer/app.py -- --data-dir data
"""
from __future__ import annotations

import argparse
import io as _io
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# allow running both as `streamlit run scviewer/app.py` and as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scviewer import io_utils as io  # noqa: E402
from scviewer import plots  # noqa: E402


# --------------------------------------------------------------------- config
def get_data_dir() -> str:
    # Streamlit passes app args after `--`
    argv = sys.argv[1:]
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("SCVIEWER_DATA_DIR", "data"))
    args, _ = ap.parse_known_args(argv)
    return args.data_dir


DATA_DIR = get_data_dir()
st.set_page_config(page_title="scviewer", layout="wide", page_icon="🔬")


@st.cache_resource(show_spinner="Loading AnnData…")
def cached_load(path: str):
    a = io.load_adata(path)
    return a, io.get_meta(a)


def apply_filters(adata, filters: dict):
    """Return a boolean mask over cells from the active metadata filters."""
    mask = np.ones(adata.n_obs, dtype=bool)
    for field, sel in filters.get("cat", {}).items():
        if sel:
            mask &= adata.obs[field].astype(str).isin(sel).values
    for field, (lo, hi) in filters.get("num", {}).items():
        v = adata.obs[field].values
        mask &= (v >= lo) & (v <= hi)
    return mask


# --------------------------------------------------------------------- sidebar
st.sidebar.title("🔬 scviewer")
st.sidebar.caption("Python-native AnnData explorer")

datasets = io.discover_datasets(DATA_DIR)
if not datasets:
    st.error(f"No .h5ad files found in `{DATA_DIR}`. "
             f"Run `scviewer/prepare.py` first, or set --data-dir.")
    st.stop()

ds_name = st.sidebar.selectbox("Dataset", list(datasets.keys()))
adata, meta = cached_load(datasets[ds_name])

cat_fields = meta["schema"]["categorical_obs"]
num_fields = meta["schema"]["numeric_obs"]
embeddings = meta["embeddings"]
group_key = meta["group_key"] or (cat_fields[0] if cat_fields else None)

st.sidebar.markdown(f"**{adata.n_obs:,}** cells × **{adata.n_vars:,}** genes")
st.sidebar.markdown(f"Embeddings: {', '.join(e.replace('X_', '').upper() for e in embeddings)}")

# --- metadata subsetting (module 4) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Subset cells")
filters = {"cat": {}, "num": {}}
for f in cat_fields:
    opts = sorted(adata.obs[f].astype(str).unique())
    if len(opts) <= 60:
        sel = st.sidebar.multiselect(f, opts, default=[], key=f"filt_{f}")
        filters["cat"][f] = sel
for f in num_fields:
    v = adata.obs[f].values.astype(float)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if lo < hi:
        r = st.sidebar.slider(f, lo, hi, (lo, hi), key=f"num_{f}")
        filters["num"][f] = r

mask = apply_filters(adata, filters)
n_sel = int(mask.sum())
if n_sel < adata.n_obs:
    st.sidebar.info(f"Filter active: {n_sel:,} / {adata.n_obs:,} cells")
# No .copy() — keeps the expression matrix on disk for backed datasets.
view = adata[mask] if n_sel < adata.n_obs else adata

# For datasets with many cells, subsample scatter plots for responsiveness.
MAX_SCATTER = int(os.environ.get("SCVIEWER_MAX_SCATTER", 200_000))


# --------------------------------------------------------------------- tabs
tab_emb, tab_expr, tab_markers, tab_compare, tab_export = st.tabs(
    ["Embedding", "Expression", "Markers / DE", "Compare", "Export"])

# ============================================================ 1. EMBEDDING
with tab_emb:
    st.subheader("Embedding")
    c1, c2, c3 = st.columns([1, 1, 1])
    emb = c1.selectbox("Embedding", embeddings,
                       index=embeddings.index("X_umap") if "X_umap" in embeddings else 0)
    mode = c2.radio("Color by", ["Metadata", "Gene expression"], horizontal=True)
    if mode == "Metadata":
        field = c3.selectbox("Field", cat_fields + num_fields,
                             index=(cat_fields + num_fields).index(group_key)
                             if group_key in (cat_fields + num_fields) else 0)
        fig = plots.embedding_scatter(view, emb, color_field=field, max_points=MAX_SCATTER)
    else:
        gene = c3.selectbox("Gene", io.gene_search(view, ""), key="emb_gene")
        gq = st.text_input("Search gene", "", key="emb_gene_search")
        if gq:
            hits = io.gene_search(view, gq)
            if hits:
                gene = st.selectbox("Matches", hits, key="emb_gene_hit")
        fig = plots.embedding_scatter(view, emb, gene=gene, max_points=MAX_SCATTER)
    st.plotly_chart(fig, width='stretch')

# ============================================================ 2. EXPRESSION
with tab_expr:
    st.subheader("Gene expression")
    default_genes = io.gene_search(view, "")[:3]
    genes = st.multiselect("Genes", options=list(view.var_names[:5000]),
                           default=default_genes, key="expr_genes",
                           help="Type to search across all genes")
    extra = st.text_input("Add gene by name", "", key="expr_add")
    if extra:
        for h in io.gene_search(view, extra)[:5]:
            if h not in genes:
                genes.append(h)

    if genes:
        colA, colB = st.columns(2)
        with colA:
            emb2 = st.selectbox("Overlay embedding", embeddings,
                                index=embeddings.index("X_umap") if "X_umap" in embeddings else 0,
                                key="expr_emb")
            st.plotly_chart(plots.multi_gene_grid(view, emb2, genes, ncol=2,
                                                   max_points=MAX_SCATTER),
                            width='stretch')
        with colB:
            gfield = st.selectbox("Group by", cat_fields,
                                  index=cat_fields.index(group_key) if group_key in cat_fields else 0,
                                  key="expr_group")
            kind = st.radio("Plot", ["violin", "box"], horizontal=True, key="expr_kind")
            st.plotly_chart(plots.violin(view, genes, gfield, kind=kind),
                            width='stretch')
        st.markdown("**Dotplot** — mean expression (color) × fraction expressing (size)")
        st.plotly_chart(plots.dotplot(view, genes, gfield), width='stretch')
    else:
        st.info("Select at least one gene.")

# ============================================================ 3. MARKERS / DE
with tab_markers:
    st.subheader("Marker genes / differential expression")
    mdf = io.markers_df(adata)
    if mdf is None:
        st.warning("No marker/DE table in this dataset. Re-run prepare.py to compute one.")
    else:
        groups = sorted(mdf["group"].unique())
        c1, c2, c3 = st.columns([1.2, 1, 1])
        grp = c1.selectbox("Group", groups, key="mk_group")
        topn = c2.slider("Top N", 5, 100, 25, key="mk_topn")
        gq = c3.text_input("Filter gene", "", key="mk_q")
        sub = mdf[mdf["group"] == grp].sort_values("rank").head(topn)
        if gq:
            sub = sub[sub["gene"].str.contains(gq, case=False)]
        st.dataframe(sub.reset_index(drop=True), width='stretch', height=420)

        pick = st.selectbox("Overlay a marker on the embedding", sub["gene"].tolist(),
                            key="mk_overlay")
        emb3 = st.selectbox("Embedding", embeddings,
                            index=embeddings.index("X_umap") if "X_umap" in embeddings else 0,
                            key="mk_emb")
        if pick:
            st.plotly_chart(plots.embedding_scatter(view, emb3, gene=pick,
                                                    max_points=MAX_SCATTER),
                            width='stretch')
        st.download_button("⬇ Download this DE table (CSV)",
                           sub.to_csv(index=False).encode(),
                           file_name=f"{ds_name}_{grp}_markers.csv", mime="text/csv")

# ============================================================ 4. COMPARE
with tab_compare:
    st.subheader("Cross-dataset / cross-sample comparison")
    st.caption("Compare a shared gene's expression, or a cell-type composition, "
               "across a second dataset or across a metadata split of this one.")
    others = [d for d in datasets if d != ds_name]
    mode_cmp = st.radio("Compare against",
                        ["Metadata split of this dataset"] +
                        (["Another dataset"] if others else []),
                        horizontal=True)

    if mode_cmp == "Metadata split of this dataset":
        split = st.selectbox("Split by", cat_fields, key="cmp_split")
        what = st.radio("Show", ["Cell-type composition", "Gene expression"],
                        horizontal=True, key="cmp_what")
        if what == "Cell-type composition" and group_key:
            st.plotly_chart(plots.composition_bar(adata, group_key, split),
                            width='stretch')
        else:
            g = st.selectbox("Gene", io.gene_search(adata, ""), key="cmp_gene1")
            gq = st.text_input("Search gene", "", key="cmp_gene1_q")
            if gq and io.gene_search(adata, gq):
                g = st.selectbox("Matches", io.gene_search(adata, gq), key="cmp_gene1_hit")
            st.plotly_chart(plots.violin(adata, [g], split), width='stretch')
    else:
        other = st.selectbox("Second dataset", others, key="cmp_other")
        a2, m2 = cached_load(datasets[other])
        shared = sorted(set(adata.var_names) & set(a2.var_names))
        st.caption(f"{len(shared):,} genes shared between the two datasets.")
        if shared:
            g = st.selectbox("Shared gene", shared[:2000], key="cmp_gene2")
            colL, colR = st.columns(2)
            gk1 = group_key
            gk2 = m2["group_key"]
            colL.plotly_chart(plots.violin(adata, [g], gk1) if gk1 else plots.violin(adata, [g], cat_fields[0]),
                              width='stretch')
            colR.plotly_chart(plots.violin(a2, [g], gk2) if gk2 else
                              plots.violin(a2, [g], m2["schema"]["categorical_obs"][0]),
                              width='stretch')

# ============================================================ 5. EXPORT
with tab_export:
    st.subheader("Export")
    st.markdown(f"Current view: **{view.n_obs:,}** cells "
                f"({'filtered' if n_sel < adata.n_obs else 'all cells'}).")
    obs_out = view.obs.copy()
    st.download_button("⬇ Download cell metadata (CSV)",
                       obs_out.to_csv().encode(),
                       file_name=f"{ds_name}_cell_metadata.csv", mime="text/csv")
    md = io.markers_df(adata)
    if md is not None:
        st.download_button("⬇ Download full marker/DE table (CSV)",
                           md.to_csv(index=False).encode(),
                           file_name=f"{ds_name}_all_markers.csv", mime="text/csv")
    st.caption("Figures can be exported to PNG via the camera icon on any plot.")
