"""Tests for scPyviewer.api — all public plotting, table, and export functions."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import scPyviewer as sv
from scPyviewer.api import _resolve_embedding, _group, _emb_label


# ================================================================== Dataset
class TestDataset:
    def test_repr(self, toy_dataset):
        r = repr(toy_dataset)
        assert "toy.prepared.h5ad" in r
        assert "cells" in r

    def test_n_obs_n_vars(self, toy_dataset):
        assert toy_dataset.n_obs == 100
        assert toy_dataset.n_vars == 50

    def test_genes_empty_query(self, toy_dataset):
        genes = toy_dataset.genes()
        assert len(genes) == 50
        assert all(isinstance(g, str) for g in genes)

    def test_genes_prefix_search(self, toy_dataset):
        hits = toy_dataset.genes("Gene00")
        assert len(hits) > 0
        assert all("Gene00" in h for h in hits)

    def test_genes_limit(self, toy_dataset):
        hits = toy_dataset.genes("Gene", limit=5)
        assert len(hits) <= 5

    def test_group_key(self, toy_dataset):
        assert toy_dataset.group_key == "cell_type"

    def test_embeddings(self, toy_dataset):
        assert "X_umap" in toy_dataset.embeddings
        assert "X_tsne" in toy_dataset.embeddings

    def test_categorical(self, toy_dataset):
        assert "cell_type" in toy_dataset.categorical
        assert "batch" in toy_dataset.categorical


# ================================================================== Helpers
class TestHelpers:
    def test_resolve_embedding_umap_default(self, toy_dataset):
        assert _resolve_embedding(toy_dataset, None) == "X_umap"

    def test_resolve_embedding_explicit(self, toy_dataset):
        assert _resolve_embedding(toy_dataset, "X_tsne") == "X_tsne"

    def test_resolve_embedding_missing_falls_back(self, toy_dataset):
        result = _resolve_embedding(toy_dataset, "X_nonexistent")
        assert result in toy_dataset.embeddings

    def test_group_default(self, toy_dataset):
        assert _group(toy_dataset, None) == "cell_type"

    def test_group_override(self, toy_dataset):
        assert _group(toy_dataset, "batch") == "batch"

    def test_emb_label(self):
        assert _emb_label("X_umap") == "UMAP"
        assert _emb_label("X_pca") == "PCA"


# ================================================================== plot_embedding
class TestPlotEmbedding:
    def test_default_color_by_group(self, toy_dataset):
        fig = sv.plot_embedding(toy_dataset)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_by_explicit_field(self, toy_dataset):
        fig = sv.plot_embedding(toy_dataset, color="batch")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_color_by_gene(self, toy_dataset):
        gene = toy_dataset.adata.var_names[0]
        fig = sv.plot_embedding(toy_dataset, gene=gene)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_embedding(self, toy_dataset):
        fig = sv.plot_embedding(toy_dataset, embedding="X_tsne")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_point_size(self, toy_dataset):
        fig = sv.plot_embedding(toy_dataset, point_size=10.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figsize(self, toy_dataset):
        fig = sv.plot_embedding(toy_dataset, figsize=(4.0, 3.0))
        assert fig.get_size_inches()[0] == pytest.approx(4.0, abs=0.1)
        plt.close(fig)

    def test_no_label_groups(self, toy_dataset):
        fig = sv.plot_embedding(toy_dataset, label_groups=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_cmap(self, toy_dataset):
        gene = toy_dataset.adata.var_names[0]
        fig = sv.plot_embedding(toy_dataset, gene=gene, cmap="plasma")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_alpha_param(self, toy_dataset):
        fig = sv.plot_embedding(toy_dataset, alpha=0.5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self, toy_dataset):
        fig = sv.plot_embedding(toy_dataset, title="My Plot")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_dpi_param(self, toy_dataset):
        fig = sv.plot_embedding(toy_dataset, dpi=72)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_hide_legend(self, toy_dataset):
        fig = sv.plot_embedding(toy_dataset, label_groups=False, show_legend=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ================================================================== plot_multigene
class TestPlotMultigene:
    def test_basic(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:3])
        fig = sv.plot_multigene(toy_dataset, genes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_gene(self, toy_dataset):
        genes = [toy_dataset.adata.var_names[0]]
        fig = sv.plot_multigene(toy_dataset, genes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ncol_param(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:4])
        fig = sv.plot_multigene(toy_dataset, genes, ncol=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_max_genes_cap(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:20])
        fig = sv.plot_multigene(toy_dataset, genes, max_genes=4)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_cmap(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:2])
        fig = sv.plot_multigene(toy_dataset, genes, cmap="magma")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_alpha_param(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:2])
        fig = sv.plot_multigene(toy_dataset, genes, alpha=0.5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_genes_filtered(self, toy_dataset):
        genes = [toy_dataset.adata.var_names[0], "NOT_A_GENE"]
        fig = sv.plot_multigene(toy_dataset, genes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_missing_genes_raises(self, toy_dataset):
        with pytest.raises(ValueError, match="none of the requested"):
            sv.plot_multigene(toy_dataset, ["FAKE1", "FAKE2"])

    def test_custom_embedding(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:2])
        fig = sv.plot_multigene(toy_dataset, genes, embedding="X_tsne")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ================================================================== plot_violin
class TestPlotViolin:
    def test_violin_default(self, toy_dataset):
        gene = toy_dataset.adata.var_names[0]
        fig = sv.plot_violin(toy_dataset, gene)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_box_kind(self, toy_dataset):
        gene = toy_dataset.adata.var_names[0]
        fig = sv.plot_violin(toy_dataset, gene, kind="box")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_show_points(self, toy_dataset):
        gene = toy_dataset.adata.var_names[0]
        fig = sv.plot_violin(toy_dataset, gene, show_points=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_group(self, toy_dataset):
        gene = toy_dataset.adata.var_names[0]
        fig = sv.plot_violin(toy_dataset, gene, group="batch")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_rotation(self, toy_dataset):
        gene = toy_dataset.adata.var_names[0]
        fig = sv.plot_violin(toy_dataset, gene, rotation=45)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_palette_override(self, toy_dataset):
        gene = toy_dataset.adata.var_names[0]
        palette = {"TypeA": "#ff0000", "TypeB": "#00ff00", "TypeC": "#0000ff"}
        fig = sv.plot_violin(toy_dataset, gene, palette=palette)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_kind_raises(self, toy_dataset):
        gene = toy_dataset.adata.var_names[0]
        with pytest.raises(ValueError, match="kind must be"):
            sv.plot_violin(toy_dataset, gene, kind="swarm")

    def test_figsize(self, toy_dataset):
        gene = toy_dataset.adata.var_names[0]
        fig = sv.plot_violin(toy_dataset, gene, figsize=(5.0, 3.0))
        assert fig.get_size_inches()[0] == pytest.approx(5.0, abs=0.1)
        plt.close(fig)


# ================================================================== plot_dotplot
class TestPlotDotplot:
    def test_basic(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:4])
        fig = sv.plot_dotplot(toy_dataset, genes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_group(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:3])
        fig = sv.plot_dotplot(toy_dataset, genes, group="batch")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_cmap(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:3])
        fig = sv.plot_dotplot(toy_dataset, genes, cmap="RdBu_r")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_size_scale(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:3])
        fig = sv.plot_dotplot(toy_dataset, genes, size_scale=300)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_standard_scale_var(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:4])
        fig = sv.plot_dotplot(toy_dataset, genes, standard_scale="var")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_standard_scale_group(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:4])
        fig = sv.plot_dotplot(toy_dataset, genes, standard_scale="group")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_standard_scale_raises(self, toy_dataset):
        genes = list(toy_dataset.adata.var_names[:3])
        with pytest.raises(ValueError, match="standard_scale"):
            sv.plot_dotplot(toy_dataset, genes, standard_scale="bad")

    def test_all_missing_genes_raises(self, toy_dataset):
        with pytest.raises(ValueError, match="none of the requested"):
            sv.plot_dotplot(toy_dataset, ["FAKE1"])


# ================================================================== plot_composition
class TestPlotComposition:
    def test_default(self, toy_dataset):
        fig = sv.plot_composition(toy_dataset)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_normalize_false(self, toy_dataset):
        fig = sv.plot_composition(toy_dataset, normalize=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_explicit_group_split(self, toy_dataset):
        fig = sv.plot_composition(toy_dataset, group="cell_type", split="batch")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, toy_dataset):
        fig = sv.plot_composition(toy_dataset, figsize=(6.0, 3.0))
        assert fig.get_size_inches()[0] == pytest.approx(6.0, abs=0.1)
        plt.close(fig)

    def test_palette_override(self, toy_dataset):
        palette = {"TypeA": "#cc0000", "TypeB": "#00cc00", "TypeC": "#0000cc"}
        fig = sv.plot_composition(toy_dataset, palette=palette)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_bar_width(self, toy_dataset):
        fig = sv.plot_composition(toy_dataset, bar_width=0.5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_sort_groups(self, toy_dataset):
        fig = sv.plot_composition(toy_dataset, sort_groups=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ================================================================== tables
class TestTables:
    def test_markers_table_returns_dataframe(self, toy_dataset):
        df = sv.markers_table(toy_dataset)
        assert isinstance(df, pd.DataFrame)
        assert "gene" in df.columns
        assert "group" in df.columns
        assert "rank" in df.columns

    def test_markers_table_group_filter(self, toy_dataset):
        df = sv.markers_table(toy_dataset, group="TypeA")
        assert set(df["group"].unique()) == {"TypeA"}

    def test_markers_table_top_n(self, toy_dataset):
        df = sv.markers_table(toy_dataset, top_n=3)
        assert df.groupby("group").size().max() <= 3

    def test_markers_table_sort_by_logfoldchange(self, toy_dataset):
        df = sv.markers_table(toy_dataset, sort_by="logfoldchange", ascending=False)
        for grp, sub in df.groupby("group"):
            lfcs = sub["logfoldchange"].values
            assert (lfcs[:-1] >= lfcs[1:]).all() or len(sub) == 1

    def test_markers_table_sort_by_pval_adj(self, toy_dataset):
        df = sv.markers_table(toy_dataset, sort_by="pval_adj")
        assert isinstance(df, pd.DataFrame)

    def test_markers_table_invalid_sort_still_returns(self, toy_dataset):
        df = sv.markers_table(toy_dataset, sort_by="nonexistent_col")
        assert isinstance(df, pd.DataFrame)

    def test_composition_table(self, toy_dataset):
        df = sv.composition_table(toy_dataset)
        assert isinstance(df, pd.DataFrame)

    def test_composition_table_normalize_false(self, toy_dataset):
        df = sv.composition_table(toy_dataset, normalize=False)
        assert isinstance(df, pd.DataFrame)
        # All values should be non-negative integers
        numeric_cols = [c for c in df.columns if c != "batch"]
        assert (df[numeric_cols].values >= 0).all()

    def test_metadata_table(self, toy_dataset):
        df = sv.metadata_table(toy_dataset)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "cell_type" in df.columns


# ================================================================== export
class TestExport:
    def test_export_figures_png(self, toy_dataset, tmp_path):
        written = sv.export_figures(toy_dataset, str(tmp_path / "figs"), formats=["png"])
        assert len(written) == 6
        assert all(p.endswith(".png") for p in written)
        assert all(os.path.exists(p) for p in written)

    def test_export_figures_svg(self, toy_dataset, tmp_path):
        written = sv.export_figures(toy_dataset, str(tmp_path / "svg_figs"), formats=["svg"])
        assert all(p.endswith(".svg") for p in written)
        assert all(os.path.exists(p) for p in written)

    def test_export_figures_multi_format(self, toy_dataset, tmp_path):
        written = sv.export_figures(toy_dataset, str(tmp_path / "multi"),
                                    formats=["png", "pdf"])
        assert len(written) == 12  # 6 plots × 2 formats

    def test_export_figures_custom_genes(self, toy_dataset, tmp_path):
        genes = list(toy_dataset.adata.var_names[:3])
        written = sv.export_figures(toy_dataset, str(tmp_path / "genes"),
                                    genes=genes, formats=["png"])
        assert len(written) == 6

    def test_export_figures_custom_dpi(self, toy_dataset, tmp_path):
        written = sv.export_figures(toy_dataset, str(tmp_path / "dpi"),
                                    formats=["png"], dpi=72)
        assert all(os.path.exists(p) for p in written)

    def test_export_tables_csv(self, toy_dataset, tmp_path):
        written = sv.export_tables(toy_dataset, str(tmp_path / "tables"))
        assert len(written) == 3
        assert all(p.endswith(".csv") for p in written)
        assert all(os.path.exists(p) for p in written)

    def test_export_tables_tsv(self, toy_dataset, tmp_path):
        written = sv.export_tables(toy_dataset, str(tmp_path / "tsv"),
                                   formats=["tsv"])
        assert all(p.endswith(".tsv") for p in written)

    def test_export_tables_top_n(self, toy_dataset, tmp_path):
        written = sv.export_tables(toy_dataset, str(tmp_path / "topn"),
                                   formats=["csv"], top_n=5)
        assert all(os.path.exists(p) for p in written)

    def test_export_tables_invalid_format_raises(self, toy_dataset, tmp_path):
        with pytest.raises(ValueError, match="unsupported table format"):
            sv.export_tables(toy_dataset, str(tmp_path / "bad"), formats=["parquet"])


# ================================================================== io_utils
class TestIoUtils:
    def test_discover_datasets(self, tmp_path):
        from scPyviewer import io_utils as io
        f = tmp_path / "test.prepared.h5ad"
        f.touch()
        result = io.discover_datasets(str(tmp_path))
        assert "test" in result

    def test_gene_search_prefix(self, toy_dataset):
        from scPyviewer import io_utils as io
        hits = io.gene_search(toy_dataset.adata, "Gene00", limit=10)
        assert len(hits) > 0
        assert all("Gene00" in h for h in hits)

    def test_gene_search_empty(self, toy_dataset):
        from scPyviewer import io_utils as io
        hits = io.gene_search(toy_dataset.adata, "", limit=5)
        assert len(hits) == 5

    def test_embedding_2d_shape(self, toy_dataset):
        from scPyviewer import io_utils as io
        xy = io.embedding_2d(toy_dataset.adata, "X_umap")
        assert xy.shape == (100, 2)

    def test_gene_vector_shape(self, toy_dataset):
        from scPyviewer import io_utils as io
        vec = io.gene_vector(toy_dataset.adata, toy_dataset.adata.var_names[0])
        assert vec.shape == (100,)
        assert vec.dtype == np.float32

    def test_gene_vector_missing_raises(self, toy_dataset):
        from scPyviewer import io_utils as io
        with pytest.raises(KeyError):
            io.gene_vector(toy_dataset.adata, "NONEXISTENT_GENE")

    def test_get_meta_has_keys(self, toy_dataset):
        from scPyviewer import io_utils as io
        meta = io.get_meta(toy_dataset.adata)
        assert "embeddings" in meta
        assert "group_key" in meta
        assert "schema" in meta

    def test_markers_df_returns_dataframe(self, toy_dataset):
        from scPyviewer import io_utils as io
        df = io.markers_df(toy_dataset.adata)
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert "gene" in df.columns

    def test_markers_df_none_when_absent(self):
        from scPyviewer import io_utils as io
        import anndata as ad
        adata = ad.AnnData(np.zeros((5, 3)))
        assert io.markers_df(adata) is None
