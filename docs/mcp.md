# MCP server

scPyviewer ships an **MCP (Model Context Protocol) server** so an LLM agent —
Claude Code, Claude Desktop, or any MCP client — can explore single-cell data
directly: discover datasets, search genes, pull marker / composition tables, and
render publication-quality figures (returned as PNG images).

## Install & register

```{code-block} bash
pip install 'scPyviewer[mcp]'
```

**Claude Code**

```{code-block} bash
claude mcp add scpyviewer -e SCPYVIEWER_DATA_DIR=/path/to/data -- scpyviewer-mcp
```

**Claude Desktop** — add to `claude_desktop_config.json`:

```{code-block} json
{
  "mcpServers": {
    "scpyviewer": {
      "command": "scpyviewer-mcp",
      "env": { "SCPYVIEWER_DATA_DIR": "/path/to/data" }
    }
  }
}
```

The server speaks stdio. Point `SCPYVIEWER_DATA_DIR` at a folder of prepared
`.h5ad` files (run `scpyviewer-prepare`, with `--subsample` for huge atlases —
see [Large data](large_data)).

## Tools

| Tool | Returns | Purpose |
|---|---|---|
| `list_datasets` | list | prepared `.h5ad` files in the data dir |
| `dataset_info` | dict | cells/genes, `group_key`, embeddings, metadata fields |
| `search_genes` | list | gene-name search |
| `set_style` | — | switch preset (bold/nature) + palette (npg/aaas/nejm/lancet/jama/…) |
| `top_markers` | table | top marker/DE genes per group |
| `composition` | table | group × split composition (fractions) |
| `plot_embedding` | PNG | UMAP/t-SNE/PCA by metadata or gene |
| `plot_dotplot` | PNG | mean expr × fraction expressing |
| `plot_violin` | PNG | gene expression by group |
| `plot_heatmap` | PNG | genes × cells, grouped |
| `plot_bar` | PNG | mean ± SEM per group |
| `plot_composition` | PNG | stacked composition bars |

All figures use the bold figures4papers-style, colour-blind-safe
{func}`~scPyviewer.set_publication_style` defaults.

## Typical flow

`list_datasets` → `dataset_info` (learn `group_key` / fields) →
`search_genes` → `plot_embedding` / `plot_dotplot` / `top_markers`.
