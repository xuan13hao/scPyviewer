---
name: scpyviewer
description: >-
  Explore single-cell AnnData (.h5ad) datasets — embeddings, gene expression,
  marker genes, and cell-type composition — through the scPyviewer MCP server.
  Use when the user wants to visualize, query, or make publication-quality
  figures from single-cell / scRNA-seq data.
---

# scPyviewer skill

scPyviewer exposes single-cell data exploration over MCP. Figures come back as
PNG images; markers and composition come back as text tables.

## Setup (once)

```bash
pip install 'scPyviewer[mcp]'
```

Register the server with your MCP client:

- **Claude Code**
  ```bash
  claude mcp add scpyviewer -e SCPYVIEWER_DATA_DIR=/path/to/data -- scpyviewer-mcp
  ```
- **Claude Desktop** — add to `claude_desktop_config.json`:
  ```json
  {
    "mcpServers": {
      "scpyviewer": {
        "command": "scpyviewer-mcp",
        "env": { "SCPYVIEWER_DATA_DIR": "/path/to/data" }
      }
    }
  }
  ```

Datasets must be prepared `.h5ad` files — run `scpyviewer-prepare raw.h5ad`
first (and `--subsample 50000` for atlases larger than RAM).

## Workflow

1. **Discover** — `list_datasets` (uses `$SCPYVIEWER_DATA_DIR`), or take an
   explicit path from the user.
2. **Orient** — `dataset_info(path)` to learn the default `group_key`, the
   available `embeddings`, and which obs columns are categorical (use these for
   `color` / `group`) vs numeric.
3. **Find genes** — `search_genes(path, query)` before plotting a gene.
4. **Visualize / query**:
   - `plot_embedding(path, color=…)` or `plot_embedding(path, gene=…)`
   - `plot_dotplot`, `plot_violin`, `plot_heatmap`, `plot_bar` for expression
   - `plot_composition(path, group=…, split=…)` for cell-type proportions
   - `top_markers(path, group=…)` / `composition(...)` for tables

Always pass a gene name that `search_genes` confirmed, and a `group`/`color`
that `dataset_info` listed as a categorical field.
