#!/usr/bin/env Rscript
# bench_r.R -- Seurat rendering benchmark on the chicken-heart dataset.
#
# Times the *same* operations the Python harness (scviewer/benchmark.py) times,
# on the *same* data, against the R/Seurat rendering substrate that ShinyCell,
# ScRDAVis, sCIRCLE and scViewer all build on: DimPlot / FeaturePlot / VlnPlot /
# DotPlot + a ggplot stacked-composition bar. We benchmark the shared rendering
# engine (not a full Shiny server) so the numbers are directly comparable to the
# Python render timings, and we say so plainly in the paper.
#
# Emits results/benchmark_r.json with the same key names as the Python
# performance block, plus load timing and peak RSS.

suppressPackageStartupMessages({
  library(Seurat); library(SeuratObject); library(Matrix)
  library(ggplot2); library(jsonlite)
})
setwd(dirname(normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE)[1]))))

spec <- fromJSON("data/for_seurat/bench_spec.json")
group_key <- spec$group_key
gene1     <- spec$gene1
top3      <- spec$top3
top5      <- spec$top5
comp_cat  <- spec$composition_cat

rss_mb <- function() {
  # resident set size of this R process, in MB (Linux /proc)
  as.numeric(strsplit(readLines(sprintf("/proc/%d/status", Sys.getpid()))[
    grep("VmRSS", readLines(sprintf("/proc/%d/status", Sys.getpid())))],
    "\\s+")[[1]][2]) / 1024
}

# force a ggplot/patchwork object to actually compute its layout+stats,
# which is the real render cost (ggplot is lazy until built/printed).
force_render <- function(p) {
  tmp <- tempfile(fileext = ".png")
  ggsave(tmp, plot = p, width = 6, height = 5, dpi = 72, limitsize = FALSE)
  unlink(tmp)
  invisible(NULL)
}

timeit <- function(expr, n = 3) {
  ts <- numeric(n)
  for (i in seq_len(n)) {
    t0 <- Sys.time()
    force(eval.parent(substitute(expr)))
    ts[i] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  }
  list(best_sec = round(min(ts), 3), mean_sec = round(mean(ts), 3), n_runs = n)
}

res <- list()

# ---- load (.rds; analogous to Python loading prepared .h5ad) ----
t0 <- Sys.time()
obj <- readRDS("data/chicken_heart.seurat.rds")
Idents(obj) <- group_key
res$load_rds <- list(best_sec = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 3),
                     n_runs = 1)
res$n_obs  <- ncol(obj)
res$n_vars <- nrow(obj)

# ---- render: embedding colored by metadata (DimPlot) ----
res$render_embedding_metadata <- timeit(
  force_render(DimPlot(obj, reduction = "umap", group.by = group_key, raster = FALSE)))

# ---- render: embedding colored by one gene (FeaturePlot) ----
res$render_embedding_gene <- timeit(
  force_render(FeaturePlot(obj, features = gene1, reduction = "umap", raster = FALSE)))

# ---- render: multi-gene grid, 3 genes (FeaturePlot faceted) ----
res$render_multigene_grid_3 <- timeit(
  force_render(FeaturePlot(obj, features = top3, reduction = "umap",
                           ncol = 3, raster = FALSE)))

# ---- render: violin by group (VlnPlot) ----
res$render_violin <- timeit(
  force_render(VlnPlot(obj, features = gene1, group.by = group_key, pt.size = 0)))

# ---- render: dotplot, 5 genes (DotPlot) ----
res$render_dotplot_5 <- timeit(
  force_render(DotPlot(obj, features = top5, group.by = group_key)))

# ---- render: composition stacked bar (ggplot on metadata) ----
comp_df <- as.data.frame(table(obj@meta.data[[group_key]], obj@meta.data[[comp_cat]]))
colnames(comp_df) <- c("group", "split", "n")
res$render_composition <- timeit(
  force_render(ggplot(comp_df, aes(x = split, y = n, fill = group)) +
               geom_col(position = "fill") + theme_minimal()))

res$peak_r_mem_mb <- round(rss_mb(), 1)
res$conversion_sec <- round(as.numeric(readLines("data/for_seurat/conv_sec.txt")), 2)
res$tool <- "Seurat (shared R rendering substrate)"
res$seurat_version <- as.character(packageVersion("Seurat"))
res$r_version <- paste(R.version$major, R.version$minor, sep = ".")

dir.create("results", showWarnings = FALSE)
write(toJSON(res, auto_unbox = TRUE, pretty = TRUE), "results/benchmark_r.json")
cat("wrote results/benchmark_r.json\n")
print(res)
