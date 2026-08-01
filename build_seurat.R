#!/usr/bin/env Rscript
# build_seurat.R -- assemble a native Seurat object from the AnnData export in
# data/for_seurat/ (lognorm.mtx + genes/cells/metadata/umap csv), and time the
# AnnData->Seurat conversion. Writes data/chicken_heart.seurat.rds and
# data/for_seurat/conv_sec.txt (the R-side conversion seconds).
suppressPackageStartupMessages({
  library(Seurat); library(SeuratObject); library(Matrix); library(jsonlite)
})
setwd(dirname(normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE)[1]))))

d <- "data/for_seurat"
t0 <- Sys.time()
m <- as(Matrix::readMM(file.path(d, "lognorm.mtx")), "CsparseMatrix")
genes <- read.csv(file.path(d, "genes.csv"), stringsAsFactors = FALSE)$gene
cells <- read.csv(file.path(d, "cells.csv"), stringsAsFactors = FALSE)$cell
rownames(m) <- make.unique(genes); colnames(m) <- cells
meta <- read.csv(file.path(d, "metadata.csv"), row.names = 1, stringsAsFactors = FALSE)
umap <- as.matrix(read.csv(file.path(d, "umap.csv"), row.names = 1))

obj <- CreateSeuratObject(counts = m, meta.data = meta)
obj <- SetAssayData(obj, layer = "data", new.data = m)
obj[["umap"]] <- CreateDimReducObject(embeddings = umap, key = "UMAP_",
                                      assay = DefaultAssay(obj))
Idents(obj) <- if ("cell_type" %in% colnames(meta)) "cell_type" else colnames(meta)[1]

conv_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
saveRDS(obj, "data/chicken_heart.seurat.rds")
writeLines(as.character(round(conv_sec, 2)), file.path(d, "conv_sec.txt"))
cat(sprintf("Built Seurat object: %d cells x %d genes in %.1fs\n",
            ncol(obj), nrow(obj), conv_sec))
