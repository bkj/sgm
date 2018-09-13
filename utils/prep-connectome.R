#!/usr/bin/env R

# prep-connectome.R
#
# (Minor) data cleaning to go from raw connectome format
# to something that SGM will like

suppressMessages(library(igraph))
set.seed(1234)

# --
# Helpers

add_isolated_nodes <- function(g, maxv, sparse) {
    A         <- g + vertices((1:maxv)[-as.numeric(V(g)$name)])
    A.ordered <- permute(A, as.numeric(V(A)$name))
    
    if(sparse) {
        A.ordered <- as_edgelist(A.ordered, names=FALSE) - 1    
    } else {
        A.ordered <- as.matrix(A.ordered[])
        A.ordered[A.ordered > 0] <- 1            
    }
    
    return(A.ordered)
}

# --
# CLI

args <- commandArgs(trailingOnly=TRUE)

if(length(args) != 2) {
    stop('prep-connectome.R <inpath> <outpath>')
}

inpath  <- args[1]
outpath <- args[2]
sparse  <- as.numeric(args[3]) == 1

# --
# IO
cat('prep-connectome.R: loading\n', file=stderr())
g1 <- read_graph(file.path(inpath, "data/raw_data/G1.edgelist"), format="ncol", directed=FALSE)
g2 <- read_graph(file.path(inpath, "data/raw_data/G2.edgelist"), format="ncol", directed=FALSE)

# --
# Clean graphs
cat('prep-connectome.R: cleaning\n', file=stderr())

# Add isolated nodes
maxv <- basename(inpath) # !! Hacky -- get intended number of nodes from data directory
maxv <- as.numeric(substr(maxv, 3, nchar(maxv))) - 1

A1.ordered <- add_isolated_nodes(g1, maxv, sparse=sparse)
A2.ordered <- add_isolated_nodes(g2, maxv, sparse=sparse)

if(sparse) {
    P_start <- cbind((1:maxv) - 1, (1:maxv) - 1)
} else {
    P_start <- diag(maxv)    
}

# --
# Save
cat('prep-connectome.R: saving\n', file=stderr())

dir.create(outpath, showWarnings=FALSE, recursive=TRUE, mode="0777")
write.csv(A1.ordered, file.path(outpath, 'A1.ordered.edges'), row.names=FALSE)
write.csv(A2.ordered, file.path(outpath, 'A2.ordered.edges'), row.names=FALSE)
write.csv(P_start, file.path(outpath, 'P_start.edges'), row.names=FALSE)