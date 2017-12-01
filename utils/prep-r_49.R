#!/usr/bin/env R

# prep-r_49.R
#
# Data cleaning to go from raw r_49 data format to 
# something that SGM likes

suppressMessages(library(igraph))
suppressMessages(library(tidyverse))
set.seed(1234)

# --
# Helpers

rsp <- function(n, g){
    s     <- sample(n)
    I     <- diag(n)
    P     <- I[s,]
    alpha <- runif(1, 0, g)
    J     <- matrix(1, n, n)
    bc    <- (1 / n) * J #this is the barycenter
    (1 - alpha) * bc + alpha * P
}


# --
# CLI

args <- commandArgs(trailingOnly=TRUE)
inpath <- ifelse(length(args) >= 1, args[1], './_data/r_49')
outpath <- ifelse(length(args) >= 2, args[2], './_results/r_49')

# --
# IO
cat('prep-r_49: loading\n', file=stderr())

g1 <- read_graph(file.path(inpath, "/data/raw_data/G1.gml"), format='gml')
g2 <- read_graph(file.path(inpath, "/data/raw_data/G2.gml"), format='gml')

train1 <- read_csv(file.path(inpath, "/data/trainData.csv"))
train2 <- read_csv(file.path(inpath, "/data/trainTargets.csv"))

train <- left_join(train1, train2, by="d3mIndex")

# --
# Clean
cat('prep-r_49: cleaning\n', file=stderr())

# Filter graph, if necessary
m <- nrow(train)
min.n <- min(vcount(g1), vcount(g2))
if (m > min.n) {
    train <- train %>% filter(G1.nodeID <= vcount(g1) & G2.nodeID <= vcount(g2))
    g1 <- induced_subgraph(g1, train$G1.nodeID)
    g2 <- induced_subgraph(g2, train$G2.nodeID)
    m <- nrow(train)
}

# Add node labels
if ( is.null(V(g1)$nodeID) | is.null(V(g2)$nodeID) ) {
    V(g1)$nodeID <- train$G1.nodeID
    V(g2)$nodeID <- train$G2.nodeID
}

matched.id1 <- match(train$G1.nodeID, V(g1)$nodeID) 
matched.id2 <- match(train$G2.nodeID, V(g2)$nodeID)

# Drop duplicate matches (if they exist)
dup <- as.numeric(names(which(table(matched.id2)>1)))
if (length(dup) > 0) {
    dup.table <- train[which(matched.id2 %in% dup),]

    dup.id1 <- match(dup.table$G1.nodeID, V(g1)$nodeID)
    dup.id2 <- match(dup.table$G2.nodeID, V(g2)$nodeID)
    
    tmp <- duplicated(matched.id2)
    matched.id1.u <- matched.id1[!tmp]
    matched.id2.u <- matched.id2[!tmp]
} else {
    matched.id1.u <- matched.id1
    matched.id2.u <- matched.id2
}

# Create (sorted) adjacency matrices
num_vertex_1 <- vcount(g1)
num_vertex_2 <- vcount(g2)

V(g1)$id <- 1:vcount(g1)
V(g2)$id <- 1:vcount(g2)

A1 <- as.matrix(g1[]); rownames(A1) <- colnames(A1) <- V(g1)$nodeID
A2 <- as.matrix(g2[]); rownames(A2) <- colnames(A2) <- V(g2)$nodeID

pair <- cbind(matched.id1.u, matched.id2.u)

m <- ifelse(num_vertex_1 == nrow(pair), floor(num_vertex_1 / 2), nrow(pair))

ordered.v1 <- c(pair[,1], setdiff(1:num_vertex_1, pair[,1]))
A1.ordered <- A1[ordered.v1, ordered.v1]

ordered.v2 <- unique(c(pair[,2], setdiff(1:num_vertex_2, pair[,2]))) # wrong for m-to-1!
A2.ordered <- A2[ordered.v2, ordered.v2]

M <- rsp(num_vertex_1 - m, g=0.5)

P_start <- diag(num_vertex_1)
P_start[(m + 1):num_vertex_1, (m + 1):num_vertex_1] <- M

# --
# Save
cat('prep-r_49: saving\n', file=stderr())

dir.create(outpath, showWarnings=FALSE, recursive=TRUE, mode="0777")
write.csv(A1.ordered, file.path(outpath, 'A1.ordered.csv'), row.names=FALSE)
write.csv(A2.ordered, file.path(outpath, 'A2.ordered.csv'), row.names=FALSE)
write.csv(P_start, file.path(outpath, 'P_start.csv'), row.names=FALSE)
