#!/usr/bin/env R

# ta1-pipeline-prep.R

library(tidyverse)
library(lattice)
library(igraph)
library(Matrix)
library(rjson)
source('sgm_ordered.R')

# --
# data

json_file <- "./r49-pipeline-config.json"
json_data <- fromJSON(paste(readLines(json_file), collapse=""))

gformat <- "gml"
g1file <- paste0(json_data$data_root,"/raw_data/G1.gml")
g2file <- paste0(json_data$data_root,"/raw_data/G2.gml")

g1 <- read_graph(g1file, format=gformat)
g2 <- read_graph(g2file, format=gformat)

train1.file <- paste0(json_data$data_root, "/trainData.csv")
train2.file <- paste0(json_data$data_root, "/trainTargets.csv")

train1 <- read_csv(train1.file)
train2 <- read_csv(train2.file)
train <- left_join(train1, train2, by="d3mIndex")

m <- nrow(train)
min.n <- min(vcount(g1),vcount(g2))
if (m > min.n) {
    train <- train %>% filter(G1.nodeID <= vcount(g1) & G2.nodeID <= vcount(g2))
    g1 <- induced_subgraph(g1, train$G1.nodeID)
    g2 <- induced_subgraph(g2, train$G2.nodeID)
    m <- nrow(train)
}

## connectome graphs don't have nodeID's
if ( is.null(V(g1)$nodeID) | is.null(V(g2)$nodeID) ) {
    V(g1)$nodeID <- train$G1.nodeID
    V(g2)$nodeID <- train$G2.nodeID
}

## ----perm----------------------------------------------------------------
# ideintify the vertex indices of the matching pairs

matched.id1 <- match(train$G1.nodeID, V(g1)$nodeID) 
#sum(V(g1)$nodeID[matched.id1] - train$G1.nodeID) # check the mapping, should be 0

matched.id2 <- match(train$G2.nodeID, V(g2)$nodeID) # 6 / 151 are duplicates!
#sum(V(g2)$nodeID[matched.id2] - train$G2.nodeID)

# --
# dup

dup <- as.numeric(names(which(table(matched.id2)>1)))
ndup <- length(dup)
if (ndup > 0) {
    dup.table <- train[which(matched.id2 %in% dup),]
    print(dup.table %>% arrange(G2.nodeID))

    dup.id1 <- match(dup.table$G1.nodeID, V(g1)$nodeID)
    dup.id2 <- match(dup.table$G2.nodeID, V(g2)$nodeID)
    tmp <- duplicated(matched.id2)
    matched.id1.u <- matched.id1[!tmp]
    matched.id2.u <- matched.id2[!tmp]
} else {
    matched.id1.u <- matched.id1
    matched.id2.u <- matched.id2
}


# --
# sim

n  <- vcount(g1)
n2 <- vcount(g2)

V(g1)$id <- 1:vcount(g1)
V(g2)$id <- 1:vcount(g2)

A1 <- as.matrix(g1[])
rownames(A1) <- colnames(A1) <- V(g1)$nodeID
A2 <- as.matrix(g2[])
rownames(A2) <- colnames(A2) <- V(g2)$nodeID

allow.mto1 <- FALSE
if (allow.mto1) {
    pair <- cbind(matched.id1, matched.id2); #dim(pair)
} else {
    pair <- cbind(matched.id1.u, matched.id2.u); #dim(pair)
}
m <- nrow(pair)
m <- ifelse(n == m, floor(m / 2), m)

patience   <- 20
ordered.v1 <- c(pair[,1], setdiff(1:n, pair[,1]))
ordered.v2 <- unique(c(pair[,2], setdiff(1:n2, pair[,2]))) # wrong for m-to-1!
A1.ordered <- A1[ordered.v1, ordered.v1]
A2.ordered <- A2[ordered.v2, ordered.v2]

set.seed(1234)
M <- rsp(n - m, g=0.5)
S <- diag(n);
S[(m+1):n, (m+1):n] <- M

# --
# Writing

print('Writing prepped files')
write.csv(A1.ordered, './data/A1.ordered')
write.csv(A2.ordered, './data/A2.ordered')
write.csv(S, './data/S')
