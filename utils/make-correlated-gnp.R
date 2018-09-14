#!/usr/bin/env R

# make-correlated-gnp.R

set.seed(789)

rsp <- function(n, g){
    s     <- sample(n)
    I     <- diag(n)
    P     <- I[s,]
    alpha <- runif(1, 0, g)
    J     <- matrix(1, n, n)
    bc    <- (1 / n) * J #this is the barycenter
    
    (1 - alpha) * bc + alpha * P
}

rg.sample <- function(P) {
    n <- nrow(P)
    U <- matrix(0, nrow = n, ncol = n)
    U[col(U) > row(U)] <- runif(n * (n-1) / 2)
    U <- (U + t(U))
    A <- (U < P) + 0
    diag(A) <- 0
    
    A
}

rg.sample.correlated.gnp <- function(P,sigma) {
    A <- rg.sample(P)
    n <- nrow(A)
    
    avec <- A[col(A) > row(A)]
    pvec <- P[col(P) > row(P)]
    bvec <- numeric(n*(n-1)/2)
    
    uvec <- runif(n*(n-1)/2)
    
    idx1 <- which(avec == 1)
    idx0 <- which(avec == 0)
    
    bvec[idx1] <- (uvec[idx1] < (sigma + (1 - sigma)*pvec[idx1])) + 0
    bvec[idx0] <- (uvec[idx0] < (1 - sigma)*pvec[idx0]) + 0
    
    B <- matrix(0, nrow = n, ncol = n)
    B[col(B) > row(B)] <- bvec
    B <- B + t(B)
    diag(B) <- 0
    
    list(A = A, B = B)
}

# --
# Params

args <- commandArgs(trailingOnly=TRUE)

if(length(args) != 3) {
    stop('make-correlated-gnp.R <sparse> <p> <rho>')
}

sparse = args[1] == 1
p      = as.numeric(args[2])
rho    = as.numeric(args[3])
cat("sparse ->", sparse, "| p ->", p, "| rho ->", rho, "\n")

seed          = 123
# num_seeds     = 200

# --
# Run

all_num_nodes = (1:20) * 500
for(n in all_num_nodes) {
    set.seed(seed)
    
    # Generate graph
    x = rg.sample.correlated.gnp(matrix(p, n, n), rho)

    # Print original match
    cat(n, '\t', sqrt(sum((x$B - x$A) ** 2)), '\n')
    
    # Seed matrix
    # P <- diag(n)
    # if(sparse) {
    #     init <- 'vertex'
    #     P[(num_seeds + 1):n, (num_seeds + 1):n] <- 0 # Initialize to vertex of polytope
    # } else {
    #     init <- 'barycenter'
    #     P[(num_seeds + 1):n, (num_seeds + 1):n] <- rsp(n - num_seeds, g=0.5) # Initialize to barycenter        
    # }
    
    if(sparse) {
        A_edgelist <- which(x$A > 0, arr.ind = T) - 1
        B_edgelist <- which(x$B > 0, arr.ind = T) - 1
        # P_edgelist <- which(P > 0, arr.ind=T) - 1
        
        outpath <- paste('_data/synthetic/sparse', p, n, sep='/')
        dir.create(outpath, showWarnings=FALSE, recursive=TRUE, mode="0777")
        
        write.csv(A_edgelist, paste(outpath, 'A', sep='/'), row.names=FALSE)
        write.csv(B_edgelist, paste(outpath, 'B', sep='/'), row.names=FALSE)    
        # write.csv(P_edgelist, paste(outpath, 'P', sep='/'), row.names=FALSE)   
    } else {
        outpath <- paste('_data/synthetic/dense', p, n, sep='/')
        dir.create(outpath, showWarnings=FALSE, recursive=TRUE, mode="0777")
        
        write.csv(x$A, paste(outpath, 'A', sep='/'), row.names=FALSE)
        write.csv(x$B, paste(outpath, 'B', sep='/'), row.names=FALSE)
        # write.csv(P,   paste(outpath, 'P', sep='/'), row.names=FALSE)
    }    
}