#!/usr/bin/env R

# make-data

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
    U[col(U) > row(U)] <- runif(n*(n-1)/2)
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

seed = 123
m    = 200
p    = 0.5
rho  = 0.7

ns = (10:20) * 500
for(n in ns) {
    set.seed(seed)
    
    # Generate graph
    x = rg.sample.correlated.gnp(matrix(p, n, n), rho)
    
    # Print original match
    cat(n, '\t', sqrt(sum((x$B - x$A) ** 2)), '\n')
    
    # Permute unseeded rows
    n_perm = length((m+1):nrow(x$B)) 
    perm   = sample(n_perm)
    x$B[(m+1):nrow(x$B),] = x$B[(m+1):nrow(x$B),][perm,]
    x$B[,(m+1):nrow(x$B)] = x$B[,(m+1):nrow(x$B)][,perm]

    # Seed matrix
    P <- diag(n)
    P[(m + 1):n, (m + 1):n] <- rsp(n - m, g=0.5)
    
    write.csv(x$A, paste0('data/A-', n), row.names=FALSE)
    write.csv(x$B, paste0('data/B-', n), row.names=FALSE)    
    write.csv(P, paste0('data/P-', n), row.names=FALSE)
}
