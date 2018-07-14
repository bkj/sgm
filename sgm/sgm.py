from time import time

def sgm(A, P, B, compute_grad, solve_lap, num_iters, tolerance):
    grad = compute_grad(A, P, B)
    
    stop = False
    for i in range(num_iters):
        iter_start_time = time()
        
        lap_start_time = time()
        T = solve_lap(grad, eye=eye)
        lap_time = time() - lap_start_time
        
        grad_t = compute_grad(A, T, B)
        c = (grad * P).sum()
        d = (grad_t * P).sum() + (grad * T).sum()
        e = (grad_t * T).sum()
        
        if (c - d + e == 0) and (d - 2 * e == 0):
            alpha = 0
        else:
            if (c - d + e == 0):
                alpha = float('inf')
            else:
                alpha = -(d - 2 * e) / (2 * (c - d + e))
        
        f1     = c - e
        falpha = (c - d + e) * alpha ** 2 + (d - 2 * e) * alpha
        
        if (alpha < tolerance) and (alpha > 0) and (falpha > 0) and (falpha > f1):
            P = alpha * P + (1 - alpha) * T
            grad = (alpha * grad) + (1 - alpha) * grad_t
        elif f1 < 0:
            P = T
            grad = grad_t
        else:
            stop = True
        
        iter_time = time() - iter_start_time
        print(json.dumps({
            "iter"          : i,
            "lap_time"      : lap_time,
            "nolap_time"    : iter_time - lap_time,
            
            "max_nodes" : int(max_nodes),
            "n_seeds"   : int(n_seeds),
        }))
        
        if stop:
            break
    
    return solve_lap(P, eye=eye)