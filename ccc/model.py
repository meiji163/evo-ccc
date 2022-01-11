import numpy as np
from scipy.spatial.distance import squareform

def update(params, substep, state_history, previous_state, policy_input):
    state = previous_state["state"]
    N = params["size"]

    # calculate fitness from interactions
    I_prod = params["productive"] 
    I_destr = params["destructive"]
    I = I_prod - I_destr
        
    # spontaneous production/destruction
    prob_flip = params["spontaneous"]
    flip = np.random.binomial(size=N, p=prob_flip, n=1)
    
    method = params["update_method"]
    if method == "random":
        # update in random order 
        order = np.random.permutation(N)
        for i in order:
            fit = I @ state @ state
            if fit[i] > 0:
                state[i] = 1
            elif fit[i] < 0:
                state[i] = 0
            if flip[i]:
                state[i] = 1 - state[i]            
    elif method == "parallel":
        # update simultaneously 
        fit = I @ state @ state
        state[np.where(fit>0)] = 1
        state[np.where(fit<0)] = 0

        flip_idx = np.where(flip>0)
        state[flip_idx] = 1 - state[flip_idx]
            
    return "state", state

def rand_interaction(n: int, avg_deg: float):
    nC2 = n*(n-1)//2
    prob = avg_deg/nC2
    I = np.zeros((n,n,n), dtype=np.int64)
    for i in range(n):
        compressed = np.random.binomial(size=nC2, p=prob, n=1)
        I[i,:] = squareform(compressed)
    return I
