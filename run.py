#!/usr/bin/env python
from .model import * 
from radcad import Model, Simulation, Experiment 

N = 100
TIMESTEPS = 500
RUNS = 1
P_SPONT = 2e-3 
DEG_PROD = 10
DEG_DESTR = 15
P_INIT_STATE = 0.2

state = np.random.binomial(size=N, p=P_INIT_STATE, n=1)
I_prod = rand_interaction(N, DEG_PROD)
I_destr = rand_interaction(N, DEG_DESTR)

params = {
    "size": [N],
    "productive": [I_prod],
    "destructive": [I_destr],
    "spontaneous": [P_SPONT],
}

initial_state = { "state": state }

state_update_blocks = [
    {
        "policies": {},
        "variables": { "state": update }
    }
]

model = Model(
        initial_state=initial_state, 
        state_update_blocks=state_update_blocks,
        params=params,
    )

simulation = Simulation(model=model, timesteps=TIMESTEPS, runs=RUNS)
simulation.model = model

result = simulation.run()

