#!/usr/bin/env python
from model import * 
from matplotlib import pyplot as plt
from radcad import Model, Simulation, Experiment 
from datetime import datetime
import logging

N = 100 # number of agents
TIMESTEPS = 600
RUNS = 1

# ======= evolution parameters =======
P_SPONT = 2e-4 # probability of spontaneous creation & destruction
DEG_PROD = 10 # avg num of productive interactions for each agent 
DEG_DESTR = 15 # avg num of destructive interactions for each agent
P_INIT_STATE = 0.25 # proportion of initial live agents 


if __name__ == "__main__":
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
    for i in range(0, len(result), TIMESTEPS+1):
        history = np.stack([step["state"] for step in result[i:i+TIMESTEPS]])

        now = datetime.now()
        time = now.strftime("%H:%M:%S")
        fname = f"sim_{i}_{time}"
        np.save(fname, history)

        # plot simulation
        plt.matshow(history.T)
        plt.xlabel("Timestep")
        plt.ylabel("Agents")
        plt.savefig(f"{fname}.png")
