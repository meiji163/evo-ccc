#!/usr/bin/env python
from model import * 
from matplotlib import pyplot as plt
from radcad import Model, Simulation, Experiment, Engine, Backend
from datetime import datetime
import logging

N = 100  # number of agents
TIMESTEPS = 5000 
RUNS = 1 

# ======= evolution parameters =======
P_SPONT = 2e-5 # probability of spontaneous creation & destruction
DEG_PROD = 10 # avg num of productive interactions for each agent 
DEG_DESTR = 11 # avg num of destructive interactions for each agent
P_INIT_STATE = 0.1 # proportion of initial live agents 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    state = np.random.binomial(size=N, p=P_INIT_STATE, n=1)
    I_prod = rand_interaction(N, DEG_PROD)
    I_destr = rand_interaction(N, DEG_DESTR)

    params = {
        "size": [N],
        "productive": [I_prod],
        "destructive": [I_destr],
        "spontaneous": [P_SPONT],
        "update_method": ["parallel"], 
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
    experiment = Experiment(simulation)
    experiment.engine = Engine(processes=1, backend=Backend.PATHOS)

    logging.info("starting simulation")
    result = experiment.run()

    logging.info("saving results")
    for i in range(0, len(result), TIMESTEPS+1):
        history = np.stack([step["state"] for step in result[i:i+TIMESTEPS]])

        now = datetime.now()
        time = now.strftime("%H:%M:%S")
        fname = f"sim_{i}_{time}"
        np.save(fname, history)

        # plot simulation
        plt.matshow(history.T, aspect="auto")
        plt.xlabel("Timestep")
        plt.ylabel("Agents")
        plt.savefig(f"{fname}.png")
