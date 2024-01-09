import os
import numpy as np
import random
import torch
from agent import DDPGAgent
import wandb
import pickle
from multiprocessing import Process


def main(iterations):
    # create log folder
    path_load = ""
    path_save = "ddpg_model/"

    # parameters
    n_episode = 100
    n_timesteps = 60 * 60 * 30
    memory_size = 1000000
    network_size_list = [64, 64, 64, 128, 128, 128]
    pretrain_step_list = [0, 1000, 10000, 0, 1000, 10000]
    batch_size_list = [32, 64, 128, 32, 64, 128]
    gamma = 0.90
    initial_random_steps = 0

    # loss parameters
    tau = 1e-4
    weight_decay = 1e-6
    learning_rate_actor_list = [1e-7, 1e-5, 1e-8, 1e-7, 1e-5, 1e-8]
    learning_rate_critic_list = [1e-6, 1e-4, 5e-7, 1e-6, 1e-4, 5e-7]

    # start a new wandb run to track this script
    wandb.login()
    name_wandb = 'case-' + str(iterations)
    wandb.init(project="euler_agent", name=name_wandb, tags=["debug"])

    runname = 'case-' + str(iterations)
    print('initiate:', runname, 'worker:', os.getpid())

    path_tensorboard = 'logs/tensorboard/run/' + runname
    if not os.path.isdir(f"{path_tensorboard}"):
        os.makedirs(f"{path_tensorboard}")
    else:
        for filename in os.listdir(path_tensorboard):
            file_path = os.path.join(path_tensorboard, filename)
            try:
                os.remove(file_path)
            except:
                pass

    if not os.path.isdir(f"{path_save}"):
        os.makedirs(f"{path_save}")
    else:
        for filename in os.listdir(path_save):
            file_path = os.path.join(path_save, filename)
            try:
                os.remove(file_path)
            except:
                pass

    # load demo on replay memory
    demo_path = "demo/sampled.pkl"
    with open(demo_path, "rb") as f:
        demo = pickle.load(f)
    print('demo size:', len(demo))

    # define seed
    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # define agent
    agent = DDPGAgent(
        demo,
        n_timesteps=n_timesteps,
        iterations=iterations,
        network_size=network_size_list[iterations],
        learning_rate_actor=learning_rate_actor_list[iterations],
        learning_rate_critic=learning_rate_critic_list[iterations],
        memory_size=memory_size,
        batch_size=batch_size_list[iterations],
        pretrain_step=pretrain_step_list[iterations],
        gamma=gamma,
        initial_random_steps=initial_random_steps,
        weight_decay=weight_decay,
        tau=tau,
        path_save=path_save,
        path_load=path_load,
        path_tensorboard=path_tensorboard,
    )

    # train agent
    agent.train(n_episode)

    # save network weights and the replay buffer
    agent.save_model()


if __name__ == '__main__':

    num_processes = 6

    processes = []
    for i in range(num_processes):
        process = Process(target=main, args=(i,))
        processes.append(process)

    # start processes
    for process in processes:
        process.start()

    # join processes
    for process in processes:
        process.join()

