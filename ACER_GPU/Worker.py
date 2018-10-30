import numpy as np
import tensorflow as tf
from utils import lambda_return, q_retrace, rollout
from ReplayBuffer import *


class Worker(object):


    def __init__(self, agent, envs, sess, worker_id, k_steps=20, DISCOUNT=0.99, step_limit=5000000, offline_ratio = 0):
        """
        Main Worker class, everything done is done here!
        
        Attributes
        ----------------
        agent : ACERNetwork
            the learning agent including the global parameter net. See ACERNetwork for details
        env : AtariEnvironment
            the environment to interact, see AtariEnvironment for details
        sess : Tensorflow session
        worker_id : str
            thread number of the worker.
        k_steps : int
            maximum trajectory length
        DISCOUNT : float
            discount value for future reward
        step_limit : int
            maximum global steps
        offline_steps : int
            amount of offline steps to be taken per online step if  0 -> A3C with Q-function as critic.
        
        """
        self.agent = agent
        self.envs = envs
        self.sess = sess
        self.name = "worker_"+worker_id
        self.memory = [ReplayBuffer(int(50000/k_steps)) for _ in envs]
        self.RETURN_STEPS = k_steps
        self.DISCOUNT = DISCOUNT
        self.MAX_STEPS = step_limit+step_limit*offline_ratio # step limit is for the environment. If a replay ratio of 4 is used, 200.000 steps in the environment -> 1.000.000  in the model.
        self.offline_ratio = offline_ratio

    def work_acer(self):
        b_states=[[None] for _ in self.envs]
        dones = [True for _ in self.envs]
        print(dones)
        step = 0
        rewardlist=[]
        print(self.name, " using ", self.offline_ratio, " per online step")

        while step < self.MAX_STEPS:
            """
            """
            for it in range(len(self.envs)):
                # n -step rollout from the environment, with n = RETURN_STEPS or until done.
                b_states[it], b_actions, b_rewards, b_mus, dones[it] = rollout(self.agent, self.envs[it], [b_states[it][-1]], dones[it], self.RETURN_STEPS)
                pi, q_a, val = self.agent.get_retrace_values(b_states[it][:-1], b_actions)
                if it == 0:
                    rewardlist.append(np.sum(b_rewards))
                    if dones[it]==True:
                        print(step,np.sum(rewardlist))
                        rewardlist=[]
                        
                importance_weights = np.divide(pi, np.add(b_mus, 1e-14))
                importance_weights_a = np.take(np.reshape(importance_weights, [-1]), (
                        np.arange(importance_weights.shape[0]) * importance_weights.shape[1] + b_actions))
            #calculate retrace values.
                retrace_targets = q_retrace(b_rewards, dones[it], q_a, val, importance_weights_a, self.DISCOUNT)
                                    
            #update step, returns current global step and summary (not used here)
                _, step = self.agent.update_step(b_states[it][:-1], b_actions, retrace_targets, importance_weights)
            # append trajectory to the replay buffer
                self.memory[it].remember((b_states[it], b_actions, b_rewards, b_mus, dones[it]))
            #offline version, instead of rollout the trajectory is sampled.
                if self.offline_ratio>0 and self.memory.can_sample():
                    for _ in range(self.offline_ratio):

                        mem_states, mem_actions, mem_rewards, mem_mus, mem_done = self.memory[it].sample_from_memory()
                        pi, q_a, val = self.agent.get_retrace_values(mem_states[:-1], mem_actions)

                        importance_weights = np.divide(pi, np.add(mem_mus, 1e-14))
                        importance_weights_a = np.take(np.reshape(importance_weights, [-1]), (
                                np.arange(importance_weights.shape[0]) * importance_weights.shape[1] + mem_actions))
                        retrace_targets = q_retrace(mem_rewards, mem_done, q_a, val, importance_weights_a, self.DISCOUNT)
                        sum, step = self.agent.update_step(mem_states[:-1], mem_actions, retrace_targets, importance_weights)