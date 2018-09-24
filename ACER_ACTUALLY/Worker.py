import numpy as np
import tensorflow as tf
from utils import lambda_return, q_retrace, rollout


class Worker(object):

    def __init__(self, agent, env, sess, worker_id, replay_buffer, k_steps=20, DISCOUNT=0.99, step_limit=5000000, online_ratio = 0.5):
        self.agent = agent
        self.env = env
        self.sess = sess
        self.name = "worker_"+worker_id
        self.memory = replay_buffer
        self.RETURN_STEPS = k_steps
        self.DISCOUNT = DISCOUNT
        self.MAX_STEPS = step_limit
        self.target_ratio = online_ratio

    def work_acer(self):
        b_states=[None]
        done = True
        online = True
        replay_ratio = 1
        step = 0

        while step < self.MAX_STEPS:
            if online or not self.memory.can_sample():
                self.agent.update_target()
                b_states, b_actions, b_rewards, b_mus, done = rollout(self.agent, self.env, [b_states[-1]], done, self.RETURN_STEPS)
                pi, q_a, val = self.agent.get_retrace_values(b_states[:-1], b_actions)

                importance_weights = np.ones_like(pi)
                importance_weights_a = np.take(np.reshape(importance_weights, [-1]), (
                        np.arange(importance_weights.shape[0]) * importance_weights.shape[1] + b_actions))
                retrace_targets = q_retrace(b_rewards, done, q_a, val, importance_weights_a, self.DISCOUNT)
                _, step = self.agent.update_step(b_states[:-1], b_actions, retrace_targets, importance_weights)
                self.memory.remember((b_states, b_actions, b_rewards, b_mus, done))
                if done:
                    replay_ratio = replay_ratio * 0.99 + 0.01
                    if replay_ratio>self.target_ratio:
                        online = False
            else:
                mem_states, mem_actions, mem_rewards, mem_mus, done = self.memory.sample_from_memory()
                pi, q_a, val = self.agent.get_retrace_values(mem_states[:-1], mem_actions)

                importance_weights = np.divide(pi, np.add(mem_mus, 1e-14))
                importance_weights_a = np.take(np.reshape(importance_weights, [-1]), (
                        np.arange(importance_weights.shape[0]) * importance_weights.shape[1] + mem_actions))
                retrace_targets = q_retrace(mem_rewards, done, q_a, val, importance_weights_a, self.DISCOUNT)
                sum, step = self.agent.update_step(mem_states[:-1], mem_actions, retrace_targets, importance_weights)
                replay_ratio = replay_ratio * 0.99
                if replay_ratio < self.target_ratio:
                    online = True

    def work_and_eval_acer(self, net_saver, TB_DIR):
        b_states = [None]
        done = True
        online = True
        replay_ratio = 1
        step = 0
        runningreward = 1
        bestreward = 0
        rewardlist=[]
        performance={}
        next_verbose = 0
        summary_writer = tf.summary.FileWriter(TB_DIR + "/tb", self.sess.graph, flush_secs=30)

        while step < self.MAX_STEPS:
            if online or not self.memory.can_sample():
                self.agent.update_target()
                b_states, b_actions, b_rewards, b_mus, done = rollout(self.agent, self.env, [b_states[-1]], done,
                                                                      self.RETURN_STEPS)
                pi, q_a, val = self.agent.get_retrace_values(b_states[:-1], b_actions)
                rewardlist.append(np.sum(b_rewards))
                importance_weights = np.ones_like(pi)
                importance_weights_a = np.take(np.reshape(importance_weights, [-1]), (
                        np.arange(importance_weights.shape[0]) * importance_weights.shape[1] + b_actions))
                retrace_targets = q_retrace(b_rewards, done, q_a, val, importance_weights_a, self.DISCOUNT)
                sum, step = self.agent.update_step(b_states[:-1], b_actions, retrace_targets, importance_weights)
                self.memory.remember((b_states, b_actions, b_rewards, b_mus, done))
                if done:
                    replay_ratio = replay_ratio * 0.99 + 0.01
                    if replay_ratio > self.target_ratio:
                        online = False
                    bestreward = np.maximum(bestreward,np.sum(rewardlist))
                    runningreward = 0.9*runningreward+0.1*np.sum(rewardlist)
                    rewardlist=[]
                    if step > next_verbose:
                        print("Worker ", self.name, "At ", step, " Running/Max: ", runningreward, bestreward,
                              " Replay Ratio: ",
                              replay_ratio)
                        print("pi:", self.agent.get_pi(b_states[-1]))
                        print("Saving Model")
                        net_saver.save(self.sess, TB_DIR + "checkpoints/model" + str(step) + ".cptk")
                    if sum is not None:
                        summary_writer.add_summary(sum, step)
                    next_verbose +=(self.MAX_STEPS/5000)
                    print(next_verbose)

            else:
                mem_states, mem_actions, mem_rewards, mem_mus, done = self.memory.sample_from_memory()
                pi, q_a, val = self.agent.get_retrace_values(mem_states[:-1], mem_actions)

                importance_weights = np.divide(pi, np.add(mem_mus, 1e-14))
                importance_weights_a = np.take(np.reshape(importance_weights, [-1]), (
                        np.arange(importance_weights.shape[0]) * importance_weights.shape[1] + mem_actions))
                retrace_targets = q_retrace(mem_rewards, done, q_a, val, importance_weights_a, self.DISCOUNT)
                sum, step = self.agent.update_step(mem_states[:-1], mem_actions, retrace_targets, importance_weights)
                replay_ratio = replay_ratio * 0.99
                if replay_ratio < self.target_ratio:
                    online = True
