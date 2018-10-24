import numpy as np
import tensorflow as tf
from utils import lambda_return, q_retrace


def train_acer(agent, env, sess, worker_id, replay_buffer, k_steps=20, DISCOUNT=0.99, step_limit=5000000,
               verbose_every=1000, net_saver=None, TB_DIR=None):
    print("Starting Agent", worker_id)
    rewardlist = []
    runningreward = 0
    bestreward = 0
    replay_ratio = 1
    avg_ep_length = 20
    RETURN_STEPS = k_steps
    b_states = [None]
    step = 0
    done = True
    online = True
    write_summary = False
    sum, summary_writer = None, None
    if worker_id == 0:
        if TB_DIR != None:
            summary_writer = tf.summary.FileWriter(TB_DIR + "/tb", sess.graph, flush_secs=30)
            write_summary = True
    while step < step_limit:
        if online or step < 1000000:
            agent.update_target()
            b_states, b_actions, b_rewards, b_mus, done = rollout(agent, env, [b_states[-1]], done, RETURN_STEPS)
            pi, q_a, val = agent.get_retrace_values(b_states[:-1], b_actions)

            importance_weights = np.ones_like(pi)
            importance_weights_a = np.take(np.reshape(importance_weights, [-1]), (
                    np.arange(importance_weights.shape[0]) * importance_weights.shape[1] + b_actions))
            retrace_targets = q_retrace(b_rewards, done, q_a, val, importance_weights_a, DISCOUNT)
            sum, step = agent.update_step(b_states[:-1], b_actions, retrace_targets, importance_weights)
            replay_buffer.remember((b_states, b_actions, b_rewards, b_mus, done))
            rewardlist.append(np.sum(b_rewards))
            if done:
                bestreward = np.maximum(np.sum(rewardlist), bestreward)
                runningreward = 0.95 * runningreward + 0.05 * np.sum(rewardlist)
                replay_ratio = replay_ratio * 0.99 + 0.01
                offline_decider = np.random.rand(1)*0.7
                if offline_decider+0.3>(1-step/step_limit):
                    online = False
                avg_ep_length = 0.9 * avg_ep_length + 0.1 * len(rewardlist)
                rewardlist = []

        else:
            mem_states, mem_actions, mem_rewards, mem_mus, done = replay_buffer.sample_from_memory()
            pi, q_a, val = agent.get_retrace_values(mem_states[:-1], mem_actions)

            importance_weights = np.divide(pi, np.add(mem_mus, 1e-14))
            importance_weights_a = np.take(np.reshape(importance_weights, [-1]), (
                    np.arange(importance_weights.shape[0]) * importance_weights.shape[1] + mem_actions))
            retrace_targets = q_retrace(mem_rewards, done, q_a, val, importance_weights_a, DISCOUNT)
            sum, step = agent.update_step(mem_states[:-1], mem_actions, retrace_targets, importance_weights)
            online = step % 2 == 0
            replay_ratio = replay_ratio*0.99

        if step % verbose_every == 0:
            print("Worker ", worker_id, "At ", step, " Running/Max: ", runningreward, bestreward, " Replay Ratio: ",
                  replay_ratio)
            print("EPlen:", avg_ep_length * RETURN_STEPS, "pi:", agent.get_pi(b_states[-1]))

        if step % 5000 == 0:
            print("Saving Model")
            net_saver.save(sess, TB_DIR + "checkpoints/model" + str(step) + ".cptk")
        if write_summary and sum is not None:
            summary_writer.add_summary(sum, step)


def train(agent, env, sess, worker_id, k_steps=20, DISCOUNT=0.99, step_limit=5000000, verbose_every=50, net_saver=None,
          TB_DIR=None):
    print("Starting Agent", worker_id)
    rewardlist = []
    runningreward = 0
    bestreward = 0
    RETURN_STEPS = k_steps
    b_states = [None]
    step = 0
    done = True
    write_summary = False
    if worker_id == 0:
        if TB_DIR != None:
            summary_writer = tf.summary.FileWriter(TB_DIR + "/tb", sess.graph, flush_secs=30)
            write_summary = True
    while step < step_limit:
        b_states, b_actions, b_rewards = [b_states[-1]], [], []
        if done:
            agent.update_target()
            b_states = [env.reset()]
            done = False
            runningreward = 0.9 * runningreward + 0.1 * np.sum(rewardlist)
            bestreward = np.maximum(bestreward, np.sum(rewardlist))
            rewardlist = []
        while not done and len(b_states) <= RETURN_STEPS:
            action = agent.get_action(b_states[-1])
            state, reward, done, _ = env.step(action)
            rewardlist.append(reward)
            b_actions.append(action)
            b_states.append(state)
            b_rewards.append(reward)

        b_values = agent.get_values(b_states)
        b_targets = lambda_return(b_rewards, b_values, done, DISCOUNT, 1)
        b_values = np.reshape(b_values[:-1], [-1])
        b_advantages = np.subtract(b_targets, b_values)
        if step < 5000:
            b_advantages = np.zeros_like(
                b_advantages)  # idea to pretrain value function to stop early divergence due to strong adv->grads
        b_ppo_pi = agent.GlobalNet.get_ppo_pi_for_actions(b_states[:-1], b_actions)
        agent.update_ppo()
        summary, step = agent.update_step(b_states[:-1], b_actions, b_targets, b_advantages, b_ppo_pi, write_summary)
        if step % verbose_every == 0:
            print("Worker ", worker_id, "At ", step, " Running/Max: ", runningreward, bestreward)

        if step % 2500 == 0:
            print("Saving Model")
            net_saver.save(sess, TB_DIR + "checkpoints/model" + str(step) + ".cptk")
        if step % 1000 == 0:
            print(agent.get_pi(b_states[-1]), b_values[-1])
        if write_summary:
            summary_writer.add_summary(summary, step)  #


def test(agent, env, runs, render=True, capture=False, capture_dir=None):
    for it in range(runs):
        state = env.reset()
        rewardlist = []
        done = False
        while not done:
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            rewardlist.append(reward)
            if render:
                env.render()
        print("Run ", it, "Reward: ", np.sum(rewardlist))


def rollout(agent, env, states, done, RETURN_STEPS):
    if done:
        states = [env.reset()]
    done = False
    actions, rewards, mus = [], [], []
    while not done and len(states) < RETURN_STEPS:
        action, pi = agent.get_action_with_probs(states[-1])
        state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        mus.append(pi)
    return (np.array(states), actions, rewards, mus, done)
