import argparse
import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import threading

from timeit import default_timer as timer

import baselines.common.tf_util as U

from baselines import logger
from baselines import multi_deepq
from baselines.multi_deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


def linear_model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


class Worker(object):
    def __init__(self, is_chief, env, model):
        self.is_chief = is_chief
        self.env = env
        self.act, self.train, self.update_target, self.debug = multi_deepq.build_train(
                make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
                q_func=model,
                num_actions=env.action_space.n,
                optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
                reuse=(not is_chief),
                )

        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        self.exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

    def run(self, session, coord):
        episode_rewards = [0.0]
        obs = self.env.reset()

        start_time = timer()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = self.act(obs[None], update_eps=self.exploration.value(t), session=session)[0]
            new_obs, rew, done, _ = self.env.step(action)
            # Store transition in the replay buffer.
            self.replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = self.env.reset()
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if self.is_chief and is_solved:
                # Show off the result
                self.env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(args.batch_size)
                    self.train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards),
                            session=session)
                    # Update target network periodically.
                if self.is_chief and t % 250 == 0:
                    self.update_target(session=session)

            if self.is_chief and done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("time elapsed", timer() - start_time)
                logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(t)))
                logger.dump_tabular()


def make_env(seed):
    # Create the environment
    env = gym.make("CartPole-v0")
    env.seed(seed)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('--worker_count', help='Batch size', type=int, default=1)
    args = parser.parse_args()

    np.random.seed(42)
    tf.set_random_seed(7)

    log_dir = "./results/multi_deepq_{}".format(args.batch_size)
    logger.configure(dir=log_dir)
    print("Running training with arguments: {} and log_dir: {}".format(args, log_dir))

    coord = tf.train.Coordinator()

    workers = []
    for i in range(args.worker_count):
        workers.append(Worker(i == 0, make_env(i), linear_model))

    with U.make_session(1) as session:
        U.initialize(session=session)

        threads = []
        for worker in workers:
            worker_fn = lambda: worker.run(session, coord)
            thread = threading.Thread(target=worker_fn)
            thread.start()
            threads.append(thread)

        coord.join(threads)
