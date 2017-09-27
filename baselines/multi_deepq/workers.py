import tensorflow as tf
import tensorflow.contrib.layers as layers
import itertools
import numpy as np

from timeit import default_timer as timer

import baselines.common.tf_util as U

from baselines import logger

from baselines import multi_deepq
from baselines.multi_deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule

from collections import namedtuple


class Config(object):

    def __init__(self):
        self.batch_size = 32
        self.gamma = 0.99
        self.replay_size = 50000
        self.exploration_length = 500000
        self.learning_rate = 5e-4
        self.worker_count = 1
        self.tf_thread_count = 8
        self.train_frequency = 4
        self.target_update_frequency = 500

    def __repr__(self):
        s = ''
        for k, v in sorted(self.__dict__.items()):
            s += ',{}={}'.format(k, v)
        return s


class Worker(object):
    def __init__(self, is_chief, env, model, config):
        self.config = config
        self.is_chief = is_chief
        self.env = env
        self.act, self.train, self.update_target, self.debug = multi_deepq.build_train(
                make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
                q_func=model,
                num_actions=env.action_space.n,
                gamma=config.gamma,
                optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate),
                reuse=(not is_chief),
                )

        self.max_iteraction_count = self.config.exploration_length * 2

        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(config.replay_size)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        self.exploration = LinearSchedule(schedule_timesteps=self.config.exploration_length, initial_p=1.0, final_p=0.02)

    def run(self, session, coord):
        episode_rewards = [0.0]
        obs = self.env.reset()

        start_time = timer()
        for t in range(self.max_iteraction_count):
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

            # show_episode = len(episode_rewards) > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            show_episode = len(episode_rewards) % 100 == 0
            if self.is_chief and show_episode:
                # Show off the result
                self.env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000 and t % self.config.train_frequency == 0:
                    obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.config.batch_size)
                    self.train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards),
                            session=session)
                # Update target network periodically.
                if self.is_chief and t % self.config.target_update_frequency == 0:
                    self.update_target(session=session)

            if self.is_chief and done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("time elapsed", timer() - start_time)
                logger.record_tabular("steps/s", t / (timer() - start_time))
                logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(t)))
                logger.dump_tabular()


class StupidWorker(object):
    def __init__(self, is_chief, env, model):
        self.env = env
        self.is_chief = is_chief

    def run(self, session, coord):
        _ = self.env.reset()

        start_time = timer()
        for t in itertools.count():
            # Take action and update exploration to the newest value
            # action = self.act(obs[None], update_eps=self.exploration.value(t), session=session)[0]
            action = 0
            _, _, done, _ = self.env.step(action)
            if done:
                obs = self.env.reset()

            if self.is_chief and t % 100 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("time elapsed", timer() - start_time)
                logger.record_tabular("steps/s", t / (timer() - start_time))
                logger.dump_tabular()


