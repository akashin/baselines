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


class Worker(object):
    def __init__(self, is_chief, env, model, batch_size, gamma=0.99):
        self.is_chief = is_chief
        self.batch_size = batch_size
        self.env = env
        self.act, self.train, self.update_target, self.debug = multi_deepq.build_train(
                make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
                q_func=model,
                num_actions=env.action_space.n,
                gamma=gamma,
                optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
                reuse=(not is_chief),
                )

        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(10000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        self.exploration = LinearSchedule(schedule_timesteps=100000, initial_p=1.0, final_p=0.02)

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

            # show_episode = len(episode_rewards) > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            show_episode = len(episode_rewards) % 10 == 0
            if self.is_chief and show_episode:
                # Show off the result
                self.env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000 and t % 4 == 0:
                    obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                    self.train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards),
                            session=session)
                # Update target network periodically.
                if self.is_chief and t % 500 == 0:
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


