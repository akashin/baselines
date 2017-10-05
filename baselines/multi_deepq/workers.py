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

from collections import namedtuple, defaultdict


class Config(object):

    def __init__(self):
        self.batch_size = 32
        self.gamma = 0.99
        self.replay_size = 50000
        self.num_iterations = 10000
        self.learning_rate = 5e-4
        self.worker_count = 1
        self.tf_thread_count = 8
        self.train_frequency = 2
        self.target_update_frequency = 500

    def __repr__(self):
        s = ''
        for k, v in sorted(self.__dict__.items()):
            s += ',{}={}'.format(k, v)
        return s


class EventTimer(object):

    def __init__(self):
        self.times = defaultdict(float)
        self.visits = defaultdict(int)
        self.total_time = 0
        self.started = False

    def start(self):
        self.started = True
        self.start_time = timer()
        self.current_time = self.start_time

    def measure(self, label):
        if not self.started:
            return

        current_time = timer()
        self.times[label] += current_time - self.current_time
        self.visits[label] += 1
        self.current_time = current_time

    def stop(self):
        if not self.started:
            return

        self.started = False
        self.total_time += timer() - self.start_time

    def print_shares(self):
        for key, value in sorted(self.times.items()):
            logger.record_tabular(key + "_share", value / self.total_time * 100.0)

    def print_averages(self):
        for key, value in sorted(self.times.items()):
            logger.record_tabular(key + "_time", value * 1.0 / self.visits[key])


class Worker(object):
    def __init__(self, is_chief, env, model, config, should_render=True):
        self.config = config
        self.is_chief = is_chief
        self.env = env
        self.should_render = should_render
        self.act, self.train, self.update_target, self.debug = multi_deepq.build_train(
                make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
                q_func=model,
                num_actions=env.action_space.n,
                gamma=config.gamma,
                optimizer=tf.train.AdamOptimizer(learning_rate=config.learning_rate),
                reuse=(not is_chief),
                )

        self.max_iteraction_count = int(self.config.num_iterations)

        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(config.replay_size)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        self.exploration = LinearSchedule(schedule_timesteps=self.config.num_iterations / 4.0, initial_p=1.0, final_p=0.02)

    def run(self, session, coord):
        episode_rewards = [0.0]
        gradients = []
        td_errors = []
        obs = self.env.reset()

        start_time = timer()
        event_timer = EventTimer()
        for t in range(self.max_iteraction_count):
            if t > 1000 and t % 500 == 0:
                event_timer.start()
            # Take action and update exploration to the newest value
            action = self.act(obs[None], update_eps=self.exploration.value(t), session=session)[0]
            event_timer.measure('act')
            new_obs, rew, done, _ = self.env.step(action)
            event_timer.measure('step')
            # Store transition in the replay buffer.
            self.replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = self.env.reset()
                episode_rewards.append(0)

            event_timer.measure('replay_buffer')

            # show_episode = len(episode_rewards) > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            show_episode = len(episode_rewards) % 100 == 0
            if self.should_render and self.is_chief and show_episode:
                # Show off the result
                self.env.render()

            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if t > 1000 and t % self.config.train_frequency == 0:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.config.batch_size)
                td_error, grad_norm = self.train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards),
                        session=session)
                gradients.append(grad_norm)
                td_errors.append(np.mean(td_error))
                # if (t / self.config.train_frequency) % 10 == 0:
                    # print("mean TD error: {}".format(np.mean(td_error)))
                    # print("Gradient norm: {}".format(grad_norm))
                event_timer.measure('train')

            # Update target network periodically.
            if self.is_chief and t % self.config.target_update_frequency == 0:
                self.update_target(session=session)
                event_timer.measure('update_target')

            event_timer.stop()

            if self.is_chief and done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("mean gradient", round(np.mean(gradients[-101:-1]), 5))
                logger.record_tabular("mean td_error", round(np.mean(td_errors[-101:-1]), 5))
                logger.record_tabular("time elapsed", timer() - start_time)
                logger.record_tabular("steps/s", t / (timer() - start_time))
                logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(t)))
                event_timer.print_shares()
                event_timer.print_averages()
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


