import numpy as np
import tensorflow as tf
import os

import baselines.common.tf_util as U

from baselines import logger

import threading

from baselines.qdqn.workers import Learner, Actor

def learner_dir(base_dir):
    return os.path.join(base_dir, "learner")

def create_learner_logger(base_dir):
    log_dir = learner_dir(base_dir)
    return logger.Logger(log_dir,
            [logger.make_output_format(f, log_dir) for f in logger.LOG_OUTPUT_FORMATS]
            )

def actor_dir(base_dir, index):
    return os.path.join(base_dir, "actor_{}".format(index))

def create_actor_logger(base_dir, index):
    log_dir = actor_dir(base_dir, index)
    return logger.Logger(log_dir,
            [logger.make_output_format(f, log_dir) for f in logger.LOG_OUTPUT_FORMATS]
            )


def train_qdqn(config, log_dir, make_env, model):
    np.random.seed(42)
    tf.set_random_seed(7)

    env = make_env(666)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()

    queue = tf.PriorityQueue(capacity=config.queue_capacity,
            types=[tf.float32, tf.int32, tf.float32, tf.float32, tf.float32],
            shapes=[observation_space.shape, action_space.shape, [], observation_space.shape, []])

    coord = tf.train.Coordinator()

    workers = []
    learner = Learner(
            learner_dir(log_dir),
            observation_space,
            action_space,
            model,
            queue,
            config,
            create_learner_logger(log_dir))
    workers.append(learner)

    for i in range(config.actor_count):
        workers.append(Actor(
            i,
            i == 0,
            make_env(i),
            model,
            queue,
            config,
            create_actor_logger(log_dir, i),
            should_render=False,))

        # workers.append(StupidWorker(i == 0, make_env(i), model))

    with U.make_session(config.tf_thread_count) as session:
        U.initialize(session=session)

        learner.load(learner_dir(log_dir), session=session);

        threads = []
        for worker in workers:
            worker_fn = lambda: worker.run(session, coord)
            thread = threading.Thread(target=worker_fn)
            thread.start()
            threads.append(thread)

        coord.join(threads)
