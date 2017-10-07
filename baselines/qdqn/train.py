import numpy as np
import tensorflow as tf
import os

import baselines.common.tf_util as U

from baselines import logger

import threading

import shutil

from tensorflow.contrib.staging import StagingArea

from baselines.qdqn.workers import Learner, Actor, Trainer

def create_logger(log_dir):
    return logger.Logger(log_dir,
            [logger.make_output_format(f, log_dir) for f in logger.LOG_OUTPUT_FORMATS]
            )

def create_json_logger(log_dir):
    return logger.Logger(log_dir,
            [logger.make_output_format(f, log_dir) for f in ['json']]
            )

def learner_dir(base_dir):
    return os.path.join(base_dir, "learner")

def create_learner_logger(base_dir):
    return create_logger(learner_dir(base_dir))

def actor_dir(base_dir, index):
    return os.path.join(base_dir, "actor_{}".format(index))

def create_actor_logger(base_dir, index):
    return create_logger(actor_dir(base_dir, index))

def trainer_dir(base_dir):
    return os.path.join(base_dir, "trainer")

def create_trainer_logger(base_dir):
    return create_logger(trainer_dir(base_dir))


def train_qdqn(config, log_dir, make_env, model, cleanup=False):
    if cleanup:
        shutil.rmtree(log_dir, ignore_errors=True)

    np.random.seed(42)
    tf.set_random_seed(7)

    env = make_env(666)
    observation_space = env.observation_space
    action_space = env.action_space
    env.close()

    actor_queue = tf.FIFOQueue(capacity=config.queue_capacity,
            dtypes=[tf.uint8, tf.int32, tf.float32, tf.uint8, tf.float32, tf.int32],
            shapes=[observation_space.shape, action_space.shape, [], observation_space.shape, [], []])

    batch_shape = [config.batch_size]
    learner_queue = StagingArea(
            dtypes=[tf.uint8, tf.int32, tf.float32, tf.uint8, tf.float32],
            shapes=[
                batch_shape + list(observation_space.shape),
                batch_shape + list(action_space.shape),
                batch_shape,
                batch_shape + list(observation_space.shape),
                batch_shape],
            memory_limit=2**32)

    coord = tf.train.Coordinator()

    workers = []
    learner = Learner(
            learner_dir(log_dir),
            observation_space,
            action_space,
            model,
            learner_queue,
            config,
            create_learner_logger(log_dir))
    workers.append(learner)

    trainer = Trainer(config, actor_queue, learner_queue, observation_space, action_space,
            create_trainer_logger(log_dir))
    workers.append(trainer)

    for i in range(config.actor_count):
        workers.append(Actor(
            i,
            i == 0,
            make_env(i),
            model,
            actor_queue,
            config,
            create_actor_logger(log_dir, i),
            create_json_logger(os.path.join(actor_dir(log_dir, i), 'episodes')),
            should_render=False,))

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
