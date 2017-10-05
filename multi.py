#!/usr/bin/env python3

import logging

from plumbum import local
from plumbum import BG
from plumbum import cli
from plumbum.cmd import kill, python3
import plumbum.commands.processes

import sys

import atexit

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--thread_count", type=int, default=1)
    parser.add_argument("--tf_thread_count", type=int, default=1)
    parser.add_argument("--process_count", type=int, default=1)
    parser.add_argument("--env", type=str, default='Pong-v0')
    args = parser.parse_args()

    processes = []
    def cleanup_processes():
        print('Cleaning up processes')
        for process in processes:
            print("Killing {}".format(process.proc.pid))
            if not process.ready():
                kill["-9", process.proc.pid].run()

    atexit.register(cleanup_processes)

    # local.env["CUDA_VISIBLE_DEVICES"] = ""

    # batch_sizes = [args.batch_size]
    # learning_rates = [5e-4]
    # train_frequencies = [2]

    # batch_sizes = [32, 64, 128, 256]
    batch_sizes = [64]
    # learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    learning_rates = [1e-4]
    # train_frequencies = [2, 4]
    train_frequencies = [4]

    for train_frequency in train_frequencies:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                for i in range(args.process_count):
                    print("Starting process {}".format(i))
                    f = python3["-m", "baselines.multi_deepq.experiments.doom",
                            "--batch_size={}".format(batch_size),
                            "--train_frequency={}".format(train_frequency),
                            "--learning_rate={}".format(learning_rate),
                            "--worker_count={}".format(args.thread_count),
                            "--tf_thread_count={}".format(args.tf_thread_count),
                            "--env_name={}".format(args.env)] & BG(stdout=sys.stdout, stderr=sys.stderr)
                    #taskset -cp $i $pid
                    processes.append(f)

                for process in processes:
                    try:
                        process.wait()
                    except plumbum.commands.processes.ProcessExecutionError as e:
                        print("Process failed with {}".format(e))


if __name__ == "__main__":
    main()
