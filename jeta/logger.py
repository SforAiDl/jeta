import logging
import os
import time
from datetime import timedelta

import pandas as pd
import tensorflow as tf
import wandb


class Locallog:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank):

    log_formatter = Locallog()
    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = "%s-%i" % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)


class TensorboardLogger(object):
    """
    Logging with tensorboard
    """

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = None

    def __enter__(self):
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.writer.set_as_default()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def scalar(self, name, value, step):
        with tf.summary.record_if(True):
            tf.summary.scalar(name, value, step=step)

    def histogram(self, name, values, step):
        with tf.summary.record_if(True):
            tf.summary.histogram(name, values, step=step)

    def custom_scalars(self, name, scalars, step):
        with tf.summary.record_if(True):
            for scalar in scalars:
                tf.summary.scalar(name, scalar, step=step)

    def custom_histograms(self, name, histograms, step):
        with tf.summary.record_if(True):
            for histogram in histograms:
                tf.summary.histogram(name, histogram, step=step)


class WandbLogger(object):
    """
    Creating a wandb logger
    """

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.logger = None

    def __enter__(self):
        self.logger = wandb.init(name=self.name, config=self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.join()

    def scalar(self, name, value, step):
        self.logger.log(name, value, step=step)

    def histogram(self, name, values, step):
        self.logger.log(name, values, step=step)

    def custom_scalars(self, name, scalars, step):
        for scalar in scalars:
            self.logger.log(name, scalar, step=step)

    def custom_histograms(self, name, histograms, step):
        for histogram in histograms:
            self.logger.log(name, histogram, step=step)
