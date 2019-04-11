from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import numpy as np
from six.moves import xrange
import tensorflow as tf
import tensorflow.contrib.slim as slim

from input_ops import create_input_ops
from config import argparser
from util import log


class Trainer(object):

    def __init__(self, config, model, dataset, dataset_test):
        self.config = config
        self.model = model
        hyper_parameter_str = 'bs_{}_lr_flow_{}_pixel_{}_d_{}'.format(
            config.batch_size,
            config.learning_rate_f,
            config.learning_rate_p,
            config.learning_rate_d,
        )

        self.train_dir = './train_dir/%s-%s-%s-num_input-%s-%s' % (
            config.dataset,
            config.prefix,
            hyper_parameter_str,
            str(config.num_input),
            time.strftime("%Y%m%d-%H%M%S")
        )

        if not os.path.exists(self.train_dir): os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        _, self.batch_train = create_input_ops(
            dataset, self.batch_size, is_training=True)
        _, self.batch_test = create_input_ops(
            dataset_test, self.batch_size, is_training=False)

        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.learning_rate_p = config.learning_rate_p
        self.learning_rate_f = config.learning_rate_f
        self.learning_rate_d = config.learning_rate_d

        self.check_op = tf.no_op()

        # --- checkpoint and monitoring ---
        all_vars = tf.trainable_variables()

        f_var = [v for v in all_vars if 'Flow' in v.op.name or 'flow' in v.op.name]
        log.warn("********* f_var ********** ")
        slim.model_analyzer.analyze_vars(f_var, print_info=True)

        p_var = [v for v in all_vars if 'Pixel' in v.op.name or 'pixel' in v.op.name]
        log.warn("********* p_var ********** ")
        slim.model_analyzer.analyze_vars(p_var, print_info=True)

        d_var = [v for v in all_vars if v.op.name.startswith('Discriminator')]
        log.warn("********* d_var ********** ")
        slim.model_analyzer.analyze_vars(d_var, print_info=True)

        # the whole model without the discriminator
        g_var = p_var + f_var

        self.f_optimizer = tf.train.AdamOptimizer(
            self.learning_rate_f,
        ).minimize(self.model.flow_loss,
                   var_list=f_var, name='optimizer_flow_loss')

        self.p_optimizer = tf.train.AdamOptimizer(
            self.learning_rate_p,
        ).minimize(self.model.pixel_loss, global_step=self.global_step,
                   var_list=p_var, name='optimizer_pixel_loss')

        self.p_optimizer_gan = tf.train.AdamOptimizer(
            self.learning_rate_p,
            beta1=0.5
        ).minimize(self.model.pixel_loss_gan, global_step=self.global_step,
                   var_list=p_var, name='optimizer_pixel_loss_gan')

        self.d_optimizer = tf.train.AdamOptimizer(
            self.learning_rate_d,
            beta1=0.5
        ).minimize(self.model.d_loss, global_step=self.global_step,
                   var_list=d_var, name='optimizer_discriminator_loss')

        self.train_summary_op = tf.summary.merge_all(key='train')
        self.test_summary_op = tf.summary.merge_all(key='test')

        self.saver = tf.train.Saver(max_to_keep=100)
        self.pretrain_saver = tf.train.Saver(var_list=all_vars, max_to_keep=1)
        self.pretrain_saver_p = tf.train.Saver(var_list=p_var, max_to_keep=1)
        self.pretrain_saver_f = tf.train.Saver(var_list=f_var, max_to_keep=1)
        self.pretrain_saver_g = tf.train.Saver(var_list=g_var, max_to_keep=1)
        self.pretrain_saver_d = tf.train.Saver(var_list=d_var, max_to_keep=1)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)
        self.max_steps = self.config.max_steps
        self.ckpt_save_step = self.config.ckpt_save_step
        self.log_step = self.config.log_step
        self.test_sample_step = self.config.test_sample_step
        self.write_summary_step = self.config.write_summary_step
        self.gan_start_step = self.config.gan_start_step

        self.checkpoint_secs = 600  # 10 min

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.pretrain_saver.restore(self.session, self.ckpt_path, )
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

        self.ckpt_path_f = config.checkpoint_f
        if self.ckpt_path_f is not None:
            log.info("Checkpoint path: %s", self.ckpt_path_f)
            self.pretrain_saver_f.restore(self.session, self.ckpt_path_f)
            log.info("Loaded the pretrain Flow module from the provided checkpoint path")

        self.ckpt_path_p = config.checkpoint_p
        if self.ckpt_path_p is not None:
            log.info("Checkpoint path: %s", self.ckpt_path_p)
            self.pretrain_saver_p.restore(self.session, self.ckpt_path_p)
            log.info("Loaded the pretrain Pixel module from the provided checkpoint path")

        self.ckpt_path_g = config.checkpoint_g
        if self.ckpt_path_g is not None:
            log.info("Checkpoint path: %s", self.ckpt_path_g)
            self.pretrain_saver_g.restore(self.session, self.ckpt_path_g)
            log.info("Loaded the pretrain Generator (Pixel&Flow) module from the provided checkpoint path")

        self.ckpt_path_d = config.checkpoint_d
        if self.ckpt_path_d is not None:
            log.info("Checkpoint path: %s", self.ckpt_path_d)
            self.pretrain_saver_d.restore(self.session, self.ckpt_path_d)
            log.info("Loaded the pretrain Discriminator module from the provided checkpoint path")

    def train(self):
        log.infov("Training Starts!")
        print(self.batch_train)

        max_steps = self.max_steps
        ckpt_save_step = self.ckpt_save_step
        log_step = self.log_step
        test_sample_step = self.test_sample_step
        write_summary_step = self.write_summary_step
        gan_start_step = self.gan_start_step

        for s in xrange(max_steps):
            # periodic inference
            if s % test_sample_step == 0:
                step, test_summary, p_loss, f_loss, loss, output, step_time = \
                    self.run_test(self.batch_test, step=s, is_train=False)
                self.log_step_message(step, p_loss, f_loss, loss, step_time, is_train=False)
                self.summary_writer.add_summary(test_summary, global_step=step)

            step, train_summary, p_loss, f_loss, loss, output, step_time = \
                self.run_single_step(self.batch_train, step=s,
                                     opt_gan=s > gan_start_step, is_train=True)
            if s % log_step == 0:
                self.log_step_message(step, p_loss, f_loss, loss, step_time)

            if s % write_summary_step == 0:
                self.summary_writer.add_summary(train_summary, global_step=step)

            if s % ckpt_save_step == 0:
                log.infov("Saved checkpoint at %d", s)
                save_path = self.saver.save(
                    self.session, os.path.join(self.train_dir, 'model'),
                    global_step=step)

    def run_single_step(self, batch, step=None, opt_gan=False, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.train_summary_op, self.model.output,
                 self.model.pixel_loss, self.model.flow_loss,
                 self.model.loss, self.check_op]

        # fetch optimizers
        if not opt_gan:
            # optimize only l1 losses
            fetch += [self.p_optimizer, self.f_optimizer]
        else:
            if step % (self.config.update_rate+1) > 0:
                # train the generator
                fetch += [self.p_optimizer_gan, self.f_optimizer]
            else:
                # train the discriminator
                fetch += [self.d_optimizer]

        fetch_values = self.session.run(
            fetch,
            feed_dict=self.model.get_feed_dict(batch_chunk, step=step)
        )
        [step, summary, output, p_loss, f_loss, loss] = fetch_values[:6]

        _end_time = time.time()

        return step, summary, p_loss, f_loss, loss, output, (_end_time - _start_time)

    def run_test(self, batch, step, is_train=False):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        step, summary, p_loss, f_loss, loss, output = self.session.run(
            [self.global_step, self.test_summary_op,
             self.model.pixel_loss, self.model.flow_loss,
             self.model.loss, self.model.output],
            feed_dict=self.model.get_feed_dict(batch_chunk, step=step, is_training=False)
        )

        _end_time = time.time()

        return step, summary, p_loss, f_loss, loss, output, (_end_time - _start_time)

    def log_step_message(self, step, p_loss, f_loss, loss, step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss: {loss:.5f} " +
                "Pixel loss: {p_loss:.5f} " +
                "Flow loss: {f_loss:.5f} " +
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         p_loss=p_loss,
                         f_loss=f_loss,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time
                         )
               )


def main():
    
    config, model, dataset_train, dataset_test = argparser(is_train=False)

    trainer = Trainer(config, model, dataset_train, dataset_test)

    log.warning("dataset: %s", config.dataset)
    trainer.train()

if __name__ == '__main__':
    main()
