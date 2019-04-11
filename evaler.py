from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import numpy as np
import imageio
import tensorflow as tf
import tensorflow.contrib.slim as slim

from input_ops import create_input_ops
from config import argparser
from util import log


class Evaler(object):

    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.train_dir = config.train_dir
        log.info("self.train_dir = %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        self.dataset = dataset

        _, self.batch = create_input_ops(dataset, self.batch_size,
                                         is_training=False,
                                         shuffle=False)

        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
        self.step_op = tf.no_op(name='step_no_op')

        # --- vars ---
        all_vars = tf.trainable_variables()
        log.warn("********* var ********** ")
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

        tf.set_random_seed(123)

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'GPU': 1},
        )
        self.session = tf.Session(config=session_config)

        # --- checkpoint and monitoring ---
        self.saver = tf.train.Saver(max_to_keep=100)

        self.checkpoint = config.checkpoint
        if self.checkpoint is None and self.train_dir:
            self.checkpoint = tf.train.latest_checkpoint(self.train_dir)
            log.info("Checkpoint path : %s", self.checkpoint)
        elif self.checkpoint is None:
            log.warn("No checkpoint is given. Just random initialization :-)")
            self.session.run(tf.global_variables_initializer())
        else:
            log.info("Checkpoint path : %s", self.checkpoint)

    def eval_run(self):
        # load checkpoint
        if self.checkpoint:
            self.saver.restore(self.session, self.checkpoint)
            log.info("Loaded from checkpoint!")

        log.infov("Start Inference and Evaluation")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.session,
                                               coord=coord, start=True)

        use_test_id_list = self.config.data_id_list is not None

        try:
            if use_test_id_list:
                with open(self.config.data_id_list, 'r') as id_list_path:
                    id_list = id_list_path.readlines()
                self.id_list = [id.strip().split(' ') for id in id_list]

            if self.config.plot_image:
                if not os.path.exists(self.config.output_dir):
                    os.makedirs(self.config.output_dir)

            if self.config.loss or self.config.plot_image:
                loss_all = []
                time_all = 0
                step = None
                s = 0
                continue_evaluate = True
                while continue_evaluate:
                    # get testing batch
                    if use_test_id_list:
                        batch_id_list = self.id_list[self.batch_size*s:self.batch_size*(s+1)]
                        batch_chunk = self.get_batch_chunk(batch_id_list)
                    else:
                        batch_chunk = self.get_batch_chunk()

                    # inference
                    step, loss, img, batch_id, step_time = \
                        self.run_single_step(batch_chunk, step=s)

                    # plot images
                    if self.config.plot_image:
                        if use_test_id_list:
                            for i in range(self.batch_size):
                                for img_key in img.keys():
                                    model_name = batch_id_list[i][0].split('_')[0]
                                    target_id = '_'.join(batch_id_list[i][0].split('_')[1:])
                                    source_id = '-'.join(['_'.join(id.split('_')[1:])
                                                          for id in batch_id_list[i][1:]])
                                    img_name = '{}_target_{}_source_{}_{}.png'.format(
                                        model_name, target_id, source_id, img_key)
                                    if self.config.plot_image:
                                        imageio.imwrite(os.path.join(
                                            self.config.output_dir, img_name),
                                            img[img_key][i])
                        else:
                            raise ValueError('Plotting images requires an id list.')

                    loss_all.append(np.array(loss.values()))
                    time_all += step_time

                    s += 1
                    if use_test_id_list:
                        continue_evaluate = s < len(self.id_list)/self.batch_size
                    else:
                        continue_evaluate = s < self.config.max_eval_steps

                    # report loss
                    if not self.config.quiet:
                        loss_avg = np.average(np.stack(loss_all), axis=0)
                        self.log_message(
                            s, loss_avg, loss.keys(), time_all,
                            write_summary=self.config.write_summary,
                            summary_file=self.config.summary_file,
                            final_step=not continue_evaluate,
                        )

        except Exception as e:
            coord.request_stop(e)

        log.warning('Completed Evaluation.')

        coord.request_stop()
        try:
            coord.join(threads, stop_grace_period_secs=3)
        except RuntimeError as e:
            log.warn(str(e))

    def get_batch_chunk(self, id_batch_list=None):
        if id_batch_list is not None:
            image = []
            pose = []
            id = []
            for id_data in id_batch_list:
                img, p = self.dataset.get_data_by_id(id_data)
                image.append(img)
                pose.append(p)
                id.append(id_data[0])
            batch_chunk = {
                'image': np.stack(image, axis=0),
                'camera_pose': np.stack(pose, axis=0),
                'id': np.stack(id, axis=0)
            }
        else:
            batch_chunk = self.session.run(self.batch)
        return batch_chunk

    def run_single_step(self, batch_chunk, step=None, is_train=False):
        _start_time = time.time()

        [step, loss, img, _] = self.session.run(
            [self.global_step, self.model.eval_loss,
             self.model.eval_img, self.step_op],
            feed_dict=self.model.get_feed_dict(batch_chunk, step=step,
                                               is_training=is_train)
        )

        _end_time = time.time()

        return step, loss, img, batch_chunk['id'][0], (_end_time - _start_time)

    def log_message(self, step, loss, loss_key, time, write_summary=False,
                    summary_file=None, is_train=False, final_step=False):
        loss_str = ""
        for key, i in sorted(zip(loss_key, range(len(loss_key)))):
            loss_str += "{}:{loss: .5f}\n".format(
                loss_key[i], loss=loss[i] if 'loss' not in loss_key[i] else loss[i]/2*3)
        log_fn = (is_train and log.info or log.infov)
        if self.config.data_id_list is None:
            data_str = 'Total datapoint: {}'.format(
                self.batch_size*self.config.max_eval_steps)
        else:
            data_str = 'Total datapoint: {} from {}'.format(
                len(self.id_list), self.config.data_id_list)
            report_tag = "[Final Avg Report] {data_str}\n" if final_step \
                else "[{split_mode:5s} step {step:5d}]\n".format(
                    split_mode=('Report'), step=step)
            msg = (
                report_tag +
                "[Loss]\n{loss_str}" +
                "[Time] ({time:.3f} sec)"
                ).format(split_mode=('Report'),
                         step=step,
                         data_str=data_str,
                         loss_str=loss_str,
                         time=time)
        log_fn(msg)
        if final_step:
            log.infov("Checkpoint: %s", self.checkpoint)
            log.infov("Dataset: %s", self.config.dataset)
            if write_summary:
                log.infov("Write the summary to: %s", summary_file)
                final_msg = 'Checkpoint: {}\nDataset: {}\n{}{}'.format(
                    self.checkpoint, self.config.dataset,
                    '' if self.config.data_id_list is None else 'Id list: {}\n'.format(
                        self.config.data_id_list),
                    msg)
                with open(summary_file, 'w') as f:
                    f.write(final_msg)


def main():

    config, model, _, dataset_test = argparser(is_train=False)

    evaler = Evaler(config, model, dataset_test)

    log.warning("dataset: %s", config.dataset)
    evaler.eval_run()

if __name__ == '__main__':
    main()
