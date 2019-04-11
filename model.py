from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from util import log, train_test_summary
from ops import conv2d, residual_block
from ops import nn_deconv2d as deconv2d
from ssim import tf_ssim
from ops import add_image_coord, resnet_block_convlstm
from ops import bilinear_sampler


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.c_dim = self.config.data_info[2]
        self.p_dim = self.config.data_info[3]
        self.n_dim = self.config.data_info[4]
        self.num_conv_layer = self.config.num_conv_layer
        self.num_res_block_flow = self.config.num_res_block_flow
        self.num_res_block_pixel = self.config.num_res_block_pixel
        self.num_scale = self.config.num_scale
        self.num_convlstm_block = self.config.num_convlstm_block
        self.num_convlstm_scale = self.config.num_convlstm_scale
        self.num_dis_conv_layer = self.config.num_dis_conv_layer
        self.moving_weight = self.config.moving_weight
        self.norm_type = self.config.norm_type
        self.gan_type = self.config.gan_type
        self.dataset_type = self.config.dataset_type
        self.local_confidence_weight = self.config.local_confidence_weight

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.input_height, self.input_width, self.c_dim],
        )

        self.camera_pose = tf.placeholder(
            name='camera_pose', dtype=tf.float32,
            shape=[self.batch_size, self.p_dim, self.n_dim],
        )

        self.step = tf.placeholder(
            name='step', dtype=tf.int32,
            shape=[],
        )

        self.is_train = tf.placeholder(
            name='is_train', dtype=tf.bool,
            shape=[],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=True):
        fd = {
            self.image: batch_chunk['image'],  # [B, h, w, c]
            self.camera_pose: batch_chunk['camera_pose'],  # [B, h, w, c]
            self.step: step
        }
        # if is_training is not None:
        fd[self.is_train] = is_training

        return fd

    def build(self, is_train=True):

        c = int(self.c_dim / self.n_dim)
        num_res_block_flow = self.num_res_block_flow
        num_res_block_pixel = self.num_res_block_pixel
        num_prior = self.n_dim - 1
        num_scale = self.num_scale
        # compare with baselines
        rescale = 1 if self.dataset_type == 'scene' else 1.5

        def Pose_Encoder(input_pose, target_pose, scope='Pose_Encoder', reuse=False):
            with tf.variable_scope(scope) as scope:
                if not reuse: log.warn(scope.name)
                _ = target_pose - input_pose
                if not reuse: log.info('{} {}'.format(
                    _.name, _.get_shape().as_list()))
                return _

        def Encoder(input_image, pose, scope='Encoder', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)
                _ = add_image_coord(input_image)

                # Pose stack
                h, w = int(_.get_shape()[1]), int(_.get_shape()[2])
                pose = tf.reshape(pose, [self.batch_size, 1, 1, -1])
                pose = tf.tile(pose, [1, h, w, 1], name='pose_map')
                _ = tf.concat([_, pose], axis=-1, name='concat_pose_img')
                if not reuse: log.info('{} {}'.format(
                    _.name, _.get_shape().as_list()))

                # Conv
                all_output = []
                for i in range(self.num_conv_layer):
                    _ = conv2d(_, 2 ** (i+5), is_train, k_h=4, k_w=4,
                               info=not reuse, norm=self.norm_type,
                               name='conv{}'.format(i+1))
                    all_output.append(_)
                return all_output

        def Joint_Decoder(pixel_input, flow_input, input_image, scope='Joint_Decoder', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)
                assert pixel_input[-1].get_shape() == flow_input[-1].get_shape()

                _p = pixel_input[-1]
                _f = flow_input[-1]

                with tf.variable_scope('Pixel'):
                    # Res blocks
                    ch = int(_p.get_shape()[-1])
                    for i in range(int(num_res_block_pixel)):
                        _p = residual_block(
                            _p, ch, is_train, info=not reuse,
                            norm=self.norm_type, name='pixel_R{}_{}'.format(ch, i))

                with tf.variable_scope('Flow'):
                    # Res blocks
                    ch = int(_f.get_shape()[-1])
                    for i in range(int(num_res_block_flow)):
                        _f = residual_block(
                            _f, ch, is_train, info=not reuse,
                            norm=self.norm_type, name='flow_R{}_{}'.format(ch, i))

                # Deconv
                pixel_output = None
                flow_output = None
                pixel_output_list = []
                flow_output_list = []
                x_list = []
                y_list = []
                pixel_mask_list = []
                flow_mask_list = []
                num_deconv_layer = int(np.log2(self.input_height/int(pixel_input[-1].get_shape()[1])))
                num_channel = [256, 128, 64, 32, 16, 8]
                for i in range(num_deconv_layer):

                    with tf.variable_scope('Flow'):
                        _f = deconv2d(_f, [num_channel[i], 3, 2],
                                      is_train, info=not reuse,
                                      norm=self.norm_type, name='flow_deconv{}'.format(i+1))

                    with tf.variable_scope('Pixel'):
                        # skip connection
                        if not (num_res_block_pixel == 0 and i == 0):
                            _p = tf.concat([_p, pixel_input.pop(-1)], axis=-1)
                        else:
                            pixel_input.pop(-1)
                        if not reuse: log.info('pixel_deconv{}_in_layer_concat {}'.format(
                            i+1, _p.get_shape().as_list()))
                        _p = deconv2d(_p, [num_channel[i], 3, 2],
                                      is_train, info=not reuse,
                                      norm=self.norm_type, name='pixel_deconv{}'.format(i+1))

                    if num_deconv_layer - i <= num_scale:
                        with tf.variable_scope('Flow'):
                            flow_output = deconv2d(_f, [int(num_channel[i]/2), 3, 1],
                                                   is_train, info=not reuse,
                                                   norm=self.norm_type,
                                                   name='flow_deconv{}_out_layer_1'.format(i+1))
                            flow_output = deconv2d(flow_output, [3, 3, 1],
                                                   is_train, info=not reuse,
                                                   norm='None',
                                                   activation_fn=None,
                                                   name='flow_deconv{}_out_layer_2'.format(i+1))

                            # bilinear sample: flow -> img
                            x, y = flow_output[:, :, :, 0], flow_output[:, :, :, 1]
                            flow_mask = flow_output[:, :, :, -1]
                            h = int(flow_output.get_shape()[1])
                            w = int(flow_output.get_shape()[2])
                            x_g = tf.convert_to_tensor(
                                np.expand_dims(2 * np.array(range(w))/w - 1, axis=0),
                                dtype=tf.float32
                            )
                            y_g = tf.convert_to_tensor(
                                np.expand_dims(2 * np.array(range(h))/h - 1, axis=1),
                                dtype=tf.float32
                            )
                            flow_output_img = bilinear_sampler(
                                tf.image.resize_nearest_neighbor(
                                    (input_image+1)/2, [h, w]),
                                x + x_g, y + y_g
                            )*2-1
                            flow_output_list.append(flow_output_img)
                            x_list.append(x)
                            y_list.append(y)
                            flow_mask_list.append(flow_mask)

                        with tf.variable_scope('Pixel'):
                            if i == num_deconv_layer-1:
                                pixel_output = deconv2d(_p, [int(num_channel[i]/2), 3, 1],
                                                        is_train, info=not reuse,
                                                        norm=self.norm_type,
                                                        name='pixel_deconv{}_out_layer_1'.format(i+1))
                                pixel_output = deconv2d(pixel_output, [c+1, 3, 1],
                                                        is_train, info=not reuse,
                                                        norm='None',
                                                        activation_fn=None,
                                                        name='pixel_deconv{}_out_layer_2'.format(i+1))
                                pixel_mask = pixel_output[:, :, :, -1]
                                pixel_output = tf.tanh(pixel_output[:, :, :, :c])
                            else:
                                log.error('pixel dummy output')
                                pixel_output = tf.zeros_like(flow_output)
                                pixel_mask = tf.zeros_like(flow_mask)
                            pixel_mask_list.append(pixel_mask)
                            pixel_output_list.append(pixel_output)

                        _f = tf.concat([_f, flow_output], axis=-1)
                        _p = tf.concat([_p, pixel_output], axis=-1)
                        if not reuse: log.info('flow_deconv{}_out_layer_concat {}'.format(
                            i+1, _f.get_shape().as_list()))
                        if not reuse: log.info('pixel_deconv{}_out_layer_concat {}'.format(
                            i+1, _p.get_shape().as_list()))

            return pixel_output_list, pixel_mask_list, flow_output_list, \
                flow_mask_list, x_list, y_list

        def D(img, scope='Discriminator', reuse=False):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)
                _ = img
                d_feature = []
                num_channel = [32, 64, 128, 256, 384, 512, 512]
                k = [4, 4, 4, 4, 4, 4, 4, 4]
                s = [2, 2, 2, 2, 2, 2, 2, 4]
                for i in range(self.num_dis_conv_layer-1):
                    _ = conv2d(_, num_channel[i], is_train, k_h=k[i], k_w=k[i],
                               s=s[i], info=not reuse,
                               norm=self.norm_type, name='conv{}'.format(i+1))
                    d_feature.append(_)
                _ = conv2d(_, 1, is_train, k_h=1, k_w=1, s=1, info=not reuse,
                           norm=self.norm_type, activation_fn=None,
                           name='conv{}'.format(i+2))
                return tf.nn.sigmoid(_), _, d_feature[:4]

        # Input {{{
        # =========
        input_image = []
        input_pose = []
        for i in range(num_prior):
            input_image.append(self.image[:, :, :, (i+1)*c:(i+2)*c])
            input_pose.append(self.camera_pose[:, :, i+1])
        target_image, target_pose = \
            self.image[:, :, :, :c], self.camera_pose[:, :, 0]
        # }}}

        # Graph {{{
        # =========
        pixel_output_list = []
        pixel_final_output_list = []
        pixel_mask_list = []
        flow_output_list = []
        flow_final_output_list = []
        flow_mask_list = []
        x_list = []
        y_list = []
        aggregate_output_list = []
        aggregate_final_output_list = []
        aggregate_flow_only_output_list = []
        aggregate_final_flow_only_output_list = []
        image_input_list = []

        for t in range(num_prior):
            encoder_pose_input = Pose_Encoder(input_pose[t], target_pose, reuse=t > 0)

            # Pixel Module {{{
            # ================
            # Encoder
            pixel_encoder_all_output = Encoder(input_image[t],
                                               encoder_pose_input,
                                               scope='Pixel_Encoder',
                                               reuse=t > 0)
            # convlstm
            if t == 0:
                h_list = []
                for s in range(self.num_convlstm_scale):
                    h_list_s = []
                    for i in range(self.num_convlstm_block):
                        h_list_s.append((None, None))
                    h_list.append(h_list_s)

            for s in range(self.num_convlstm_scale):
                layer_idx = len(pixel_encoder_all_output)-s-1
                x = pixel_encoder_all_output[layer_idx]

                for i in range(self.num_convlstm_block):
                    x, h_list[s][i] = resnet_block_convlstm(
                        x, h_list[s][i],
                        norm=self.norm_type,
                        name="Pixel_res_block_convlstm_s{}b{}".format(s+1, i+1),
                        reuse=t > 0)
                pixel_encoder_all_output[layer_idx] = x
            # }}}

            # Flow Module {{{
            # ================
            # Encoder
            flow_encoder_all_output = Encoder(input_image[t],
                                              encoder_pose_input,
                                              scope='Flow_Encoder',
                                              reuse=t > 0)
            # }}}

            # Joint Decoder {{{
            # =================
            pixel_decoder_output, pixel_mask_list_t, \
                flow_decoder_output, flow_mask_list_t, \
                x_list_t, y_list_t = \
                Joint_Decoder(pixel_encoder_all_output,
                              flow_encoder_all_output,
                              input_image[t], reuse=t > 0)

            for p in pixel_decoder_output:
                pixel_output_list.append(p)
                image_input_list.append(input_image[t])
            pixel_final_output_list.append(pixel_decoder_output[-1])

            for f in flow_decoder_output:
                flow_output_list.append(f)
            flow_final_output_list.append(flow_decoder_output[-1])

            for x in x_list_t:
                x_list.append(x)

            for y in y_list_t:
                y_list.append(y)

            for mp in pixel_mask_list_t:
                pixel_mask_list.append(mp)

            for mf in flow_mask_list_t:
                flow_mask_list.append(mf)
            # }}}

        # Aggregate {{{
        # =============
        all_mask_list = []
        flow_only_mask_list = []
        for t in range(num_prior):
            for s in range(num_scale):
                # produce mask by softmax
                idx = [v * num_scale + s for v in range(num_prior)][:t+1]
                all_mask = tf.concat([
                    tf.expand_dims(pixel_mask_list[t*num_scale+s], axis=-1),
                    tf.stack([flow_mask_list[v] for v in idx], axis=-1)], axis=-1)
                flow_only_mask = all_mask[:, :, :, 1:]
                # softmax
                all_mask = tf.nn.softmax(all_mask)
                flow_only_mask = tf.nn.softmax(flow_only_mask)
                all_mask_list.append(all_mask)
                flow_only_mask_list.append(flow_only_mask)

                # apply mask
                all_image = tf.concat([
                    tf.expand_dims(pixel_output_list[t*num_scale+s], axis=-1),
                    tf.stack([flow_output_list[v] for v in idx], axis=-1)], axis=-1)
                aggregate_output = tf.reduce_sum(tf.expand_dims(all_mask, axis=-2) * all_image, axis=-1)
                aggregate_output_list.append(aggregate_output)

                flow_only_image = all_image[:, :, :, :, 1:]
                aggregate_flow_only_output = tf.reduce_sum(
                    tf.expand_dims(flow_only_mask, axis=-2) * flow_only_image, axis=-1)
                aggregate_flow_only_output_list.append(aggregate_flow_only_output)
            aggregate_final_output_list.append(aggregate_output)
            aggregate_final_flow_only_output_list.append(aggregate_flow_only_output)
        train_test_summary('energy/pixel_mask_energy', tf.reduce_sum(all_mask[:, :, :, 0]))
        train_test_summary('energy/flow_mask_energy', tf.reduce_sum(all_mask[:, :, :, 1:]/num_prior))
        # collect softmax masks
        pixel_softmax_mask_list = []
        flow_softmax_mask_list = []
        for t in range(num_prior):
            for s in range(num_scale):
                pixel_softmax_mask_list.append(
                    all_mask_list[t*num_scale+s][:, :, :, 0])
        for s in range(num_scale):
            for t in range(num_prior):
                flow_softmax_mask_list.append(
                    all_mask_list[num_prior*num_scale-s-1][:, :, :, t+1])
        # }}}

        # Build loss and self.accuracy {{{
        # weights for multi-scale flow preidictions
        if self.moving_weight == 'uniform':
            weights = np.ones(num_scale).astype(np.float32)
        elif self.moving_weight == 'shift':
            weights_deep = 2**np.flip(np.array(range(1, num_scale+1)).astype(np.float32), axis=0)
            weights_shallow = 2**np.array(range(1, num_scale+1)).astype(np.float32)
            weights_ratio = tf.reduce_max(
                tf.stack([(tf.convert_to_tensor(3e4) - tf.cast(self.step, tf.float32)) /
                          tf.convert_to_tensor(3e4), 0], axis=0))
            weights = weights_ratio*weights_deep + (1-weights_ratio)*weights_shallow
            weights /= tf.reduce_sum(weights)
        elif self.moving_weight == 'step':
            step_size = 5e3
            weights = tf.to_float(tf.greater_equal(
                self.step,
                tf.cast(tf.convert_to_tensor(
                    np.array(range(num_scale))*step_size), tf.int32),
            ))

        if self.debug:
            train_test_summary("scale_weights",
                               tf.reshape(tf.convert_to_tensor(weights), [1, 1, num_scale, 1]),
                               max_outputs=1, summary_type='image')

        # Discriminator graph
        d_real, d_real_logits, d_real_feature = D(target_image, reuse=False)
        if self.config.gan_type == 'ls':
            d_real = d_real_logits
        d_fake_list = []
        d_fake_logits_list = []
        d_fake_feature_list = []
        # Input of the discriminator
        for i in range(len(pixel_final_output_list)):
            p = pixel_final_output_list[i]
            d_fake, d_fake_logits, d_fake_feature = D(p, reuse=True)
            if self.config.gan_type == 'ls':
                d_fake = d_fake_logits
            d_fake_list.append(d_fake)
            d_fake_logits_list.append(d_fake_logits)
            d_fake_feature_list.append(d_fake_feature)

        # dis loss
        if self.config.gan_type == 'normal':
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_real_logits, labels=tf.ones_like(d_real)))
        elif self.config.gan_type == 'ls':
            d_loss_real = tf.reduce_mean(
                (d_real_logits - tf.ones_like(d_real)) ** 2)
        d_loss_fake = 0
        g_loss = 0
        for i in range(len(d_fake_list)):
            d_fake = d_fake_list[i]
            d_fake_logits = d_fake_logits_list[i]
            if self.config.gan_type == 'normal':
                d_loss_fake += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_fake_logits, labels=tf.zeros_like(d_fake)))
                g_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_fake_logits, labels=tf.ones_like(d_fake)))
            elif self.config.gan_type == 'ls':
                d_loss_fake += tf.reduce_mean(
                    (d_fake_logits - tf.zeros_like(d_fake)) ** 2)
                g_loss += tf.reduce_mean(
                    (d_fake_logits - tf.ones_like(d_fake)) ** 2)
            else:
                raise ValueError('Undefined type of GAN: {}'.format(self.config.gan_type))
        self.d_loss = d_loss_real + d_loss_fake/num_prior
        self.g_loss = g_loss/num_prior
        # Add g loss to the pixel module later
        train_test_summary("gan_loss/d_loss", self.d_loss)
        train_test_summary("gan_loss/g_loss", g_loss/num_prior)
        train_test_summary("gan_loss/d_loss_real", d_loss_real)
        train_test_summary("gan_loss/d_loss_fake", d_loss_fake)
        train_test_summary("gan_loss/d_real_prob", tf.reduce_mean(d_real))
        train_test_summary("gan_loss/d_fake_prob",
                           tf.reduce_mean(tf.stack(d_fake_list)))

        # Build loss {{{
        # ==============
        self.eval_loss = {}

        def build_loss(output_list, weights, loss_name, scale=False):
            total_loss = 0
            # l1 loss
            l1_loss = 0
            for i in range(len(output_list)):
                current_weight = weights[i % num_scale] if scale else 1
                img = output_list[i]
                l1_loss += tf.reduce_mean(tf.abs(
                    img -
                    tf.image.resize_images(
                        target_image,
                        [int(img.get_shape()[1]), int(img.get_shape()[2])]
                    ))) * current_weight
            l1_loss = l1_loss / len(output_list)
            train_test_summary("{}_loss/l1_loss".format(loss_name), l1_loss)
            self.eval_loss['{}_l1_loss'.format(loss_name)] = l1_loss

            total_loss += l1_loss

            # Report testing loss
            idx = [v * num_scale + num_scale - 1 for v in range(num_prior)]
            if loss_name == 'flow':
                all_final_scale_pred = tf.stack([output_list[v] for v in idx], axis=-1)
                for i in range(all_final_scale_pred.shape[-1]):
                    self.eval_loss['{}_avg_report_loss_{}'.format(loss_name, i)] = \
                        tf.reduce_mean(tf.abs(all_final_scale_pred[:, :, :, :, :i+1] -
                                              tf.expand_dims(target_image, axis=-1)))
            else:
                for i in idx:
                    if self.dataset_type == 'object':
                        ssim = tf_ssim(1-(output_list[i]+1)/2,
                                       1-(target_image+1)/2, mean_metric=False)
                    else:
                        ssim = tf_ssim((output_list[i]+1)/2,
                                       (target_image+1)/2, mean_metric=False)
                    report_loss = tf.reduce_mean(
                        tf.abs(output_list[i] - target_image))*rescale
                    report_ssim = tf.reduce_mean(ssim)
                    self.eval_loss['{}_report_loss_{}'.format(loss_name, i)] = report_loss
                    self.eval_loss['{}_report_ssim_{}'.format(loss_name, i)] = report_ssim
                    train_test_summary("{}_loss/report_loss_{}".format(
                        loss_name, i), report_loss)
                    train_test_summary("{}_loss/report_ssim_{}".format(
                        loss_name, i), report_ssim)
                    # plot the last one on Tensorboard: final_report loss
                train_test_summary("final_report/{}_loss".format(
                    loss_name), report_loss)
                train_test_summary("final_report/{}_ssim".format(
                    loss_name), report_ssim)

            train_test_summary("{}_loss/total_loss".format(loss_name), total_loss)
            self.eval_loss['{}_total_loss'.format(loss_name)] = total_loss
            if num_prior > 1:
                improvement = tf.reduce_mean(
                    tf.abs(output_list[num_scale-1] if scale
                           else output_list[0] - target_image) - tf.abs(output_list[-1] - target_image))
                train_test_summary("{}_loss/improvement".format(loss_name), improvement)
                self.eval_loss['{}_improvement'.format(loss_name)] = improvement
            return total_loss

        def build_local_confidence_loss(
            output_list, mask_list, weights,
            loss_name, scale=False, regularizer_weight=1e-2
        ):
            # l1 loss
            l1_loss = 0
            for i in range(len(output_list)):
                normalized_mask = tf.reshape(tf.nn.l2_normalize(
                    tf.reshape(mask_list[i], [self.batch_size, -1]), dim=-1),
                    [self.batch_size, self.input_height, self.input_width])
                current_weight = weights[i % num_scale] if scale else 1
                img = output_list[i]
                loss_map = tf.reduce_mean(tf.abs(
                    img - tf.image.resize_images(
                        target_image,
                        [int(img.get_shape()[1]), int(img.get_shape()[2])])
                ), axis=-1)
                l1_loss += tf.reduce_mean(loss_map * normalized_mask) * current_weight / \
                    (int(img.get_shape()[1]) * int(img.get_shape()[2])) * regularizer_weight
            l1_loss = l1_loss / len(output_list)
            return l1_loss

        pixel_loss = build_loss(
            pixel_final_output_list,
            weights, 'pixel')

        flow_loss = build_loss(
            flow_output_list, weights,
            'flow', num_scale > 1)

        flow_only_aggreate_loss = build_loss(
            aggregate_flow_only_output_list, weights,
            'flow_only_aggregate', num_scale > 1)

        aggregate_loss = build_loss(
            aggregate_final_output_list,
            tf.reshape(tf.tile(tf.reshape(weights[-1], [1, -1]), [1, num_scale]), [-1]),
            'aggregate', num_scale > 1)

        # oracle baseline: pick the best of pixel or flow
        all_output_stack = tf.stack(
            [v for v in pixel_final_output_list+flow_final_output_list], axis=-1)
        all_output_stack_loss = tf.reduce_mean(tf.abs(
            all_output_stack - tf.expand_dims(target_image, axis=-1)), axis=[1, 2, 3])*rescale
        all_output_stack_loss_min = tf.reduce_mean(
            tf.reduce_min(all_output_stack_loss, axis=-1))
        all_output_stack_ssim = []
        self.eval_loss['best_of_pixel_of_flow_report_loss'] = all_output_stack_loss_min
        for i in range(all_output_stack.get_shape().as_list()[-1]):
            if self.dataset_type == 'object':
                ssim = tf_ssim(1-(all_output_stack[:, :, :, :, i]+1)/2,
                               1-(target_image+1)/2, mean_metric=False)
            else:
                ssim = tf_ssim((all_output_stack[:, :, :, :, i]+1)/2,
                               (target_image+1)/2, mean_metric=False)
            all_output_stack_ssim.append(tf.reduce_mean(ssim, axis=[1, 2, 3]))
        all_output_stack_ssim = tf.stack(all_output_stack_ssim, axis=-1)
        all_output_stack_ssim_max = tf.reduce_mean(
            tf.reduce_max(all_output_stack_ssim, axis=-1))
        self.eval_loss['best_of_pixel_of_flow_report_ssim'] = all_output_stack_ssim_max

        # local confidence
        pixel_mask_loss = build_local_confidence_loss(
            pixel_output_list, pixel_softmax_mask_list, weights, 'pixel',
            regularizer_weight=self.local_confidence_weight)
        flow_mask_loss = build_local_confidence_loss(
            flow_output_list, flow_softmax_mask_list, weights, 'flow',
            num_scale > 1, regularizer_weight=self.local_confidence_weight)

        # sum up all the losses
        # pixel loss = pixel module loss + global confidence + local confidence
        self.pixel_loss = 10*(pixel_loss + aggregate_loss + pixel_mask_loss)
        # pixel loss with GAN = pixel loss + GAN loss
        self.pixel_loss_gan = self.pixel_loss + self.g_loss
        # flow loss = flow module loss + global confidence
        self.flow_loss = flow_loss + aggregate_loss + flow_mask_loss
        self.output = []
        self.loss = self.pixel_loss + self.flow_loss

        self.pixel_mask_list = pixel_softmax_mask_list
        self.flow_mask_list = flow_softmax_mask_list
        self.all_mask_list = all_mask_list

        train_test_summary("loss_summary/pixel_loss", self.pixel_loss)
        train_test_summary("loss_summary/flow_loss", self.flow_loss)
        # }}}

        self.eval_img = {}
        if self.dataset_type == 'object':
            target_image_vis = 1 - target_image
            flow_only_output_vis = 1 - aggregate_final_flow_only_output_list[-1]
        else:
            target_image_vis = target_image
            flow_only_output_vis = aggregate_final_flow_only_output_list[-1]

        self.display_image = tf.concat([
            tf.ones_like(target_image),
            tf.ones_like(target_image),
            tf.ones_like(target_image),
            target_image_vis,
            flow_only_output_vis,
            target_image_vis,
            tf.ones_like(target_image),
            target_image_vis,
        ], axis=1)

        for i in range(len(pixel_output_list)):
            input_vis = image_input_list[i]
            pixel_vis = tf.image.resize_nearest_neighbor(
                pixel_output_list[i], [self.input_height, self.input_width])
            flow_vis = tf.image.resize_nearest_neighbor(
                flow_output_list[i], [self.input_height, self.input_width])
            aggregate_vis = tf.image.resize_nearest_neighbor(
                aggregate_output_list[i], [self.input_height, self.input_width])
            pixel_mask_vis = tf.image.resize_nearest_neighbor(
                tf.tile(tf.expand_dims(
                    all_mask_list[i][:, :, :, 0], axis=-1), [1, 1, 1, c]),
                [self.input_height, self.input_width])
            scale_idx = i % num_scale - num_scale
            prior_idx = int(i / num_scale)
            flow_mask_vis = tf.image.resize_nearest_neighbor(
                tf.tile(tf.expand_dims(
                    all_mask_list[scale_idx][:, :, :, prior_idx+1], axis=-1), [1, 1, 1, c]),
                [self.input_height, self.input_width])
            x_vis = tf.image.resize_nearest_neighbor(
                tf.tile(tf.expand_dims(x_list[i], axis=-1), [1, 1, 1, c]),
                [self.input_height, self.input_width]
            )
            y_vis = tf.image.resize_nearest_neighbor(
                tf.tile(tf.expand_dims(y_list[i], axis=-1), [1, 1, 1, c]),
                [self.input_height, self.input_width]
            )

            if self.dataset_type == 'object':
                input_vis = 1 - input_vis
                pixel_vis = 1 - pixel_vis
                flow_vis = 1 - flow_vis
                aggregate_vis = 1 - aggregate_vis
            vis = tf.concat([input_vis, x_vis, y_vis, flow_vis, flow_mask_vis,
                             pixel_vis, pixel_mask_vis, aggregate_vis], axis=1)
            self.display_image = tf.concat([self.display_image, vis], axis=2)

        # discriminator output visualization
        d_output_vis = tf.image.resize_nearest_neighbor(
            d_real, [self.input_height, self.input_width])
        for d_fake in d_fake_list:
            d_output_vis = tf.concat(
                [d_output_vis,
                 tf.zeros([self.batch_size, self.input_height,
                           self.input_width*(num_scale-1), 1]),
                 tf.image.resize_nearest_neighbor(
                     d_fake, [self.input_height, self.input_width])
                 ], axis=2)
        d_output_vis = tf.tile(d_output_vis, [1, 1, 1, 3])
        self.display_image = tf.concat([self.display_image, d_output_vis], axis=1)

        self.eval_img['display'] = tf.clip_by_value(
            self.display_image, 0 if self.dataset_type == 'object' else -1, 1)

        # multi-scale weight training visualization
        if self.moving_weight == 'shift' or self.moving_weight == 'step':
            repeat_weights = tf.reshape(tf.tile(tf.reshape(weights, [1, -1]), [1, num_prior]), [-1])
            weight_vis = tf.reshape(tf.convert_to_tensor(
                tf.concat([tf.zeros([1]), repeat_weights], axis=0)),
                [1, 1, num_scale*num_prior+1, 1])
            weight_vis = tf.tile(weight_vis, [self.batch_size, 1, 1, 3])
            weight_vis = tf.image.resize_nearest_neighbor(
                weight_vis,
                [self.input_height, self.input_width*(num_scale*num_prior+1)]
            )
            self.display_image = tf.concat([self.display_image, weight_vis], axis=1)

        self.display_image = tf.clip_by_value(
            self.display_image, 0 if self.dataset_type == 'object' else -1, 1)

        # adjust the ratio of image summary
        [_, h, w, _] = self.display_image.get_shape().as_list()
        ratio_threshold = 1.6
        if float(h)/w > ratio_threshold:
            adjust_w = int(h/ratio_threshold)
            dummy_image = tf.ones([self.batch_size, h, adjust_w-w, c])
            self.display_image = tf.concat([self.display_image, dummy_image], axis=2)
        elif float(w)/h > ratio_threshold:
            adjust_h = int(w/ratio_threshold)
            dummy_image = tf.ones([self.batch_size, adjust_h-h, w, c])
            self.display_image = tf.concat([self.display_image, dummy_image], axis=1)

        train_test_summary("img_summary", self.display_image, max_outputs=4, summary_type='image')
        print('\033[93mSuccessfully loaded the model.\033[0m')
