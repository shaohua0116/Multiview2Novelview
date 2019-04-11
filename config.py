import argparse
import os
import numpy as np

from model import Model


def argparser(is_train=True):

    def str2bool(v):
        return v.lower() == 'true'
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='the mini-batch size')
    parser.add_argument('--prefix', type=str, default='default',
                        help='a nickname for the training')
    parser.add_argument('--dataset', type=str, default='car', choices=[
        'car', 'chair', 'kitti', 'synthia'],
        help='you can add your own dataset here')
    parser.add_argument('--num_input', type=int, default=2,
                        help='the number of source images')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='load the latest checkpoint from a directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='load all the parameters including the flow and '
                             'pixel modules and the discriminator')
    parser.add_argument('--checkpoint_p', type=str, default=None,
                        help='load the parameters of the pixel module')
    parser.add_argument('--checkpoint_f', type=str, default=None,
                        help='load the parameters of the flow module')
    parser.add_argument('--checkpoint_g', type=str, default=None,
                        help='load the parameters of both the flow and pixel module')
    parser.add_argument('--checkpoint_d', type=str, default=None,
                        help='load the parameters of the discriminator')
    # Log
    parser.add_argument('--log_step', type=int, default=10,
                        help='the frequency of outputing log info')
    parser.add_argument('--ckpt_save_step', type=int, default=5000,
                        help='the frequency of saving a checkpoint')
    parser.add_argument('--test_sample_step', type=int, default=100,
                        help='the frequency of performing testing inference during training')
    parser.add_argument('--write_summary_step', type=int, default=100,
                        help='the frequency of writing TensorBoard summaries')
    # Learning
    parser.add_argument('--max_steps', type=int, default=1000000,
                        help='the max training iterations')
    parser.add_argument('--learning_rate_p', type=float, default=5e-5,
                        help='the learning rate of the pixel module')
    parser.add_argument('--learning_rate_f', type=float, default=1e-4,
                        help='the learning rate of the flow module')
    parser.add_argument('--learning_rate_d', type=float, default=1e-4,
                        help='the learning rate of the discriminator')
    parser.add_argument('--local_confidence_weight', type=int, default=1e-2,
                        help='the weight of the confidence prediction objective')
    # Architecture
    parser.add_argument('--num_res_block_pixel', type=int, default=0,
                        help='the number of residual block in the bottleneck of the pixel module')
    parser.add_argument('--num_res_block_flow', type=int, default=4,
                        help='the number of residual block in the bottleneck of the flow module')
    parser.add_argument('--num_dis_conv_layer', type=int, default=5,
                        help='the number of convolutional layers of the discriminator')
    parser.add_argument('--num_conv_layer', type=int, default=5,
                        help='the number of convolutional layers of '
                             'the encoder of both the flow and pixel modules')
    parser.add_argument('--num_convlstm_block', type=int, default=2,
                        help='the number of residual ConvLSTM block of the pixel module')
    parser.add_argument('--num_convlstm_scale', type=int, default=3,
                        help='how many innermost layers of the pixel module '
                             'have a residual ConvLSTM connection')
    parser.add_argument('--norm_type', type=str, default='None',
                        choices=['batch', 'instance', 'None'],
                        help='the type of normalization')
    # GAN
    parser.add_argument('--gan_type', type=str, default='ls', choices=['ls', 'normal'],
                        help='the type of GAN losses such as LS-GAN, WGAN, etc')
    parser.add_argument('--gan_start_step', type=int, default=300000,
                        help='start to optimize the GAN loss when the model is stable')
    parser.add_argument('--update_rate', type=int, default=1,
                        help='update G more frequently than D')
    # Multi-scale prediction: this is not reporeted in the paper
    # The main idea is to imporve the flow module by training it to start from
    # predict a coarser flow fields (similar to progressive learning GAN
    # proposed by Karras et al. ICLR 2017)
    parser.add_argument('--num_scale', type=int, default=1,
                        help='the number of multi-scale flow prediction '
                             '(1 means without multi-scale prediction)')
    parser.add_argument('--moving_weight', type=str, default='uniform',
                        choices=['uniform', 'shift', 'step'],
                        help='gradually learn each scale from coarse to fine')   

    # Testing config {{{
    # ========
    # checkpoint
    parser.add_argument('--max_eval_steps', type=int, default=500,
                        help='max steps of randomly sampling testing tuple.'
                             'do not need to specify this when a data list is given')
    parser.add_argument('--data_id_list', type=str, default=None,
                        help='specify a list of data point that you want to evaluate')
    # task type
    parser.add_argument('--loss', type=str2bool, default=True,
                        help='report the loss')
    parser.add_argument('--write_summary', type=str2bool, default=False,
                        help='write the summary of this evaluation as a text file')
    parser.add_argument('--plot_image', type=str2bool, default=False,
                        help='rendered predicted images')
    # write summary file
    parser.add_argument('--quiet', type=str2bool, default=False)
    parser.add_argument('--summary_file', type=str, default='report.txt',
                        help='the path to the summary file')
    parser.add_argument('--output_dir', type=str,
                        help='the output directory of plotted images')
    # }}}

    config = parser.parse_args()

    if config.dataset in ['car', 'chair']:
        config.dataset_type = 'object'
        import datasets.object_loader as dataset
    elif config.dataset in ['kitti', 'synthia']:
        config.dataset_type = 'scene'
        import datasets.scene_loader as dataset

    dataset_train, dataset_test = \
        dataset.create_default_splits(config.num_input, config.dataset)
    image, pose = dataset_train.get_data(dataset_train.ids[0])

    config.data_info = np.concatenate([np.asarray(image.shape), np.asarray(pose.shape)])

    # --- create model ---
    model = Model(config, debug_information=config.debug, is_train=is_train)

    return config, model, dataset_train, dataset_test
