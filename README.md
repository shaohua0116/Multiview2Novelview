# Multi-view to Novel view:Synthesizing novel views with Self-Learned Confidence 

## Descriptions
This project is a TensorFlow implementation of [**Multi-view to Novel view:Synthesizing novel views with Self-Learned Confidence**](https://shaohua0116.github.io/Multiview2Novelview/), which is published in ECCV 2018. We provide codes, datasets, and checkpoints. 

In this work, we address the task of **multi-view novel view synthesis**, where we are interested in synthesizing a target image with an arbitrary camera pose from given source images. We propose an end-to-end trainable framework that learns to exploit multiple viewpoints to synthesize a novel view without any 3D supervision. Specifically, our model consists of a **flow prediction module** (*flow predictor)* and a **pixel generation module** (*recurrent pixel generator*) to directly leverage information presented in source views as well as hallucinate missing pixels from statistical priors. To merge the predictions produced by the two modules given multi-view source images, we introduce a **self-learned confidence aggregation mechanism**. An illustration of the proposed framework is as follows.

<p align="center">
    <img src="https://shaohua0116.github.io/Multiview2Novelview/img/model.jpg" height="256"/>
</p>

We evaluate our model on images rendered from 3D object models ([Shapenet](https://www.shapenet.org/)) as well as real and synthesized scenes ([KITTI](http://www.cvlibs.net/datasets/kitti/) and [Synthia](http://synthia-dataset.net/)). We demonstrate that our model is able to achieve state-of-the-art results as well as progressively improve its predictions when more source images are available.

\*This code is still being developed and subject to change.


## Prerequisites

- Python 2.7
- [Tensorflow 1.3.0](https://github.com/tensorflow/tensorflow/tree/r1.0)
- [SciPy](http://www.scipy.org/install.html)
- [NumPy](http://www.numpy.org/)
- [colorlog](https://pypi.python.org/pypi/colorlog)
- [cv2](https://pypi.python.org/pypi/opencv-python)
- [h5py](http://docs.h5py.org/en/latest/build.html#install)
- [Pillow](https://pillow.readthedocs.io/en/latest/installation.html#basic-installation)

## Datasets

### [Shapenet](https://www.shapenet.org/)

### [KITTI](http://www.cvlibs.net/datasets/kitti/)

### [Synthia](http://synthia-dataset.net/)

## Usage

### Train
```bash
$ python trainer.py 
```

### Test
```bash
$ python evaler.py
```

## Results

## Checkpoints
We provide checkpoints and evaluation report files of our models for all experiments.
- ShapeNet Cars
- ShapeNet Chairs
- KITTI
- Synthia

## Related work
- [Multi-view 3D Models from Single Images with a Convolutional Network](https://arxiv.org/abs/1511.06702) in CVPR 2016
- [View Synthesis by Appearance Flow](https://arxiv.org/abs/1605.03557) in ECCV 2016
- [Transformation-Grounded Image Generation Network for Novel 3D View Synthesis](https://arxiv.org/abs/1703.02921) in CVPR 2017
- [Weakly-supervised Disentangling with Recurrent Transformations for 3D View Synthesis](https://arxiv.org/abs/1601.00706) in NIPS 2015

## Cite the paper
If you find this useful, please cite
```
@inproceedings{sun2016multiview,
  title={Multi-view to Novel view: Synthesizing Views via Self-Learned Confidence},
  author={Sun, Shao-Hua and Huh, Minyoung and Liao, Yuan-Hong and Zhang, Ning and Lim, Joseph J},
  booktitle={European conference on computer vision},
  year={2018},
}
```

## Authors
[Shao-Hua Sun](http://shaohua0116.github.io/), [Minyoung Huh](http://minyounghuh.com/), [Yuan-Hong Liao](https://andrewliao11.github.io/), [Ning Zhang](https://people.eecs.berkeley.edu/~nzhang/), and [Joseph J. Lim](http://www-bcf.usc.edu/~limjj/)
