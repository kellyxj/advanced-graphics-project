# Advanced Graphics Project: NeRFs
In this tutorial, implements NeRF from first principles. Mitsuba is used to provide an interface for specifying camera extrinsics.

## What is a NeRF?

A neural radiance field is a way of representing a scene in support of [novel view synthesis](https://en.wikipedia.org/wiki/View_synthesis). The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use [volume rendering](https://en.wikipedia.org/wiki/Volume_rendering) to render new views. The radiance field can be optimized by gradient descent, providing a straightforward and robust improvement over [multi-view stereo](https://carlos-hernandez.org/papers/fnt_mvs_2015.pdf).

Power point slides:
[https://docs.google.com/presentation/d/1-yWFlR1S8QoORCyof0gXyeNREdqYPdd8KghILn4fpCA](https://docs.google.com/presentation/d/1-yWFlR1S8QoORCyof0gXyeNREdqYPdd8KghILn4fpCA)

This presentation was given for the advanced graphics seminar. This class had a lot of students with an interest in neural networks, so it treats NeRF from a deep learning perspective.

[https://docs.google.com/presentation/d/10__u8DagReH-SO_2vzRXxDx3FVcuOv0gvLUTrkZywLg](https://docs.google.com/presentation/d/10__u8DagReH-SO_2vzRXxDx3FVcuOv0gvLUTrkZywLg)

This presentation was given for computational optics. It discusses NeRF as a solver for the rendering equation.

## Dependencies:
Use your preferred package manager (e.g. conda or pip) to install
* numpy
* pytorch
* matplotlib
* Mitsuba
* jupyterlab
* ipywidgets
The recommended setup is to clone this repository and run tutorial.ipynb locally using VSCode's jupyter notebook integration.

Instructions for installing pip https://pip.pypa.io/en/stable/installation/
Instructions for installing pytorch https://pytorch.org/get-started/locally/. Select CUDA if you have a NVIDIA GPU. Select CPU otherwise

## Extending this project:
This project can be easily extended. Try adding an alternative NeRF class to models.py or importing new scene XML files from the Mitsuba website to the "scenes"
directory of this repository.
