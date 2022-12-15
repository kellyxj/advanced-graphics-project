# Advanced Graphics Project: NeRFs
In this tutorial, we implemented a simple NeRf using Mitsuba. This implementation optimizes a neural representation for a single 
scene and rendering new views.

## What is a NeRF?

A neural radiance field is a simple fully connected network trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Power point slides:
[https://docs.google.com/presentation/d/1-yWFlR1S8QoORCyof0gXyeNREdqYPdd8KghILn4fpCA/edit#slide=id.p](https://docs.google.com/presentation/d/1-yWFlR1S8QoORCyof0gXyeNREdqYPdd8KghILn4fpCA/edit#slide=id.p)
This presentation was given for the advanced graphics seminar. This class had a lot of students with a deep learning background

[https://docs.google.com/presentation/d/10__u8DagReH-SO_2vzRXxDx3FVcuOv0gvLUTrkZywLg/slide=id.p](https://docs.google.com/presentation/d/10__u8DagReH-SO_2vzRXxDx3FVcuOv0gvLUTrkZywLg/slide=id.p)
This presentation was given for computational optics. 

In our tutorial, we demonstrate how to build a simple NeRF along with some demonstrations of the training process. 

## Dependencies:
Use your preferred package manager (e.g. conda or pip) to install
* numpy
* pytorch
* matplotlib
* Mitsuba

Instructions for installing pip https://pip.pypa.io/en/stable/installation/
Instructions for installing pytorch https://pytorch.org/get-started/locally/. Select CUDA if you have a NVIDIA GPU. Select CPU otherwise

## Extending this project:
This project can be easily extended. Try adding an alternative NeRF class to models.py or importing new scene XML files from the Mitsuba website to the "scenes"
directory of this repository.
