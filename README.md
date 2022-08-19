# Advanced Graphics Project: NeRFs
In this tutorial, we implemented a simple NeRf using Mitsuba. This implementation optimizes a neural representation for a single 
scene and rendering new views.

## What is a NeRF?

A neural radiance field is a simple fully connected network trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.

In our tutorial, we demonstrate how to build a simple NeRF along with some demonstrations of the training process. 

## Topics covered: 
* SetUp
* Creating multiple sensors and defining a camera
* A brief explaination of the camera matrix
* Training NeRFs
* Implementing differentiable integrators for emissive volumes
* Setting up optimization scenes
* Results

Our tutorial employs widgets for user interaction as well so different parameters can be tested with our data

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
