# Facial Feature Detection Tool

## Introduction
This Facial Feature Detection Tool is developed for the Computer Vision course at the University of Jordan. It employs image processing centered approach to detect facial features, and uses the Viola-Jones algorithm for the actual detection.

## System Requirements
* MATLAB (Perferably the latest version)
* Image Processing Toolbox
* Computer Vision Toolbox

## Getting Started
Launch the tool in MATLAB.
Familiarize yourself with the main interface, image input methods, and feature detection options.

## Preprocessing Steps in Image Analysis
The tool includes steps such as grayscale conversion, histogram equalization, image sharpening, edge enhancement, contrast adjustment, Gaussian filtering, and image resizing.

Refer to the user manual for more details.

## Feature Detection

#### Adjustable parameters:
* Resolution Multiplier
* Gaussian Kernel
* Scaling
* Gamma correction
* Face Detection Threshold

#### Select facial features for detection:
* Face
* Eyes
* Mouth
* Nose

#### Detect Tilted Faces
This feature makes the tool recognize tilted or rotated faces; however this is computationaly expensive so it is turned off by default.

#### Fallback (Geometric Analysis)
This feature makes the tool fallback to geometric analysis techniques when the Viola-Jones algorithm fails to detect any features; it is generally less accurate, so it is also turned off by default.

## Troubleshooting
Refer to the user manual for common issues like no features detected or incorrect detection.

<hr>
Users are encouraged to experiment with different settings for optimal results.