# CSC334bd-Final-Project---Midline-Identification

## Installation

Most packages used in this code will need to be downloaded into the terminal using pip install. 
1. pip install numpy
2. pip install pandas
3. pip install Pillow
4. pip install torch
5. pip install tqdm
6. pip install torch torchvision
7. pip install matplotlib


To run the code if these are installed in the terminal, you can use python Deep-Learning-Code.py

## Purpose

This deep learning model was created in order to begin to automate midline identification in images of zebrafish brains. In the Barresi Lab, we generate data with fourencent tagging or transgenic fish lines that allow us to better understand how the fish brain develops. Ideally the dataset from this code, and the code itself will continue to evolve and be scaled up for use in 3 dimentions with a larger goal to be integrated into existing code within Syglass (an image analysis and visualization platform untilizing virtual reality) to be able to calcualte angle of motion of individual cells toward the midline. 

## Dataset

The dataset used here was created by me, with images sourced from previous and current members of the Barresi Lab, found on the Google Drive. It contains images with different stains/transgenic lines. 

## Current Functionality

The largest drawback of this code right now is not having a suffienct and consistent dataset to train the model on. Many of the images used have different amounts of background noise, as well as be taken from multiple angles. We, as scientists who work with zebrafish, know to identify the shape of the eyes and find the area between then to identify the midline, however this code doesn't know what it is looking at. Thus, it is not particularly accurate, but it does demonstrate capabilities to learn (as loss decreases throughout the epochs), but with more data to train on, the hope is it will get better (which it has as I have expanded the dataset throughout this process).

## Credit

Much of this code is adapted from our Lab 5 project on Data Forgery, and ChatGPT was used to help with this process. Many thanks to Halie Rando for all the support, can't wait to see where this goes from here!
