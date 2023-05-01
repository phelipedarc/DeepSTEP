<img src="https://www.cbpf.br/~icrc2013/images/cbpf_logo.gif"  width="150" />


# Deepstep: A Convolutional Neural Network for Transient Detection in STEP pipeline

Transient astronomical objects, such as supernovae and kilonovae, exhibit variable luminosity, making their identification in the sky a crucial task for astronomers. Traditionally, this identification has been done through visual inspection by trained professionals, aided by image differentiation algorithms. However, this process can produce many false positives, with the most common artifact known as a "Bad Subtraction". To reduce the number of candidates requiring visual inspections and speed up the Transient Detection process on the STEP pipeline, we propose the use of Convolutional Neural Networks (CNNs) to classify the images as Transients or Artifacts (Non-Transients).

In this work, we utilized the family of CNN models known as Mobilenet (Howard et al. 2017) to address this classification problem. Our dataset consists of real transient candidates detected by our pipeline, with each candidate observed using four channels (g, r, i, z) and three different exposure times, resulting in a total of 12 images for each observation. These images are simultaneously fed into the network through independent channels in the format of (51 x 51 x 3), similar to how traditional image analyses are fed into CNNs.

Our CNN, named Deepstep, reduced the amount of Artifacts by 96.9% while maintaining a true positive rate of 92.7%. This means that only 3.1% of the images classified as transient were actual Artifacts, demonstrating the network's ability to filter out artifacts without losing transients. While CNNs have been applied to difference images before (Duev et al. 2019, A. Shandonay et al. 2022), this work is the first to search for transients in the southern area with images from T80S.

## Usage:
This repository provides the code for implementing Deepstep in your pipeline. The repository includes the necessary scripts for preprocessing and testing the CNN.
Note: This repository is **exclusively** for the STEP pipeline usage, thus the trained model is on the CBPF machine:
- STEP/Andre_phelipe/SPLUS/Model_Splus/NEWcorrectSplus.h5

## Credits:
This project was developed by Phelipe Darc and CBPF/SPLUS/STEP contributors.
- email: Phelipedarc@gmail.com
