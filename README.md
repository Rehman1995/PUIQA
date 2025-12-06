⭐ PUIQA: Physics-Informed Multiscale No-Reference Underwater Image Quality Assessment

Official implementation of PUIQA, a physics-guided and multiscale feature-driven framework for no-reference underwater image quality assessment (NR-UIQA).
This repository provides all components required to reproduce the experiments and results reported in the manuscript.

🔍 Overview

PUIQA introduces a principled quality assessment pipeline that integrates:

Physics-based descriptors
(non-uniform illumination, veiling-light gradient, specular suppression)

Perceptual statistics
(color statistics, edge energy, entropy)

Multiscale features
(Laplacian texture, frequency-domain cues, FPFF descriptors)

Lightweight SVR regression
for robust, interpretable quality prediction.

The method is designed for generalization across underwater domains, handling real and synthetic distortions without requiring reference images.

🚀 Features

✔ Physics-informed feature extraction

✔ Multiscale perceptual and frequency descriptors

✔ SVR-based quality prediction

✔ Ablation tools for analyzing descriptor contributions

✔ Benchmarking scripts for UWIQA and UID2021
<img width="3400" height="1472" alt="Main" src="https://github.com/user-attachments/assets/2b885637-2d56-49c3-9c7b-9465540443b2" />

✔ Reproducible experimental pipeline

✔ Lightweight and suitable for real-time or near real-time integration
