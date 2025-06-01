# Real-time Domain Adaptation in Semantic Segmentation

This work explores unsupervised domain adaptation (UDA) for real-time seman tic segmentation using PIDNet as a lightweight backbone and the LoveDA benchmark for evaluation. We quantify the performance degradation when transferring from urban to rural scenes, test augmentation strategies as zero-cost remedies, and implement two efficient UDA methods: adversarial feature alignment and DACS-style image mixing. Furthermore we compare our results with others real-time segmentation models.

Our main contributions are:
1. We provide a quantitative analysis of domain shift for
real-time semantic segmentation fot different models
on the LoveDA benchmark, unveiling a large mIoU
drop when transferring from urban to rural imagery.

2. We show that carefully chosen data augmentations al-
ready reduce the gap with zero runtime overhead.

3. We adapt two popular UDA paradigms—adversarial
feature/segmentation map alignment and DACS image
mixing—to the real-time regime, preserving the infer-
ence speed of PIDNet while recovering part of the lost
accuracy.
