This is the implementation of our paper ``PEdger++: Practical Edge Detection via Assembling Cross Information"

# Introduction
Edge detection, serving as a crucial component in numerous vision-based applications, aims to effectively extract object boundaries and/or salient edges from natural images. To be viable for broad deployment across devices with varying computational capacities, edge detectors shall balance high accuracy with low computational complexity. This paper addresses the challenge of achieving that balance: {how to efficiently capture discriminative features without relying on large-size and sophisticated models}. We propose PEdger++, a collaborative learning framework designed to reduce computational costs and model sizes while improving edge detection accuracy. The core principle of our PEdger++ is that cross-information derived from  heterogeneous  architectures, diverse training moments, and multiple parameter samplings, is beneficial to enhance learning from an ensemble perspective. Extensive ablation studies together with experimental comparisons on the BSDS500, NYUD and Multicue datasets demonstrate the effectiveness of our approach, both quantitatively and qualitatively, showing clear improvements over existing methods.  We also provide multiple versions of the model with varying computational requirements, highlighting PEdger++'s adaptability with respect to different resource constraints.

# Training
Before starting the training process, run:

```python lmdb_io.py```

If you want to train our PEdger++, run:

```python train.py```

# Testing
If you want to test our pre-trained mode, run:

```python test_edge.py```
