# CanSat MHE — Aerial 3D Terrain Estimation

3D terrain reconstruction system developed for the **CSDCMS CanSat Design Challenge**, a Canadian engineering competition where student teams design, build, and launch autonomous miniature satellites.

Our CanSat's secondary mission was to reconstruct the terrain below it from aerial imagery while operating under the compute and sensing constraints of a **Raspberry Pi-based flight computer**.

## Overview

This project adapts **TerrainMesh**, a research model for metric 3D terrain reconstruction from aerial imagery.

TerrainMesh represents terrain as a connected 3D mesh rather than estimating each pixel independently. Image features and sparse geometric information are mapped onto mesh vertices, then a **Graph Neural Network (GNN)** passes information between neighboring vertices to iteratively refine their elevations.

This approach is particularly suited to **structured and urban terrain**, where buildings, roads, and sharp elevation changes benefit from an explicitly geometric representation.

## Adapting TerrainMesh for CanSat

The original model assumes research-quality datasets and substantially more predictable inputs than an autonomous CanSat receives in flight. Our work focused on adapting the pipeline to the deployment environment.

### Training Data & Domain Adaptation

A custom preprocessing and augmentation pipeline was developed to reduce the **domain shift** between existing aerial datasets and imagery captured by the CanSat.

The pipeline includes:

* configurable **Ground Sample Distance (GSD)** processing to better match the scale expected at flight altitude
* paired RGB and DSM preprocessing for model training
* sparse depth/elevation sampling
* motion and optical blur
* degraded camera/image quality
* lighting and weather variation, including cloud effects
* rotations and reflections

These augmentations simulate differences in **camera quality, atmospheric conditions, motion, lighting, altitude, and image scale**, allowing the model to train on inputs closer to those expected during an actual flight.

### Embedded Deployment

The inference pipeline was also adapted for deployment on a **Raspberry Pi**, where CPU, memory, storage, and power are substantially more constrained than the hardware normally used for research models.

The system combines computer vision, graph neural networks, 3D geometry, geospatial data processing, domain adaptation, and embedded machine learning.

## Repository Structure

```text
data/
├── prep/       # RGB/DSM preprocessing, GSD-aware cropping and depth sampling
└── augment/    # Flight-condition and camera augmentations

terrainmesh/    # Adapted TerrainMesh model and inference pipeline
```

## Based On

Adapted from **TerrainMesh: Metric-Semantic Terrain Reconstruction from Aerial Images Using Joint 2D-3D Learning** by Qiaojun Feng and Nikolay Atanasov.
