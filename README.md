# CanSat Terrain Estimation

Onboard terrain reconstruction system developed for the **CSDCMS CanSat Design Challenge**, a Canadian engineering competition in which student teams design, build, and launch a soda-can-sized satellite. Advanced CanSats are launched by rocket to approximately 1 km altitude and must operate autonomously during descent.

## Project

Our CanSat's secondary mission was to estimate the **3D structure of terrain from aerial imagery** captured during flight.

The model is adapted from **TerrainMesh**, a research system for reconstructing metric 3D terrain meshes from aerial RGB images. Rather than predicting an independent depth value for every image pixel, TerrainMesh represents the environment as a connected mesh and refines its geometry using a **Graph Neural Network (GNN)**.

The pipeline combines:

* aerial RGB imagery
* sparse geometric/depth information
* image features projected onto mesh vertices
* graph convolutions between neighboring vertices

The GNN uses this connectivity to refine vertex elevations and produce a coherent **3D terrain surface**. This representation is particularly useful for structured and urban environments, where buildings, roads, and other elevation changes need to be represented geometrically rather than as a simple flat height map.

## Embedded Adaptation

The original TerrainMesh implementation was designed as a research model with substantially more compute available than a CanSat can provide.

For this project, the pipeline was adapted to run **onboard a Raspberry Pi**, requiring changes to make inference practical under embedded hardware constraints.

The goal was to turn a research-oriented computer vision model into an autonomous flight system capable of:

1. capturing aerial imagery during descent,
2. preprocessing images onboard,
3. running terrain estimation with limited CPU, memory, power, and storage,
4. generating a reconstructed terrain representation without relying on cloud computation.

This project combined **computer vision, graph neural networks, 3D geometry, embedded ML, and aerospace systems integration**.

## Based On

This implementation is adapted from **TerrainMesh: Metric-Semantic Terrain Reconstruction from Aerial Images Using Joint 2D-3D Learning** by Qiaojun Feng and Nikolay Atanasov.
