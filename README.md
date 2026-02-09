# Convolutional Neural Networks - Aerial Landscapes Classification

## Project Overview

This project implements and analyzes convolutional neural networks (CNNs) for classifying aerial landscape images. The work demonstrates understanding of CNN architecture design, the role of inductive bias in deep learning, and the advantages of convolutional layers over fully connected networks for image data.

## Dataset

**Name**: Aerial Landscapes Dataset  
**Classes**: 15 landscape categories
- Agriculture
- Airport
- Beach
- City
- Desert
- Forest
- Grassland
- Highway
- Lake
- Mountain
- Parking
- Port
- Railway
- Residential
- River

**Source**: Local dataset organized in class-based directories  
**Format**: RGB images (various dimensions)

### Dataset Justification

This dataset is appropriate for convolutional neural networks because:

1. **Spatial Structure**: Aerial images contain strong spatial patterns and textures that CNNs are designed to capture
2. **Translation Invariance**: Landscape features can appear at different positions in the image, requiring translation-invariant feature detection
3. **Hierarchical Features**: CNNs can learn hierarchical representations from low-level textures to high-level landscape structures
4. **Color Information**: RGB channels provide crucial information for distinguishing landscape types (e.g., water vs. vegetation vs. urban areas)

## Project Structure

```
neuralNetwork-ai-infrastructure/
├── Aerial_Landscapes/          # Dataset directory
│   ├── Agriculture/
│   ├── Airport/
│   └── ...
├── Convolutional Layers.ipynb  # Main implementation notebook
└── README.md
```

## Implementation Tasks

### 1. Dataset Exploration (EDA) 

- [x] Dataset size and class distribution analysis
- [x] Image dimensions and channels inspection
- [x] Sample visualization from each class
- [x] Preprocessing requirements identification

### 2. Baseline Model (Non-Convolutional)

- [ ] Flatten + Dense architecture implementation
- [ ] Training and validation performance
- [ ] Parameter count analysis
- [ ] Limitations documentation

### 3. Convolutional Architecture Design

- [ ] Custom CNN architecture design
- [ ] Architectural decisions justification:
  - Number of convolutional layers
  - Kernel sizes
  - Stride and padding choices
  - Activation functions
  - Pooling strategy
- [ ] Implementation and training
- [ ] Performance comparison with baseline

## Requirements

```
Python 3.x
tensorflow
numpy
matplotlib
```


## Learning Objectives

This project demonstrates:

- Understanding of convolutional layer mathematics and intuition
- Analysis of how architectural decisions affect learning
- Comparison between convolutional and fully connected layers for image data
- Ability to perform meaningful EDA for neural network tasks
- Clear communication of architectural and experimental decisions

## Author
David Sarria
