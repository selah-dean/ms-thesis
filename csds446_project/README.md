# MLB At-Bat Outcome Prediction using Graph Neural Networks

This repository contains the implementation of Graph Neural Networks (GNNs) to predict the outcomes of Major League Baseball at-bat encounters. The project models pitcher-hitter matchups as a bipartite graph, where pitchers and hitters are represented as nodes, and previous at-bats between them form the edges.

## Project Overview

Baseball analytics has traditionally relied on statistical models and machine learning approaches that treat players in isolation. This project introduces a novel approach by modeling the relational structure between pitchers and hitters using GNNs, capturing the complex dependencies in their interactions.

## Data Sources

The data for this project comes from two main sources:

1. **Baseball Savant**: Player attributes including process statistics for pitchers and hitters

2. **Retrosheet**: Play-by-play data for at-bat outcomes and situational information 

## Model Architecture

The project implements two GNN model variants:

1. **MLBMatchupPredictor**: Base model using GATv2Conv layers in a heterogeneous graph structure
2. **MLBMatchupPredictorTemporal**: Enhanced model incorporating temporal encodings for time-aware predictions

Both models use:
- HeteroConv wrapper to handle heterogeneous graph structure
- GATv2Conv for flexible attention mechanisms
- Multi-head attention to capture different aspects of player relationships
- AttentionEdgeClassifier for outcome prediction