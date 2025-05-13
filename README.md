# ms-thesis

This repository contains the code and related materials for my thesis, completed as part of the M.S. in Computer Science program at Case Western Reserve University.

## Thesis Overview

My thesis research lies at the intersection of **machine learning** and **baseball analytics**, with a particular focus on using **graph neural networks (GNNs)** to model **pitcher-versus-hitter matchups** in Major League Baseball (MLB).

### Research Objective

To develop a graph-based machine learning model that predicts the outcome of an MLB at-bat based on historical data. Each pitcher-hitter matchup is modeled as a temporal, bipartite graph, where:

- Nodes represent **pitchers** and **hitters** with associated features (e.g., handedness, pitch arsenal, batting tendencies).
- Edges represent **individual at-bats**, containing attributes like the date, pitch count, and outcome (e.g., strikeout, walk, groundball).
- The task involves **predicting edge labels** (future at-bat outcomes), accounting for the evolving nature of the graph over time.

This work aims to expand the use of **graph neural networks** in baseball analytics, an area where such models are currently underutilized compared to traditional machine learning methods.

## Repository Structure
    ├── csds446_project/                  
    │   └── Project code from CSDS 446: Machine Learning on Graphs.             
    │   Early exploration of GNNs for modeling baseball matchups.
    ├── (to be expanded with thesis code, models, datasets, and analysis)

## Academic Information 
- **Degree Program**: M.S. in Computer Science on the thesis track with a depth area in Databases & Data Mining
- **University**: Case Western Reserve University
- **Thesis Advisor**: Dr. Kevin Xu
- **Author**: Selah Dean
- **Expected Graduation**: May 2026

