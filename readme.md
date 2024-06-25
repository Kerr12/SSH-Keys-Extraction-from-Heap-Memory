## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Generation](#Data Generation)
- [Proposed Methods](#Proposed Methods)



## Introduction

This research endeavor pertains to the investigation of SSH keys extraction employing deep and machine learning methodologies. A diverse array of approaches is examined to address this issue comprehensively, delving into the conception of creating a novel technique centered on deep state machines.

## Installation

1. Clone the repository to your local machine
2. Create a new conda environment using the `environment.yml`  file

    `conda env create -f environment.yml` 

## Data Generation

In order to procure the data required for all proposed methodologies, data must be generated through the utilization of the `Data Generation` notebook. Depending on the specific required experiment, the entire notebook can be executed, or selective execution of specific sections can be undertaken.
The notebook is comprehensively annotated, providing detailed explanations of each cell, along with appropriate guidelines on when to conclude the execution for each task.

## Proposed Methods
 
In this context, we present a brief description of the notebooks relative to the different experimental scenarios and the respective proposed methodologies.

- `Decision trees and Random Forests` : This notebook pretains to the experimentations done using decision trees and random forests as classifiers to detect the presence of SSH keys withing data segments.


- `SSH_extraction pipeline`: This notebook presents a novel approach that integrates classification and regression tasks for the purpose of SSH key extraction.

- `Deep State Machines`: In this Notebook we display a couple of experimentations of SSH key extraction using deep state machines.

- `Scripts`: This directory contains different scripts used for training, validation, testing and various other experimentations.



