# News-Recommendation-MIND

This repository contains the implementation of a personalized news recommendation system using the MIND (Microsoft News Dataset). The system leverages matrix factorization techniques to provide users with relevant news articles based on their browsing history and interactions. The project includes data preprocessing, exploratory data analysis (EDA), model development, and evaluation metrics to measure the effectiveness of the recommendations.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset) 
- [Usage](#usage)

## Introduction

The objective of this project is to develop a personalized news recommendation system to enhance user engagement by suggesting relevant news articles. The system is built using matrix factorization techniques and evaluated using metrics such as Mean Reciprocal Rank (MRR), Discounted Cumulative Gain (DCG), and Normalized DCG (nDCG).

## Dataset

The MIND dataset was collected from anonymized behavior logs of the Microsoft News website. It includes:
- 1 million users with at least 5 news clicks during a 6-week period.
- Impression logs used for training, validation, and testing.
- A smaller version (MIND-small) with 50,000 users.

For more details on the dataset, please refer to the [official MIND dataset page](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md).

## Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/News-Recommendation-MIND.git
    cd News-Recommendation-MIND
    ```

2. **Install the necessary dependencies:**
    - If you are using Google Colab, you need to install PyTorch Lightning and download the dataset from drive. Add the following code block at the beginning of your notebook:
    ```python
    !pip install pytorch-lightning
    !python download_data.py
    ```


