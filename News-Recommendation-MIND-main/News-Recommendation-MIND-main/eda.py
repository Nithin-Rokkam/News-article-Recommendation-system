
"""

This module contains functions for performing exploratory data analysis (EDA)
on the news recommendation dataset.

Functions:
    plot_category_distribution(news): Plots the distribution of news categories.
    plot_user_login_histogram(raw_behaviour): Plots the histogram of user login frequencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def plot_category_distribution(news):
    """
    Plots the distribution of news categories in the dataset.

    Args:
        news (pd.DataFrame): DataFrame containing news articles with a 'category' column.

    Returns:
        None
    """
    count = news[['category','subcategory']].value_counts()
    index = []

    for i in count.index:
        index.append(np.array(i))

    index = np.array(index)
    df = pd.DataFrame(columns=['Category','Sub Category','Values'])
    df['Category'] = index[:,0]
    df['Sub Category'] = index[:,1]
    df['Values'] = count.values
    return px.bar(data_frame=df,x='Category',y='Values',color='Sub Category')

def plot_user_login_histogram(raw_behaviour):
    """
    Plots a histogram of the login frequencies for the top 15 users.

    Args:
        raw_behaviour (pd.DataFrame): DataFrame containing user behaviour data with a 'userId' column.

    Returns:
        None
    """
    user_login_counts = raw_behaviour['userId'].value_counts().head(15)
    plt.figure(figsize=(10, 6))
    user_login_counts.plot(kind='bar')
    plt.title('Top 15 Users by Login Frequency')
    plt.xlabel('User ID')
    plt.ylabel('Login Count')
    plt.show()
