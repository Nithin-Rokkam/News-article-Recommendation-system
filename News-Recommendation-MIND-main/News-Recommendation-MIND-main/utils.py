
"""

This module contains utility functions for the news recommendation system, including functions for evaluation, 
ranking and generating recommendations.

Functions:
    generate_recommendations(ind2item, ind2user, model, n): Generates top n recommendations for the first 10 users
    user_click(list_of_recommended_articles): Simulates a user's click on the recommended articles.
    ranking_func(list_of_recommended_articles, user_click): Generates a ranking list based on the user's click.
    mean_reciprocal_rank(rankings): Computes the Mean Reciprocal Rank (MRR) for the rankings.
    discounted_cumulative_gain(rankings): Computes the Discounted Cumulative Gain (DCG) for the rankings.
    Normalised_dcg(dcg_score, rankings): Computes the Normalized DCG (NDCG) for the rankings.
"""

import numpy as np

def generate_recommendations(ind2item, ind2user, model, news, n):
    """
    Generates news recommendations for a sample of users based on the trained model.

    Args:
        ind2item (dict): Mapping of numerical item indices to item IDs.
        ind2user (dict): Mapping of numerical user indices to user IDs.
        model (torch.nn.Module): Trained news recommendation model.

    Returns:
        tuple: Tuple containing lists of user IDs and their corresponding recommendations.
    """
    import torch
    import pandas as pd
    
    users, recommendations = [], []
    for user_id in range(1, 11, 1):
        item_id = list(ind2item.keys())
        userIdx = [user_id] * len(item_id)
        users.append(ind2user[user_id])
        
        predictions = model.forward(torch.IntTensor(userIdx), torch.IntTensor(item_id))
        
        # Select top n argmax
        top_index = torch.topk(predictions.flatten(), n).indices
        
        # Filter for top 10 suggested items
        filters = [ind2item[ix.item()] for ix in top_index]
        recommendations.append(news[news["itemId"].isin(filters)]['itemId'].values)
    
    return users, recommendations



def user_click(list_of_recommended_articles):
    """
    Simulates a user's click on the recommended articles. For simplicity, it always returns the second article.

    Args:
        list_of_recommended_articles (list): List of recommended articles.

    Returns:
        str: The article that the user clicked on.
    """
    return list_of_recommended_articles[1]

def ranking_func(list_of_recommended_articles, user_click):
    """
    Generates a ranking list based on the user's click. The list will have 1 at the index of the clicked article and 0 elsewhere.

    Args:
        list_of_recommended_articles (list): List of recommended articles.
        user_click (str): The article that the user clicked on.

    Returns:
        list: Ranking list with 1 at the index of the clicked article and 0 elsewhere.
    """
    ranking = [0] * len(list_of_recommended_articles)
    idx = list_of_recommended_articles.index(user_click)
    ranking[idx] = 1
    return ranking

def mean_reciprocal_rank(list_of_recommended_articles):
    """
    Computes the Mean Reciprocal Rank (MRR) based on the list of recommended articles and user's click.

    Args:
        list_of_recommended_articles (list): List of recommended articles.

    Returns:
        float: Mean Reciprocal Rank (MRR).
    """
    user_choice = user_click(list_of_recommended_articles)
    rankings = ranking_func(list_of_recommended_articles, user_choice)
    reciprocal_ranks = [1 / (i + 1) for i, rank in enumerate(rankings) if rank == 1]
    if reciprocal_ranks:
        mrr = np.mean(reciprocal_ranks)
        return mrr
    else:
        return 0.0

def discounted_cumulative_gain(list_of_recommended_articles):
    """
    Computes the Discounted Cumulative Gain (DCG).

    Args:
        list_of_recommended_articles (list): List of recommended articles.

    Returns:
        float: Discounted Cumulative Gain (DCG).
    """

    user_choice = user_click(list_of_recommended_articles)
    rankings = ranking_func(list_of_recommended_articles, user_choice)
    return sum((2**rank - 1) / np.log2(i + 2) for i, rank in enumerate(rankings))

def Normalised_dcg(list_of_recommended_articles):
    """
    Computes the Normalized Discounted Cumulative Gain (NDCG) for the rankings.

    Args:
        list_of_recommended_articles (list): List of recommended articles.

    Returns:
        float: Normalized Discounted Cumulative Gain (NDCG).
    """
    user_choice = user_click(list_of_recommended_articles)
    rankings = ranking_func(list_of_recommended_articles, user_choice)

    dcg_score = discounted_cumulative_gain(list_of_recommended_articles)
    ideal_dcg = sum((2**rank - 1) / np.log2(i + 2) for i, rank in enumerate(sorted(rankings, reverse=True)))
    if ideal_dcg == 0:
        return 0.0
    return dcg_score / ideal_dcg
