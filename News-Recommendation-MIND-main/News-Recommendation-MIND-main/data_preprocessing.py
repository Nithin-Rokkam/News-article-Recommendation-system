
import pandas as pd
import numpy as np


def process_impression(s):
    """
    Processes the impression string to extract clicked and non-clicked items,
    and maps them to numerical indices using the provided item-to-index mapping.

    Args:
        s (str): Impression string containing item IDs and click indicators.

    Returns:
        tuple: Tuple containing lists of non-clicked items and the clicked item.
    """
    list_of_strings = s.split(" ")
    itemid_rel_tuple = [l.split("-") for l in list_of_strings]
    noclicks = []
    for entry in itemid_rel_tuple:
        if entry[1] == '0':
            noclicks.append(entry[0])
        if entry[1] == '1':
            click = entry[0]
    return noclicks, click


def process_click_history(s, item2ind):
    """
    Converts the click history string into a list of item indices.

    Args:
        s (str): Click history string containing space-separated item IDs.
        item2ind (dict): Mapping of item IDs to numerical indices.

    Returns:
        list: List of item indices corresponding to the click history.
    """
    list_of_strings = str(s).split(" ")
    return [item2ind.get(l, 0) for l in list_of_strings]


def load_and_preprocess_data(behaviour_filepath, news_filepath):
    """
    Loads the raw behaviour data from the specified file, removes null values,
    maps user IDs to numerical indices, and converts timestamps to epoch hours.

    Args:
        filepath (str): Path to the raw behaviour data file.

    Returns:
        pd.DataFrame: Preprocessed behaviour data.
        dict: Mapping of user IDs to numerical indices.
        dict: Mapping of news item IDs to numerical indices.
    """
    raw_behaviour = pd.read_csv(behaviour_filepath, sep="\t",
    names=["impressionId","userId","timestamp","click_history","impressions"])
    print("Data loaded")


    # Map user IDs to numerical indices
    unique_userIds = raw_behaviour['userId'].unique()
    ind2user = {idx + 1: itemid for idx, itemid in enumerate(unique_userIds)}
    user2ind = {itemid: idx + 1 for idx, itemid in enumerate(unique_userIds)}
    raw_behaviour['userIdx'] = raw_behaviour['userId'].map(lambda x: user2ind.get(x, 0))

    # Convert timestamps to epoch hours
    raw_behaviour['epochhrs'] = pd.to_datetime(raw_behaviour['timestamp']).values.astype(np.int64) / 1e6 / 1000 / 3600
    raw_behaviour['epochhrs'] = raw_behaviour['epochhrs'].round()

    # Load news data and build index of items
    news = pd.read_csv(news_filepath, sep="\t",
        names=["itemId","category","subcategory","title","abstract","url","title_entities","abstract_entities"]
    )
    ind2item = {idx + 1: itemid for idx, itemid in enumerate(news['itemId'].values)}
    item2ind = {itemid: idx for idx, itemid in ind2item.items()}

    # Converting the click history
    raw_behaviour['click_history_idx'] = raw_behaviour.click_history.map(lambda s:  process_click_history(s, item2ind))

    # Extract click and non click items
    raw_behaviour['noclicks'], raw_behaviour['click'] = zip(*raw_behaviour['impressions'].map(process_impression))
    raw_behaviour['noclicks'] = raw_behaviour['noclicks'].map(lambda list_of_strings: [item2ind.get(l, 0) for l in list_of_strings])
    raw_behaviour['click'] = raw_behaviour['click'].map(lambda x: item2ind.get(x,0))
    return raw_behaviour, news, ind2user, user2ind, ind2item, item2ind
