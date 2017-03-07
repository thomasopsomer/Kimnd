import numpy as np
import pandas as pd
from utils import flatmap
from gow import top30_similarity


def incoming_text_similarity(dataset, m, user):
    # Dataset containing all previous emails sent to person 'user'
    dataset_to_rec = dataset[dataset.recipient == user]
    # Measure similarity between m and all the messages received
    ## METTRE TEXTS
    dataset_similar = top30_similarity(m, dataset_to_rec, texts)
    df_incoming = pd.DataFrame(columns=['user', 'sender', 'recipient', 'incoming text'])
    for c in dataset_similar['sender']: ## TO CHANGE NAME, SEE WITH PIERRE
        df_incoming = df_incoming.append(pd.DataFrame(
            [user, c, user, 1], columns=df_incoming.columns)
        )
    return df_incoming


def outgoing_text_similarity(dataset, m, user):
    # Dataset containing all previous emails sent by person 'user'
    dataset_from_rec = dataset[dataset.recipient == user]
    # Measure similarity between m and all the messages received
    ## METTRE TEXTS
    dataset_similar = top30_similarity(m, dataset_from_rec, texts)
    df_outgoing = pd.DataFrame(columns=['user', 'sender', 'recipient', 'outgoing text'])
    for c in dataset_similar['recipient']: ## TO CHANGE NAME, SEE WITH PIERRE
        df_outgoing = df_outgoing.append(pd.DataFrame(
            [user, user, c, 1], columns=df_outgoing.columns)
        )
    return df_outgoing

# TO DO: CHECK WITH CHAIMAA DATASET OF FEATURES ; CHECK WITH PIERRE DATASET OF SIMILAR MESSAGES