# coding: utf-8
import itertools
import igraph
import copy
import math
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity
import utils
import pandas as pd
import time


def tw_idf(text, idf, id2word, avg_len, type="closeness", b=0.003, window=3):
    # if len(text) == 1:
    #     import pdb; pdb.set_trace()
    # build the graph
    graph = terms_to_graph(text, window)

    doc_len = len(text)

    if len(text) == 1:
        metrics = [(text[0], 0)]
    else:
        metrics = compute_node_centrality(graph, type=type)

    feature_row = np.zeros(len(idf))
    # for each term compute its tw
    indices = []
    values = []
    for term in set(text):
        if term not in id2word.token2id:
            continue
        index = id2word.token2id[term]
        idf_term = idf[term]
        denominator = (1 - b + (b * (float(doc_len) / avg_len)))
        metrics_term = [metric[1]
                        for metric in metrics if metric[0] == term][0]
        # store index and values in list
        indices.append(int(index))
        values.append(metrics_term)
    # build the sparse vector using the list of indices and values
    indices_j = np.array(indices)
    indices_i = np.array(len(indices) * [0])
    tw_idf = coo_matrix((values, (indices_i, indices_j)),
                               shape=(1, len(id2word)))
    tw_idf = csr_matrix(tw_idf)
    return tw_idf


def compute_idf(texts, id2word):
    """
    compute idf of all terms and put it in a dictionary
    """
    # # unique terms
    # all_unique_terms = id2word.keys()

    # store IDF values in dictionary
    n_doc = id2word.num_docs

    idf = copy.deepcopy(id2word.token2id)
    counter = 0

    for element in idf.keys():
        # number of documents in which each term appears
        df = sum([element in terms for terms in texts])
        # idf
        idf[element] = math.log10(float(n_doc + 1) / df)
        counter += 1
        if counter % 1e3 == 0:
            print counter, "terms have been processed"
    avg_len = sum(len(terms) for terms in texts) / len(texts)
    return idf, avg_len


def terms_to_graph(terms, window_size):
    """
    This function returns a directed, weighted igraph from a list of terms
    (the tokens from the pre-processed text) e.g., ['quick','brown','fox']
    Edges are weighted based on term co-occurence within a sliding window of
    fixed size 'w'
    """
    from_to = {}

    if len(terms) > window_size:
        # create initial complete graph (first w terms)
        terms_temp = terms[0:window_size]
        indexes = list(itertools.combinations(range(window_size), r=2))

        new_edges = []

        for my_tuple in indexes:
            new_edges.append(tuple([terms_temp[i] for i in my_tuple]))

        for new_edge in new_edges:
            if new_edge in from_to:
                from_to[new_edge] += 1
            else:
                from_to[new_edge] = 1

        # then iterate over the remaining terms
        for i in xrange(window_size, len(terms)):
            # term to consider
            considered_term = terms[i]
            # all terms within sliding window
            terms_temp = terms[(i-window_size+1):(i+1)]

            # edges to try
            candidate_edges = []
            for p in xrange(window_size-1):
                candidate_edges.append((terms_temp[p], considered_term))

            for try_edge in candidate_edges:

                # if not self-edge
                if try_edge[1] != try_edge[0]:

                    # if edge has already been seen, update its weight
                    if try_edge in from_to:
                        from_to[try_edge] += 1

                    # if edge has never been seen, create it and assign it a unit weight
                    else:
                        from_to[try_edge] = 1

    # create empty graph
    g = igraph.Graph(directed=True)

    # add vertices
    g.add_vertices(sorted(set(terms)))

    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())

    # set edge and vertice weights
    g.es['weight'] = from_to.values()  # based on co-occurence within sliding w
    g.vs['weight'] = g.strength(weights=from_to.values())  # weighted degree

    return g


def compute_node_centrality(graph, type="degree"):
    types = ["degree", "w_degree", "closeness", "w_closeness"]
    if type not in types:
        raise ValueError("Type {} is not implemented".format(
            type))
    if type == "degree":
        results = graph.degree()
        results = [round(float(result)/(len(graph.vs)-1), 5)
                   for result in results]

    if type == "w_degree":
        results = graph.strength(weights=graph.es["weight"])
        results = [round(float(result)/(len(graph.vs)-1), 5)
                   for result in results]

    if type == "closeness":
        results = graph.closeness(normalized=True)
        results = [round(value, 5) for value in results]

    if type == "w_closeness":
        results = graph.closeness(normalized=True,
                                  weights=graph.es["weight"])
        results = [round(value, 5) for value in results]

    return zip(graph.vs["name"], results)


def top_n_similarity(n, mid, df_user_messages, twidf_df):
    # Compute tw-idf for 'messages' and 'user_messages'
    #twidf_message = tw_idf(message, idf, id2word, avg_len)
    twidf_message = twidf_df[twidf_df['mid'] == mid]['twidf']
    df_user_messages['score'] = pd.Series(np.zeros(len(df_user_messages)))
    for ind, row in df_user_messages.iterrows():
        #print(row['tokens'])  # to debug
        #twidf_user_mess = tw_idf(row['tokens'], idf, id2word, avg_len)
        twidf_user_mess = twidf_df[twidf_df['mid'] == row['mid']]['twidf']
        df_user_messages.loc[ind, 'score'] = cosine_similarity(twidf_message.reshape((1, -1)),
                                                               twidf_user_mess.reshape((1, -1)))[0, 0]
    # TRAITER SCORES 0 /!\
    return df_user_messages.nlargest(n, 'score')


if __name__ == "__main__":

    dataset_path = "data/training_set.csv"
    mail_path = "data/training_info.csv"
    train_df = utils.load_dataset(dataset_path, mail_path, train=True)

    train_df = utils.preprocess_bodies(train_df)
    df_user_messages = train_df.head(100)
    texts = train_df["tokens"]
    # Message to compare
    message = texts[0]
    # Compute idf
    id2word = corpora.Dictionary(texts)
    id2word.filter_extremes(no_below=4, no_above=0.2, keep_n=100000)
    idf, avg_len = compute_idf(texts, id2word)
    # Computing similarity between message and df_user_messages
    n = 5
    twidf_df = pd.DataFrame(columns=['mid', 'twidf'])
    twidf_df['twidf'] = train_df['mid'].map(lambda x:
                                            tw_idf(train_df[train_df['mid'] == x]['tokens'].values[0],
                                                   idf, id2word, avg_len)
                                            )
    twidf_df['mid'] = train_df['mid']
    result = top_n_similarity(n, message, df_user_messages, twidf_df)
