# coding: utf-8
import itertools
import heapq
import operator
import igraph
import copy
import time
import numpy as np


def tw_idf(text, idf, id2word, type="closeness", b=0.003):

    # build the graph
    graph = terms_to_graph(text, window)

    doc_len = len(text)

    metrics = compute_node_centrality(graph, type=type)

    feature_row = np.zeros(len(idf))
    # for each term compute its tw
    for term in set(text):
        index = id2word.token2id(term)
        idf_term = idf[term]
        denominator = (1 - b + (b * (float(doc_len) / avg_len)))
        metrics_term = [metric[1]
                        for metric in metrics if metric[0] == term][0]

        # store TW-IDF values
        feature_row[index] = (
            float(metrics_term[0]) / denominator) * idf_term
    return feature_row


def compute_idf(texts, id2word):
    """
    compute idf of all terms and put it in a dictionary
    """
    # unique terms
    all_unique_terms = id2word.keys()

    # store IDF values in dictionary
    n_doc = id2word.num_doc

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
    return idf


def terms_to_graph(terms, window_size):
    """
    This function returns a directed, weighted igraph from a list of terms
    (the tokens from the pre-processed text) e.g., ['quick','brown','fox']
    Edges are weighted based on term co-occurence within a sliding window of
    fixed size 'w'
    """

    from_to = {}

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

    return(g)


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
        results = [round(value, 5) for value in resultss]

    return(zip(graph.vs["name"], results))


def print_top10(feature_names, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    # coef stores the weights of each feature (in unique term), for each class
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
                          " ".join(feature_names[j] for j in top10)))


def print_bot10(feature_names, clf, class_labels):
    """Prints features with the lowest coefficient values, per class"""
    for i, class_label in enumerate(class_labels):
        bot10 = np.argsort(clf.coef_[i])[0:9]
        print("%s: %s" % (class_label,
                          " ".join(feature_names[j] for j in bot10)))
