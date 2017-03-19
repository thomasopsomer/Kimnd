# Kimnd

Code for the ALTEGRAD project on email prediction.

## File description

- gow.py: Module to build the graph of words
- greetings.py: Model to extract greetings features from message bodies
- utils.py: A set of helper functions
- spacy_utils.py: A set of helper function related to NLP and spacy
- textual_features.py: Module to extract features from message bodies
- topic_modeling.py: Module to extract topics from message bodies
- doc2vec.py: Module to learn document representation using gensim Doc2Vec
- flat_dataset.py: Module that gather helper function to flatten the dataset
- average_precision.py: Function to evaluate precesion @k
- temporal_features.py: Module to extract temporal features
- training_clf_emb.py: Module to replicate result with Doc2Vec embedding. Need all other features to be already extracted and picklized.
- enron_graph_corpus: Module with an object to represent the graph as well as the corpus and to build / load representation for words, messaages and peoples.

## Main

To replicate our submission result, one may execute the jupyter notebook 
`name_of_notebook`


## Dependencies

Code is tested with python 2.7 and use the following packages:

- pandas
- numpy
- scipy
- spaCy
- regex
- gensim
- sklearn
