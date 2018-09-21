"""Run LSTM on one sentence
Usage:
  demo.py  --sentence=<sentence>

Example:
    python demo.py --sentence='the man walks to the garden.'

Options:
    -h --help     Show this screen.
    --sentence=<sentence> a sentence
"""
import operator
import spacy
nlp = spacy.load('en_default')
from my_classes import Token
import pandas
import tensorflow as tf
from wsd_class import WsdLstm
import wn_utils
from nltk.corpus import wordnet as wn
from docopt import docopt


def sent2token_objs(sentence, verbose=0):
    """
    given a sentence, process them with spacy
    and represent them using my_classes.Token instances
    
    :param str sentence: a sentence
    
    :rtype: list
    :return: list of my_classes.Token instances
    """
    token_objs = []

    for token_id, spacy_token_obj in enumerate(nlp(sentence)):

        token_obj = Token(
                    token_id,
                    spacy_token_obj.text,
                    spacy_token_obj.lemma_,
                    universal_pos=spacy_token_obj.pos_)


        if verbose >= 1:
            print(token_obj.token_id, 
                  token_obj.text, 
                  token_obj.lemma, 
                  token_obj.pos)

        token_objs.append(token_obj)
    return token_objs


arguments = docopt(__doc__)
sentence_tokens = sent2token_objs(arguments['--sentence'], verbose=1)
meanings = pandas.read_pickle('demo_resources/meanings.p')
vocab_path = 'demo_resources/model-h2048p512/gigaword-lstm-wsd.index.pkl'
model_path = 'demo_resources/model-h2048p512/lstm-wsd-gigaword-google'

with tf.Session() as sess:  # your session object:
    wsd_lstm_obj = WsdLstm(model_path=model_path,
                           vocab_path=vocab_path,
                           sess=sess)
    
    for target_index, target_token_obj in enumerate(sentence_tokens):


        candidates, gold_inside = wn_utils.candidate_selection(wn,
                                                               token=target_token_obj.text,
                                                               target_lemma=target_token_obj.lemma,
                                                               pos=target_token_obj.pos)

        if not candidates:
            print('no candidates found for: %s %s' % (target_token_obj.text, target_token_obj.lemma))
            continue 

        candidate_identifiers = []
        identifier2synset = dict()
        for synset in candidates:
            identifier = wn_utils.synset2identifier(synset, '30')
            identifier2synset[identifier] = synset
            candidate_identifiers.append(identifier)

        wsd_strategy, \
        highest_meaning, \
        meaning2confidence, \
        target_embedding = wsd_lstm_obj.wsd_on_test_instance(sess,
                                                             sentence_tokens,
                                                             target_index=1,
                                                             candidate_meanings=candidate_identifiers,
                                                             meaning_embeddings=meanings,
                                                             meaning_instances=None,
                                                             method='averaging',
                                                             debug=1)
        
        
        print()
        print(target_token_obj.text, target_token_obj.lemma)
        print('wsd strategy', wsd_strategy)
        print('chosen synset', highest_meaning)

        for meaning, confidence in sorted(meaning2confidence.items(),
                                          key=operator.itemgetter(1),
                                          reverse=True):
            synset = identifier2synset[meaning]
            print(meaning, confidence, synset.definition())



