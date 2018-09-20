
# coding: utf-8

# In[43]:


import spacy
nlp = spacy.load('en_default')
from my_classes import Token
import pandas
import tensorflow as tf
from wsd_class import WsdLstm
import wn_utils
from nltk.corpus import wordnet as wn


sentence = 'the man walks to the fence'



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

sentence_tokens = sent2token_objs('the man walks to the garden', verbose=1)




meanings = pandas.read_pickle('demo_resources/meanings.p')




vocab_path = 'demo_resources/model-h2048p512/gigaword-lstm-wsd.index.pkl'
model_path = 'demo_resources/model-h2048p512/lstm-wsd-gigaword-google'

with tf.Session() as sess:  # your session object:
    wsd_lstm_obj = WsdLstm(model_path=model_path,
                           vocab_path=vocab_path,
                           sess=sess)
    target_index = 1
    target_token_obj = sentence_tokens[target_index]

    candidates, gold_inside = wn_utils.candidate_selection(wn,
                                                           token=target_token_obj.text,
                                                           target_lemma=target_token_obj.lemma,
                                                           pos=target_token_obj.pos)

    candidates = [wn_utils.synset2identifier(synset, '30')
                  for synset in candidates]

    wsd_strategy, \
    highest_meaning, \
    meaning2confidence, \
    target_embedding = wsd_lstm_obj.wsd_on_test_instance(sess,
                                                         sentence_tokens,
                                                         target_index=1,
                                                         candidate_meanings=candidates,
                                                         meaning_embeddings=meanings,
                                                         meaning_instances=None,
                                                         method='averaging',
                                                         debug=1)
    print(wsd_strategy, highest_meaning)
    print(meaning2confidence)



