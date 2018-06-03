"""Perform WSD

Assumption is that:
a) configs/main.json contains main experiments settings
b) configs/<exp>.json contains configuration for the experiment

Usage:
  perform_wsd.py --exp=<exp>

Example:
    python perform_wsd.py --exp='synset---se13---semcor'

Options:
  -h --help     Show this screen.
  --exp=<exp> the name of the experiment
"""
import json
import pandas
from docopt import docopt
import tensorflow as tf
import official_scorer
from wsd_class import WsdLstm

arguments = docopt(__doc__)

main_config = json.load(open('configs/main.json'))
path_exp_config = '%s/%s/settings.json' % (main_config['experiments_folder'],
                                           arguments['--exp'])
exp_config = json.load(open(path_exp_config))

from nltk.corpus import wordnet as wn
import pickle
word_vocab_path = 'output/vocab.2018-05-10-7d764e7.pkl'
word2id = pickle.load(open(word_vocab_path, 'rb'))
id2word = {i: w for w, i in word2id.items()}
hdn_vocab_path = 'output/hdn-vocab.2018-05-18-f48a06c.pkl'
hdn2id = pickle.load(open(hdn_vocab_path, 'rb'))
hdn_list_vocab_path = 'output/hdn-list-vocab.2018-05-18-f48a06c.pkl'
hdn_list2id = pickle.load(open(hdn_list_vocab_path, 'rb'))

def synset2identifier(synset, wn_version):
    """
    return synset identifier of
    nltk.corpus.reader.wordnet.Synset instance

    :param nltk.corpus.reader.wordnet.Synset synset: a wordnet synset
    :param str wn_version: supported: '171 | 21 | 30'

    :rtype: str
    :return: eng-VERSION-OFFSET-POS (n | v | r | a)
    e.g.
    """
    offset = str(synset.offset())
    offset_8_char = offset.zfill(8)

    pos = synset.pos()
    if pos == 'j':
        pos = 'a'

    identifier = 'eng-{wn_version}-{offset_8_char}-{pos}'.format_map(locals())

    return identifier


def synsets_graph_info(wn_instance, wn_version, lemma, pos):
    """
    extract:
    1. hyponym under lowest least common subsumer

    :param nltk.corpus.reader.wordnet.WordNetCorpusReader wn_instance: instance
    of nltk.corpus.reader.wordnet.WordNetCorpusReader
    :param str wn_version: supported: '171' | '21' | '30'
    :param str lemma: a lemma
    :param str pos: a pos

    :rtype: dict
    :return: mapping synset_id
        -> 'under_lcs' -> under_lcs identifier
        -> 'path_to_under_lcs' -> [sy1_iden, sy2_iden, sy3_iden, ...]
    """
    sy_id2under_lcs_info = dict()

    synsets = wn_instance.synsets(lemma, pos=pos)

    synsets = set(synsets)

    if len(synsets) == 1:
        sy_obj = synsets.pop()
        target_sy_iden = synset2identifier(sy_obj, wn_version)
        sy_id2under_lcs_info[target_sy_iden] = {'under_lcs': None,
                                                'under_lcs_obj': None,
                                                'sy_obj' : sy_obj,
                                                'path_to_under_lcs': []}
        return sy_id2under_lcs_info


    for sy1 in synsets:

        target_sy_iden = synset2identifier(sy1, wn_version)

        min_path_distance = 100
        closest_lcs = None

        for sy2 in synsets:
            if sy1 != sy2:
                try:
                    lcs_s = sy1.lowest_common_hypernyms(sy2, simulate_root=True)
                    lcs = lcs_s[0]
                except:
                    lcs = None
                    print('wordnet error', sy1, sy2)

                path_distance = sy1.shortest_path_distance(lcs, simulate_root=True)

                if path_distance < min_path_distance:
                    closest_lcs = lcs
                    min_path_distance = path_distance

        under_lcs = None
        for hypernym_path in sy1.hypernym_paths():
            for first, second in  zip(hypernym_path, hypernym_path[1:]):
                if first == closest_lcs:
                    under_lcs = second

                    index_under_lcs = hypernym_path.index(under_lcs)
                    path_to_under_lcs = hypernym_path[index_under_lcs + 1:-1]

                    under_lcs_iden = synset2identifier(under_lcs, wn_version)
                    path_to_under_lcs_idens = [synset2identifier(synset, wn_version)
                                               for synset in path_to_under_lcs]

                    sy_id2under_lcs_info[target_sy_iden] = {'under_lcs': under_lcs_iden,
                                                            'under_lcs_obj': under_lcs,
                                                            'sy_obj' : sy1,
                                                            'path_to_under_lcs': path_to_under_lcs_idens}

    return sy_id2under_lcs_info


from nltk.corpus import wordnet as wn

id2synset = {synset2identifier(s, '30'):s for s in wn.all_synsets('n')}
def get_hdns(lemma):
    graph_info = synsets_graph_info(wn_instance=wn,
                                wn_version='30',
                                lemma=lemma,
                                pos='n')
    return {info['under_lcs']: synset
            for synset, info in graph_info.items() 
            if info['under_lcs']}
    
from sklearn.metrics.pairwise import cosine_similarity

mono_path = 'output/monosemous-context-embeddings.2018-05-27-5cd9bb6.npz'
import numpy as np
monos = np.load(mono_path)
mono_words, mono_embs, mono_hdn_lists = monos['mono_words'], monos['mono_embs'], monos['mono_hdn_lists']

from collections import defaultdict

'''
def disambiguate(word, embs):
    hdn2synset = get_hdns(word)
    hdn_list = tuple(sorted(hdn2synset))
    if hdn_list not in hdn_list2id:
        return {}
    cases_of_same_hdn_list = (mono_hdn_lists == hdn_list2id[hdn_list])
    if not np.any(cases_of_same_hdn_list):
        return {}
    relevant_words = [id2word[i] for i in mono_words[cases_of_same_hdn_list]]
    relevant_hdns = []
    for w in relevant_words:
        hypernyms = [synset2identifier(s, '30') for s in wn.synsets(w, 'n')[0].hypernym_paths()[0]]
        rel_hdn, = [h for h in hypernyms if h in hdn2synset]
        relevant_hdns.append(rel_hdn)
    sims = cosine_similarity([embs], mono_embs[cases_of_same_hdn_list])[0]
    hdn2score = defaultdict(float)
    for hdn, sim in zip(relevant_hdns, sims):
        if sim > 0:
            hdn2score[hdn] += sim
    synset2score = {hdn2synset[hdn]: score for hdn, score in hdn2score.items()}
    return synset2score
'''

def disambiguate(word, embs):
    hdn2synset = get_hdns(word)
    hdn_list = tuple(sorted(hdn2synset))
    if hdn_list not in hdn_list2id:
        return {}
    cases_of_same_hdn_list = (mono_hdn_lists == hdn_list2id[hdn_list])
    if not np.any(cases_of_same_hdn_list):
        return {}
    relevant_words = [id2word[i] for i in mono_words[cases_of_same_hdn_list]]
    relevant_mono_synsets = []
    for w in relevant_words:
        s, = wn.synsets(w, 'n')
        relevant_mono_synsets.append(s)

    relevant_hdns, relevant_synsets = [], []
    for w in relevant_words:
        hypernyms = [synset2identifier(s, '30') 
                     for s in wn.synsets(w, 'n')[0].hypernym_paths()[0]]
        rel_hdn, = [h for h in hypernyms if h in hdn2synset]
        relevant_hdns.append(rel_hdn)
        relevant_synsets.append(hdn2synset[rel_hdn])
    relevant_embs = mono_embs[cases_of_same_hdn_list]
    
    synset2embs = defaultdict(list)
    for s, e in zip(relevant_synsets, relevant_embs):
        synset2embs[s].append(e)
    synset2score = {s: sum(embs_list)/len(embs_list)
                    for s, embs_list in synset2embs.items()}
    return synset2score

#     synset2synset_sims = [s1.wup_similarity(id2synset[s2]) 
#                           for s1, s2 in zip(relevant_mono_synsets, relevant_synsets)]
#     embeddings_sims = cosine_similarity([embs], mono_embs[cases_of_same_hdn_list])[0]
#     hdn2score = defaultdict(list)
#     for hdn, sim1, sim2 in zip(relevant_hdns, embeddings_sims, synset2synset_sims):
#         if sim1 > 0:
#             hdn2score[hdn].append([sim1, sim2])
#     for hdn in hdn2score:
#         sims1, sims2 = zip(*hdn2score[hdn])
#         hdn2score[hdn] = np.average(sims1, weights=sims2) 
#     synset2score = {hdn2synset[hdn]: score for hdn, score in hdn2score.items()}
#     return synset2score

with tf.Session() as sess:  # your session object:

    # path = 'output/hdn-large.2018-05-21-b1d1867-best-model'
    # path = 'output/hdn-large.2018-05-25-e069882-best-model'
    # saver = tf.train.import_meta_graph(path + '.meta', clear_devices=True)
    # saver.restore(sess, path)
    # x, logits, lens, candidates = load_tensors(sess)

    wsd_lstm_obj = WsdLstm(model_path=main_config['model_path'],
                           vocab_path=main_config['vocab_path'],
                           sess=sess)


    wsd_df = pandas.read_pickle(exp_config['output_wsd_df_path'])

#     meanings = pandas.read_pickle(exp_config['meaning_instances_path'])
    meanings = pandas.read_pickle(exp_config['meanings_path'])
    meaning_freqs = pandas.read_pickle(exp_config['annotated_data_stats'])

    colums_to_add = ['lstm_acc', 'emb_freq', 'wsd_strategy', 'lstm_output']
    for colum_to_add in colums_to_add:
        wsd_df[colum_to_add] = [None for _ in range(len(wsd_df))]


    correct = 0

    for row in wsd_df.itertuples():

        df_index = row.Index

        # target index of token
        target_index = None
        target_token_id = row.token_ids[0]
        for index, token_obj in enumerate(row.sentence_tokens):
            if token_obj.token_id == target_token_id:
                target_index = index
        assert type(target_index) == int, 'no target token index for %s' % df_index

        # emb freq
        emb_freq = {meaning_id : meaning_freqs.get(meaning_id, 0)
                    for meaning_id in row.candidate_meanings}

        # apply wsd
        if len(row.candidate_meanings) > 1:
            wsd_strategy = 'hdn'
            sentence_as_ids = [word2id.get(token_obj.text) or word2id['<unkn>'] for w in row.sentence_tokens]
            sentence_as_ids[target_index] = word2id['<target>']
            if '<eos>' in word2id:
                sentence_as_ids.append(word2id['<eos>'])
            
            embs = wsd_lstm_obj.apply_model(sess, [sentence_as_ids], [len(sentence_as_ids)])[0]
            word = row.sentence_tokens[target_index].text
#             meaning2confidence1 = disambiguate(word, embs)
#             print(meaning2confidence1)
#             meaning2confidence = meaning2confidence1
            wsd_strategy, \
            highest_meaning, \
            meaning2confidence2 = wsd_lstm_obj.wsd_on_test_instance(sess=sess,
                                                                   sentence_tokens=row.sentence_tokens,
                                                                   target_index=target_index,
                                                                   candidate_meanings=row.candidate_meanings,
                                                                   meaning_embeddings=meanings,
                                                                   debug=2)
            print(meaning2confidence2)
            meaning2confidence = meaning2confidence2
#             trust_factor = 1
#             meaning2confidence = {id_: (val or trust_factor*meaning2confidence1.get(id_, 0.0))
#                                   for id_, val in meaning2confidence2.items()}
            
            if meaning2confidence:
                highest_meaning = max(meaning2confidence, key=lambda m: meaning2confidence[m])
                if highest_meaning in row.synset2sensekey:
                    print(row.target_lemma, row.candidate_meanings, highest_meaning)
                else:
                    highest_meaning = row.candidate_meanings[0]
                    print('Ignored: ', row.target_lemma, row.candidate_meanings)
            else:
                highest_meaning = row.candidate_meanings[0]
                print('Ignored: ', row.target_lemma, row.candidate_meanings)
        else:
            highest_meaning = row.candidate_meanings[0]

        # score it
        if exp_config['level'] == 'synset':
            lstm_acc = highest_meaning in row.source_wn_engs
        elif exp_config['level'] == 'sensekey':
            lstm_acc = highest_meaning in row.lexkeys

        wsd_df.set_value(df_index, col='lstm_output', value=highest_meaning)
        wsd_df.set_value(df_index, col='lstm_acc', value=lstm_acc)
        wsd_df.set_value(df_index, col='wsd_strategy', value=wsd_strategy)
        wsd_df.set_value(df_index, col='emb_freq', value=emb_freq)

        if lstm_acc:
            correct += 1



pandas.to_pickle(wsd_df, exp_config['output_wsd_df_path'])


official_scorer.create_key_file(wsd_df, exp_config['system_path'], exp_config['level'])

official_scorer.score_using_official_scorer(scorer_folder=main_config['scorer_folder'],
                                            system=exp_config['system_path'],
                                            key=exp_config['key_path'],
                                            json_output_path=exp_config['json_results_path'])


results = json.load(open(exp_config['json_results_path']))
print(results)








