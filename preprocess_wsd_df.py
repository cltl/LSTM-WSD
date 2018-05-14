"""Preprocess WSD dataframe

Assumption is that:
a) configs/main.json contains main experiments settings
b) configs/<exp>.json contains configuration for the experiment

Usage:
  preprocess_wsd_df.py --exp=<exp>

Example:
    python preprocess_wsd_df.py --exp='synset---se13---semcor'

Options:
  -h --help     Show this screen.
  --exp=<exp> the name of the experiment
"""
import json
import pandas
import os
from nltk.corpus import wordnet as wn
from docopt import docopt
import load_utils
import wn_utils


def update_wsd_df(wsd_df, wn_version, level):
    """
    extract relevant information for experiment:
    a) add column candidate_meanings for each row
    b) compute set of all polysemous candidate meanings

    :param pandas.core.frame.DataFrame wsd_df: a wsd competition dataframe
    :param str wn_version: supported: '30'
    :param str level: supported: 'synset | sensekey'

    :rtype: tuple
    :return: (wsd_df,
              all_polysemous_candidate_meanings,
    """
    columns_to_add = ['candidate_meanings', 'synset2sensekey']

    for key in columns_to_add:
        wsd_df[key] = [None for _ in range(len(wsd_df))]

    all_polysemous_synsets = set()  # synsets that are candidates of polysemous lemmas
    all_polysemous_sensekeys = set()  # sensekeys that are candidates of polysemous lemmas

    for row in wsd_df.itertuples():
        row_index = row.Index

        candidates, gold_inside = wn_utils.candidate_selection(wn,
                                                               token=row.token,
                                                               target_lemma=row.target_lemma,
                                                               pos=row.pos,
                                                               gold_lexkeys=row.lexkeys)

        if not gold_inside:
            print('gold synset candidate not available for: %s' % row.token_ids[0])

        synset_ids = [wn_utils.synset2identifier(candidate, wn_version)
                      for candidate in candidates]

        if len(synset_ids) >= 2:
            all_polysemous_synsets.update(synset_ids)

        sensekeys, synset2sensekeys = wn_utils.get_synset2sensekeys(wn,
                                                  candidates,
                                                  wn_version,
                                                  row.target_lemma,
                                                  row.pos,
                                                  debug=False)

        assert synset2sensekeys

        if len(sensekeys) >= 2:
            all_polysemous_sensekeys.update(sensekeys)

        if not any(lexkey in row.lexkeys
                   for lexkey in sensekeys):
            print()
            print('gold sensekey candidate not available for: %s' % row.token_ids[0])

        if level == 'synset':
            candidate_meanings = synset_ids
        elif level == 'sensekey':
            candidate_meanings = sensekeys

        wsd_df.set_value(row_index, col='candidate_meanings', value=candidate_meanings)
        wsd_df.set_value(row_index, col='synset2sensekey', value=synset2sensekeys)


    if level == 'synset':
        all_polysemous_candidate_meanings = all_polysemous_synsets
    elif level == 'sensekey':
        all_polysemous_candidate_meanings = all_polysemous_sensekeys

    return wsd_df, all_polysemous_candidate_meanings


arguments = docopt(__doc__)

main_config = json.load(open('configs/main.json'))

path_exp_config = 'configs/%s.json' % arguments['--exp']
exp_config = json.load(open(path_exp_config))

load_utils.update_settings_with_paths(main_config=main_config,
                                      exp_config=exp_config)
wsd_df = pandas.read_pickle(exp_config['wsd_df_path'])


wn_version = exp_config['wn_version']
level = exp_config['level']


wsd_df, all_polysemous_candidate_meanings = update_wsd_df(wsd_df=wsd_df,
                                                          wn_version=wn_version,
                                                          level=level)

pandas.to_pickle(wsd_df,
                 exp_config['output_wsd_df_path'])

pandas.to_pickle(all_polysemous_candidate_meanings,
                 exp_config['candidates_path'])


stats_path = os.path.join(exp_config['exp_output_folder'],
                          'preprocess_stats.txt')
with open(stats_path, 'w') as outfile:
    outfile.write('# rows wsd df: %s\n' % len(wsd_df))
    outfile.write('# polysemous candidate meanings: %s\n' % len(all_polysemous_candidate_meanings))


# asserts
for row in wsd_df.itertuples():
    assert row.candidate_meanings

# write updated experiment config to file
output_path_config = os.path.join(exp_config['exp_output_folder'],
                                  'settings.json')
with open(output_path_config, 'w') as outfile:
    json.dump(exp_config, outfile)

