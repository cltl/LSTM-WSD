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
import official_scorer
from wsd_class import WsdLstm

arguments = docopt(__doc__)

main_config = json.load(open('configs/main.json'))
path_exp_config = '%s/%s/settings.json' % (main_config['experiments_folder'],
                                           arguments['--exp'])
exp_config = json.load(open(path_exp_config))

wsd_lstm_obj = WsdLstm(model_path=main_config['model_path'],
                       vocab_path=main_config['vocab_path'])


wsd_df = pandas.read_pickle(exp_config['output_wsd_df_path'])

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
    wsd_strategy, \
    highest_meaning, \
    meaning2confidence = wsd_lstm_obj.wsd_on_test_instance(sentence_tokens=row.sentence_tokens,
                                                           target_index=target_index,
                                                           candidate_meanings=row.candidate_meanings,
                                                           meaning_embeddings=meanings,
                                                           debug=2)

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








