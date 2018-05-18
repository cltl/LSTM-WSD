"""Convert annotated data into input format LSTM

Assumption is that:
a) configs/main.json contains main experiments settings
b) configs/<exp>.json contains configuration for the experiment

Usage:
  lstm_input.py --exp=<exp>

Example:
    python preprocess_wsd_df.py --exp='synset---se13---semcor'

Options:
  -h --help     Show this screen.
  --exp=<exp> the name of the experiment
"""

import json
import pickle
import os
import pandas
from collections import defaultdict
from docopt import docopt

arguments = docopt(__doc__)

main_config = json.load(open('configs/main.json'))
path_exp_config = '%s/%s/settings.json' % (main_config['experiments_folder'],
                                           arguments['--exp'])
exp_config = json.load(open(path_exp_config))

all_meanings = pandas.read_pickle(exp_config['candidates_path'])
polysemous_meanings = pandas.read_pickle(exp_config['polysemous_candidates_path'])

stats = dict()
for relevant_meaning in all_meanings:
    if relevant_meaning not in stats:
        stats[relevant_meaning] = defaultdict(int)

with open(exp_config['lstm_input'], 'w') as outfile:
    for corpus in exp_config['corpora']:

        instances_path = os.path.join(main_config[corpus], 'instances.bin')
        instances = pickle.load(open(instances_path, 'rb'))

        meaning2ids_path = os.path.join(main_config[corpus],
                                        '%s_index.bin' % exp_config['level'])
        meaning2ids = pickle.load(open(meaning2ids_path, 'rb'))

        # relevant ids
        relevent_ids = set()
        for a_meaning, ids in meaning2ids.items():
            if a_meaning in all_meanings:
                relevent_ids.update(ids)

            for instance_id, instance_obj in instances.items():
                if instance_id in relevent_ids:
                    for annotation, training_example in instance_obj.sent_in_lstm_format(level=exp_config['level'],
                                                                                         only_keep=all_meanings):
                        stats[annotation][corpus] += 1
                        stats[annotation]['total'] += 1

                        if annotation in polysemous_meanings:
                            outfile.write(instance_id + '\t' + training_example + '\n')

pandas.to_pickle(stats, exp_config['annotated_data_stats'])

annotated_data_stats_path = os.path.join(exp_config['exp_output_folder'],
                                         'annotated_data_stats.tsv')

headers = ['meaning', 'polysemy', 'total'] + exp_config['corpora']
with open(annotated_data_stats_path, 'w') as outfile:
    outfile.write('\t'.join(headers) + '\n')

    for relevant_meaning, freqs in stats.items():

        cat = 'monosemous'
        if relevant_meaning in polysemous_meanings:
            cat = 'polysemous'

        one_row = [relevant_meaning, cat]
        for corpus in ['total'] + exp_config['corpora']:
            one_row.append(str(freqs[corpus]))

        outfile.write('\t'.join(one_row) + '\n')











