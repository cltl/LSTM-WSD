"""Compute sense embeddings

Assumption is that:
a) configs/main.json contains main experiments settings
b) configs/<exp>.json contains configuration for the experiment

Usage:
  compute_sense_embeddings.py --exp=<exp>

Example:
    python compute_sense_embeddings.py --exp='synset---se13---semcor'

Options:
  -h --help     Show this screen.
  --exp=<exp> the name of the experiment
"""
import json
import pandas
import os
from docopt import docopt
from collections import defaultdict
from wsd_class import WsdLstm

arguments = docopt(__doc__)

main_config = json.load(open('configs/main.json'))
path_exp_config = '%s/%s/settings.json' % (main_config['experiments_folder'],
                                           arguments['--exp'])
exp_config = json.load(open(path_exp_config))

meaning2context_embds = defaultdict(list)

wsd_lstm_obj = WsdLstm(model_path=main_config['model_path'],
                       vocab_path=main_config['vocab_path'])


num_target_embeddings = 0

for instance_id, \
    target_index, \
    annotation, \
    target_embedding in wsd_lstm_obj.apply_on_lstm_input_file(lstm_input_path=exp_config['lstm_input'],
                                                              batch_size=exp_config['batch_size']):

    meaning2context_embds[annotation].append((instance_id, target_index, target_embedding))

    num_target_embeddings += 1


meaning2avg_embedding = dict()
for meaning, embeddings in meaning2context_embds.items():
    total = sum([embedding[2]
                 for embedding in embeddings])
    average = total / len(embeddings)

    if len(embeddings) == 1:
        assert all(average == embeddings[0][2])
    meaning2avg_embedding[meaning] = average


pandas.to_pickle(meaning2avg_embedding, exp_config['meanings_path'])
pandas.to_pickle(meaning2context_embds, exp_config['meaning_instances_path'])


sense_embedding_stats_path = os.path.join(exp_config['exp_output_folder'],
                                          'meaning_embeddings_stats.txt')

with open(sense_embedding_stats_path, 'w') as outfile:
    outfile.write('number of target embeddings: %s\n' % num_target_embeddings)
    outfile.write('number of meanings: %s\n' % len(meaning2avg_embedding))

