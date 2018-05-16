import os
import json
import subprocess


def create_key_file(wsd_df, key_path, level):
    """
    create key file based on information in
    wsd_output.p in experiment folder

    :param pandas.core.frame.DataFrame wsd_df: wsd dataframe
    :param str key_path: path where keyfile will be stored
    :param str level: supported sensekey | synset
    """
    with open(key_path, 'w') as outfile:

        for row in wsd_df.itertuples():

            lstm_output = row.lstm_output

            if level == 'synset':
                target_sensekey = row.synset2sensekey[lstm_output]
            elif level == 'sensekey':
                target_sensekey = lstm_output

            token_id = row.token_ids[0]

            export = [token_id, target_sensekey]
            outfile.write(' '.join(export) + '\n')



def score_using_official_scorer(scorer_folder,
                                system,
                                key,
                                json_output_path):
    """
    score using official scorer

    :param str scorer_folder: scorer folder
    :param str system: path to system output
    :param str key: path to gold file (with answers from manual annotation)
    :param str json_output_path: path to where json with results are stored
    """
    command = 'cd {scorer_folder} && java Scorer "{key}" "../../{system}"'.format_map(locals())

    output = subprocess.check_output(command, shell=True)
    output = output.decode("utf-8")
    results = dict()
    for metric_output in output.split('\n')[:3]:

        metric, value = metric_output[:-1].split('=\t')

        value = value.replace(',', '.')

        results[metric] = float(value) / 100

    assert set(results) == {'P', 'R', 'F1'}

    with open(json_output_path, 'w') as outfile:
        json.dump(results, outfile)


