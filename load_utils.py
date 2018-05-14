import shutil
import os

def update_settings_with_paths(main_config, exp_config):
    """
    update experiment json with path keys

    :param dict main_config: dictionary with information
    about main settings (so not experiment specific such as where the model is stored)
    :param dict exp_config: dictionary with information about
    settings for one specific experiments (see folder configs)
    """
    exp_output_folder = os.path.join(main_config['experiments_folder'],
                                     exp_config['label'])
    exp_config['exp_output_folder'] = exp_output_folder

    if os.path.exists(exp_config['exp_output_folder']):
        shutil.rmtree(exp_config['exp_output_folder'])
    os.makedirs(exp_config['exp_output_folder'], exist_ok=True)

    # input wsd df
    competition = exp_config['competition']
    assert competition in {'se2', 'se13'}, 'competition %s not supported'
    wsd_df_path = 'resources/{competition}-aw-framework.p'.format_map(locals())
    exp_config['wsd_df_path'] = wsd_df_path

    # output_wsd_df
    output_wsd_df_path = os.path.join(exp_config['exp_output_folder'],
                                      'wsd_output.p')
    exp_config['output_wsd_df_path'] = output_wsd_df_path

    # candidate meanings
    candidates_path = os.path.join(exp_config['exp_output_folder'],
                                  'candidate_meanings.p')
    exp_config['candidates_path'] = candidates_path


    # lstm input path
    exp_config['lstm_input'] = os.path.join(exp_config['exp_output_folder'],
                                            'lstm_input.txt')

    # embedding instances output path
    exp_config['meaning_instances_path'] = os.path.join(exp_config['exp_output_folder'],
                                                         'meaning_instances.p')

    # embeddings path
    exp_config['meanings_path'] = os.path.join(exp_config['exp_output_folder'],
                                               'meanings.p')

