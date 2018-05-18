from glob import glob
import json

with open('experiments.sh', 'w') as outfile:

    outfile.write('cd ..\n')
    outfile.write('rm -rf output\n')
    outfile.write('mkdir output\n')
    for json_path in sorted(glob('*json')):

        if json_path == 'main.json':
            continue

        exp_config = json.load(open(json_path))

        label = json_path[:-5]
        level, competition, corpora = label.split('---')
        corpora = corpora.split('_')

        assert level == exp_config['level']
        assert competition == exp_config['competition']
        assert corpora == exp_config['corpora']
        assert label == exp_config['label']

        command = 'sbatch --time=04:00:00 --output=jobs/%s one_experiment.sh %s' % (label, label)

        outfile.write(command + '\n')
