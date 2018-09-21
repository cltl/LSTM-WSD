cd ..
rm -rf output
mkdir output
sbatch --time=04:00:00 --output=jobs/sensekey---se13---omsti one_experiment.sh sensekey---se13---omsti
sbatch --time=04:00:00 --output=jobs/sensekey---se13---semcor one_experiment.sh sensekey---se13---semcor
sbatch --time=04:00:00 --output=jobs/sensekey---se13---semcor_omsti one_experiment.sh sensekey---se13---semcor_omsti
sbatch --time=04:00:00 --output=jobs/sensekey---se2---omsti one_experiment.sh sensekey---se2---omsti
sbatch --time=04:00:00 --output=jobs/sensekey---se2---semcor---instance one_experiment.sh sensekey---se2---semcor---instance
sbatch --time=04:00:00 --output=jobs/sensekey---se2---semcor one_experiment.sh sensekey---se2---semcor
sbatch --time=04:00:00 --output=jobs/sensekey---se2---semcor_omsti one_experiment.sh sensekey---se2---semcor_omsti
sbatch --time=04:00:00 --output=jobs/synset---se13---omsti one_experiment.sh synset---se13---omsti
sbatch --time=04:00:00 --output=jobs/synset---se13---semcor one_experiment.sh synset---se13---semcor
sbatch --time=04:00:00 --output=jobs/synset---se13---semcor_omsti one_experiment.sh synset---se13---semcor_omsti
sbatch --time=04:00:00 --output=jobs/synset---se2---omsti one_experiment.sh synset---se2---omsti
sbatch --time=04:00:00 --output=jobs/synset---se2---semcor---instance one_experiment.sh synset---se2---semcor---instance
sbatch --time=04:00:00 --output=jobs/synset---se2---semcor one_experiment.sh synset---se2---semcor
sbatch --time=04:00:00 --output=jobs/synset---se2---semcor_omsti one_experiment.sh synset---se2---semcor_omsti

#sbatch --time=04:00:00 --output=jobs/synset---all---semcor one_experiment.sh synset---all---semcor
