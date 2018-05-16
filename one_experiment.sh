#!/bin/bash
#BATCH --time=12:00:00
#SBATCH -C TitanX
#SBATCH --gres=gpu:1

module load python/3.5.2
module load cuda80/toolkit
module load cuda80/blas
module load cuda80
module load cuDNN/cuda80/6.0.21

#check if enough arguments are passed, else print usage information
if [ $# -eq 0 ];
then
    echo
    echo "Usage:                    : $0 experiment_name"
    echo "experiment_name           : name of experiment to run"
    exit -1;
fi

experiment=$1

#preprocess wsd df
python preprocess_wsd_df.py --exp=$experiment

ret=$?
if [ $ret -ne 0 ]; 
then
    exit 1
fi

# annotated data to input lstm
python lstm_input.py --exp=$experiment

ret=$?
if [ $ret -ne 0 ]; 
then
    exit 1
fi

#compute meaning embeddings
python compute_sense_embeddings.py --exp=$experiment


ret=$?
if [ $ret -ne 0 ]; 
then
    exit 1
fi

sleep 10 

#perform wsd
python perform_wsd.py --exp=$experiment
