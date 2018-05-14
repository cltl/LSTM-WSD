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

# annotated data to input lstm
python lstm_input.py --exp=$experiment

#compute meaning embeddings
python compute_sense_embeddings.py --exp=$experiment