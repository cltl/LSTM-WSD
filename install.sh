rm -rf resources

mkdir resources

cd resources

# download wsd dataframes
wget https://github.com/MartenPostma/WSD-Gold_standard-Analyst/raw/master/dataframes/se2-aw-framework.p
wget https://github.com/MartenPostma/WSD-Gold_standard-Analyst/raw/master/dataframes/se13-aw-framework.p

# download evaluation datasets
wget http://lcl.uniroma1.it/wsdeval/data/WSD_Unified_Evaluation_Datasets.zip
unzip WSD_Unified_Evaluation_Datasets.zip
cd WSD_Unified_Evaluation_Datasets
javac Scorer.java

cd ..

# download model
#wget http://kyoto.let.vu.nl/~minh/wsd/model-h2048p512.zip
#unzip model-h2048p512.zip

# install modules
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user nltk
printf 'import nltk; nltk.download("wordnet")' | python3
pip3 install --user tensorflow

# download annotated data
git clone https://github.com/cltl/semcor_omsti
cd semcor_omsti
bash install.sh
python3 convert_to_ulm.py -i SemCor -o semcor
# uncomment the following line to also preload omsti (which takes a long time)
#python convert_to_ulm.py -i OMSTI -o omsti

cd ../../..
