
rm -rf demo_resources

mkdir demo_resources

cd demo_resources

# download synset embeddings using semcor as annotated data
wget http://kyoto.let.vu.nl/~postma/LSTM-WSD/meanings.p

# download model
wget http://kyoto.let.vu.nl/~minh/wsd/model-h2048p512.zip
unzip model-h2048p512.zip

# install modules
pip3 install --user numpy
pip3 install --user scipy
pip3 install --user nltk
pip3 install --user pandas
printf 'import nltk; nltk.download("wordnet")' | python3
pip3 install --user tensorflow
pip3 install --user spacy
pip3 install --user https://github.com/explosion/spacy-models/releases/download/en_core_web_md-1.2.1/en_core_web_md-1.2.1.tar.gz
python3 -m spacy link en_core_web_md en_default
