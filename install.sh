


rm -rf resources

mkdir resources

cd resources
wget https://github.com/MartenPostma/WSD-Gold_standard-Analyst/raw/master/dataframes/se2-aw-framework.p
wget https://github.com/MartenPostma/WSD-Gold_standard-Analyst/raw/master/dataframes/se13-aw-framework.p

wget http://lcl.uniroma1.it/wsdeval/data/WSD_Unified_Evaluation_Datasets.zip
unzip WSD_Unified_Evaluation_Datasets.zip
cd WSD_Unified_Evaluation_Datasets
javac Scorer.java
cd ../..