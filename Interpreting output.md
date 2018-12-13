# Interpreting output

All the information concerning one specific experiment is stored in a folder.
This folder contains a file called **wsd_output.p**, which is a [pandas dataframe](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html).

Each row in the dataframe represent an instance of a semantic evaluation, e.g., [senseval-2 or SemEval-2013 task 12](https://web.archive.org/web/20170127124151/http://lcl.uniroma1.it:80/wsdeval/evaluation-data).

Each row contains the following information:
* **competition**: the competition to which the instance belongs: *se2-aw-framework* (senseval-2) or *se13-framework* (SemEval-2013 task 12). The suffix *-framework* indicates that the evaluation data comes from the [Unified Evaluation Framework](http://lcl.uniroma1.it/wsdeval/)
* **target_lemma**: the target lemma, e.g., *art*
* **pos**: the part of speech: n (noun), v (verb), a (adjective), r (adverb)
* **candidate_meanings**: the candidate synsets of the *lemma* and *pos* combination, ordered by their sense rank. The first synset in the list has sense rank 1 (Most Frequent Sense), the second sense rank 2, etc.
* **lexkeys**: the *gold sensekeys*, e.g., the ones that were annotated by the human annotation. (see the [WordNet glossary](https://wordnet.princeton.edu/documentation/wngloss7wn) for the definitions of *sense* and *sensekey*)
* **source_wn_engs**: the *synsets* of the *gold sensekeys*
* **sense rank**: the sense rank of the *gold sensekeys* and *source_wn_engs*.
* **lstm_output**: the synset that the LSTM selected.
* **lstm_acc**: True: the system correctly disambiguated the instance, False: the system made a mistake.
* **emb_freq**: dictionary mapping synset -> information about training data. Value is either:
    * *the integer 0*: no annotated data was available for the synsets
    * *a collections.defaultdict* e.g., 'eng-30-05638987-n': defaultdict(<class 'int'>, {'semcor': 9, 'total': 9}) with information about number of annotated instances for the synset per included corpus
