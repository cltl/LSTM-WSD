# Running an experiment

For each experiment, a config file needs to be created.
The config files are stored in the folder **configs**.
For example, the content of **configs/synset-se2-semcor.json** is:

```json
{"label": "synset---se2---semcor",
"level": "synset",
"batch_size" : 300,
"competition": "se2",
"wn_version": "30",
"corpora": ["semcor"],
"label_propagation": false,
"wsd_technique": "averaging"}

```

* **label** this is the label of the experiment. The output of this experiment will be stored in the folder **output/LABEL**
* **level** either *synset* or *sensekey*. This determines whether the candidate meanings of a target lemma will be represented at the synset-level or at the sensekey-level.
* **batch_size**: determines the batch size for the creation of the sensekey/synset embeddings (is used in **wsd_class.py method 'apply_on_lstm_input_file'**)
* **competition**: indicates which senseval/semeval is evaluated on (is needed for evaluation)
* **wn_version**: WordNet version: always "30"
* **corpora**: annotated corpora used to create sensekey/synset embeddings: ["semcor"] or ["omsti"] or ["semcor", "omsti"]
* **label_propagation**: [NOT IMPLEMENTED] hence always false
* **wsd_technique**: most stable is **averaging**.


Running an experiment is then performed by calling the **bash** script **one_experiment.sh** (see **experiments.sh** for how to call it on das5), which itself calls four python scripts
* **preprocess_wsd_df.py**: performs lookup to determine candidate senses/synsets for each target lemma. **wsd_output.p** is created containing all relevant experiment information.
* **lstm_input.py**: creates a file in which the annotated data is formatted such that we use the LSTM to create sense/synset embeddings
* **compute_sense_embeddings.py**: the sense/synset embeddings are created using the LSTM
* **perform_wsd.py**: WSD is performed using information from all the previous steps (Please note that **wsd_output.p** (which contains the WSD output) is overwritten in this step)

**Tip**: almost all paths are defined in **load_utils.py method **update_settings_with_paths**.