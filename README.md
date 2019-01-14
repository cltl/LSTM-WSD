## LSTM-WSD

This package contains the code to apply the model from the following paper:
```xml
@InProceedings{C18-1030,
  author = 	"Le, Minh
		and Postma, Marten
		and Urbani, Jacopo
		and Vossen, Piek",
  title = 	"A Deep Dive into Word Sense Disambiguation with LSTM",
  booktitle = 	"Proceedings of the 27th International Conference on Computational Linguistics",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"354--365",
  location = 	"Santa Fe, New Mexico, USA",
  url = 	"http://aclweb.org/anthology/C18-1030"
}
```

## DEMO

run `demo_install.sh` to download resources and install modules
to be able to run the demo. This was tested with Python3.5 and Python3.6.

After running the install script:
* *on das5* see `demo.job` for example

## Replication

run `install.sh` to download resources and install modules
to be able to replicate experiments.

to replicate the **averaging** experiments, run the following bash script on das5:
* cd configs
* bash experiments.sh
