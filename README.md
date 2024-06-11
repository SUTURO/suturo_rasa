# SUTURO RASA
This repository contains the files for the rasa action server used in [suturo_nlp](https://github.com/SUTURO/suturo_nlp).

### Installing rasa
[Suturo wiki: Rasa](https://github.com/SUTURO/SUTURO-documentation/wiki/Rasa)

### How to use:
Prerequisites:

Assumes python3 and rasa are already installed. Open a terminal, and type in the following commands, one by one.

sudo pip install nltk
python3
import nltk
nltk.download('punkt')
exit()

Once the above is done, here is the generate-train-test procedure:

Open a terminal and go to this folder. Then run the following commands, one by one:

rasa init
python3 ./command_and_test_generator.py
rasa train
rasa run --enable-api

For automated testing, open a different terminal and go to this folder. Then run
python3 ./testing.py

The results are in errors.log.

