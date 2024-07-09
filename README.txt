Prerequisites:

Assumes python3 and rasa are already installed.

STRONG RECOMMENDATION: install rasa in a virtual environment. In subsequent command listings, the string "(rasa)" marks that the command must be run
in the virtual environment where rasa is installed.

The venv is needed because some dependencies of rasa conflict with, e.g., the dependencies of the speech to text module whisper. We assume whisper
was also installed in a venv, and mark with "(whisper)" the commands that should be run in the whisper venv.

0. Setting up training data

In the folder of this repository clone, you can edit:

entities.yml: adjust the list of items the user can request/robot can bring.
templates.yml: adjust templates for training/testing examples.

1. Setting up NLP

Open a terminal and go to a location where you can create a folder. Create a folder for a rasa project, for example

mkdir rasa_restaurant
cd rasa_restaurant
(rasa) rasa init

When prompted for a path for the rasa project, just press enter. When prompted whether to train a model, type N then enter.

Open another terminal and go to the folder of this repository clone. Then run the following command to generate training data

python3 ./command_and_test_generator.py -o <path to your rasa project> -Ntrain <number of training examples, try 1000> -Ntest <number of testing examples>

You should also edit the rules.yml and stories.yml in your rasa project's data folder by removing all rules and stories.

In your rasa project, run

(rasa) rasa train

2. Running NLP

In the rasa project folder, run

(rasa) rasa run --enable-api

Once the rasa server is up and running message appears, you can run the activate_language_processing script, e.g. by

(whisper) rosrun activate_language_processing nlp_gpsr.py

