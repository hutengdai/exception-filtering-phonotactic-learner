# Exception-filtering Phonotactic Learner

## Overview
This is a public repo of the Exception-filtering Phonotactic Learner in Dai (to appear) paper. This tool is designed to model phonotactic constraints and evaluate grammaticality judgments based on user-provided linguistic data. The tool also uses customizable settings to test different datasets and phonotactic theories. I only tested the code on Mac, but the code should be runnable on Windows and Linux as well.

## Prerequisite
Let's first make sure you have Python 3 and pip installed on your system. For example, run this in your command line (Terminal for Mac or Command Prompt for Windows system):
```bash
python3 --version
```
(This checks for Python 3 specifically, as macOS comes with Python 2.7 pre-installed as "python".) If you get error message like "command not found: python3", you probably don't have python properly installed. You can download python from https://www.python.org/downloads/. You can find many video tutorials on the installation of python.


## Installation
To date, the easist and safest way to install pynini is through conda, run:
```bash
conda install -c conda-forge pynini
```
See more details about pynini here: https://www.openfst.org/twiki/bin/view/GRM/Pynini


## Other requirements
You can install all required Python libraries using:
```bash
pip3 install -r numpy pandas matplotlib scipy plotnine
```


## Running the code
The main algorithm is in learner_segment_based.py. Run it from the command line, providing necessary arguments directly. Here’s how to execute the script with all required and optional parameters:
```bash
python learner.py <language> <structure> <max_threshold> [--weight WEIGHT] [--model MODEL]
```

Currently,
- <*language*> can be "toy", "english", "polish", or "turkish", as examined in the paper.
- <*structure*> can be "local" or "nonlocal". When you use "nonlocal", the code automatically selects the vowel tier based on the feature file. (I plan to improve this part of the code in the future.)
- <*max_threshold*> can be any value between 0.01 and 1.
- <*weight*> and <*model*> are both optional arguments, meaning you can ignore them if you just need a quick demonstration. The default value of *weight* is 10 but sometimes lower value can yield higher correlation scores in evaluation.
- The default value of <*model*> is "filtering". When you change it to "gross", you turn the Exception-filtering model to the baseline model in my paper (see section 4).

For example, if you want to train on the English data, run:
```bash
python learner.py english local 0.1 10 filtering
```
Then you can find the result with assigned grammaticality of each word in the testing data at:
```bash
data/english/judgment_struc-local_thr-0.1_pen-10_model-filtering.txt
```
and the interpretable constraints (grammar) in
```bash
data/english/matrix_struc-local_thr-0.1_pen-10_model-filtering.txt
```

If you want to try some new datasets, I recommend first create a copy of "data/english", then change "english" to the name of your language, and change the Features.txt, LearningData.txt, and TestingData.txt to your own dataset. All words/items in your dataset should be space-separated. LearningData.txt should just be a long list of existing words, while TestingData.txt should be novel forms that are not in the LearningData.txt, otherwise your evaluation will be questioned.


(For advanced coders, wfst.py is based on a wrapper of pynini written by Colin Wilson https://github.com/colincwilson/wynini. This is used in the estimation of Expected Frequency. I thank Colin for his support on this implementation.) 

## Support
For issues, questions, or contributions, please contact my email huteng@umich.edu. Please note that my availability will be significantly limited starting September 2024, but I will do my best to respond to your queries as promptly as possible.

Have fun!


