# Natural Language Processing
Natural Language Processing Coursework - Imperial College London Computing, Feb 2026.

## Guide to repo structure
* The best performing model - including model weightings, training arguments, config, etc - and its code (`train.py`) are stored in `/BestModel`.
* Code for exploratory data analysis and local / global evaluation is stored in `/src`.
* `dev.txt` and `test.txt` (dev set and test set predictions respectively) are stored in `/predictions`.
* The various CSVs and TSVs outlining the dev set, training set, test set etc are stored in `/data`.

## Guide to reproducing my results
* Run this command in your terminal: `git clone https://github.com/RhianGit/NaturalLanguageProcessing` or download this repo and upload it to Google Colab - use a GPU runtime for faster results.
* To train the model, run the file `/BestModel/train.py`.
* For global or local evaluation, run the files `/src/global_evaluation.py` or `/src/local_evaluation.py` respectively.
* To check my exploratory data analysis results, run the file `/src/exploratory_data_analysis.py`.
