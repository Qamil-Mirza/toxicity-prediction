# Toxicity Prediction Using Deep Learning

## Project News & Status Quo
This is still a work in progress! Stay tuned for more updates:

[x] - Add Hyperparameter Tuning Logic
[ ] - Add logging to train.py
[ ] - Add hyperparameters to cv_results for ease of analysis
[ ] - Write plotting logic for model loss
[ ] - write plotting logic for test mean auc
[ ] - Add additional Hyperparameters for flexibility
[ ] - Write Notebook exploring sparse features
[ ] - Integrate sparse features into pipeline
[ ] - Dockerize
[ ] - Cleanup documentation

## About
This work is my attempt at replicating the approach outlined in "Toxicity Prediction using Deep Learning" by Thomas Unterthiner et al., where deep neural networks are applied to predict chemical toxicity. Their work demonstrated the effectiveness of multi-task learning and feature learning in toxicity prediction, setting a new standard in the field. You can find their paper [here](https://arxiv.org/abs/1503.01445). 

I'm also using the data from the tox21 dataset, obtained [here](http://bioinf.jku.at/research/DeepTox/tox21.html)

## The Dataset
The Tox21 dataset contains 12,060 training samples and 647 test samples that represent chemical compounds. Moreover we have 801 dense features that represent chemical descriptors (molecular weight, Aromaticity, Bertz Complexity Index, etc.) as well as 272,776 sparse features in matrix market format that represent chemical substructures such as ECFP10.

## High-Level Approach
This approach uses deep neural networks (DNNs) to predict toxicity, leveraging DNNs' ability to automatically learn complex features. The model employs multi-task learning, allowing it to predict multiple toxicity types simultaneously by sharing representations across tasks. This is especially useful when labeled data is sparse, as it can benefit from correlations between different toxicity types.

## Project Folders/Files Explanation
- docs: Contains additional documentation on the toxicity prediction project
- notebooks: Jupyter notebooks containing my initial exploration of the data
- src/logs: Contains training, testing, and cross validation logs
- src/models: Contains model architecture as well as a Scaler and Imputer fitted on the data
- src/plots: WIP
- src/results: Records AUC scores across the 12 tasks for the final testing round as well as cross validation
- src/utils/loggers.py: Function to setup multiple loggers for different parts of the model development pipeline
- src/utils/processData.py: Functions that server to process data for training and testing the model
- src/utils/visualize.py: WIP
- config.yaml: Contains project run configurations (hyperparameters, cross-validation settings, imputation/scaling strategies, data paths)
- src/main.py: Contains model training logic and saves model to the current directory
- src/test.py: Contains model testing logic and saves AUC on the 12 tasks to results in csv format
- src/train.py: Contains functions that support main.py in model training
- src/tune_and_validate.py: Contains cross-validation logic and stores the results of cross validation to results in csv format

# Usage
First, `cd` into src and run the following commands to execute different parts of the model pipeline:

- train model: `python main.py`
- test model: `python test.py`
- cross-validation: `python tune_and_validate.py`

To adjust hyperparameters, simply edit the `config.yaml` file, adjusting the hyperparameters to your liking (more hyperparameters coming soon).

# References

[Mayr2016] Mayr, A., Klambauer, G., Unterthiner, T., & Hochreiter, S. (2016). DeepTox: Toxicity Prediction using Deep Learning. Frontiers in Environmental Science, 3:80.

[Huang2016] Huang, R., Xia, M., Nguyen, D. T., Zhao, T., Sakamuru, S., Zhao, J., Shahane, S., Rossoshek, A., & Simeonov, A. (2016). Tox21Challenge to build predictive models of nuclear receptor and stress response pathways as mediated by exposure to environmental chemicals and drugs. Frontiers in Environmental Science, 3:85.

Unterthiner, Thomas, et al. "Toxicity Prediction Using Deep Learning." ArXiv, 2015,  https://arxiv.org/abs/1503.01445. Accessed 9 Nov. 2024.