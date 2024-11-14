# Toxicity Prediction Using Deep Learning

## Project News & Status Quo
This is still a work in progress! Stay tuned for more updates

## About
This work is my attempt at replicating the approach outlined in "Toxicity Prediction using Deep Learning" by Thomas Unterthiner et al., where deep neural networks are applied to predict chemical toxicity. Their work demonstrated the effectiveness of multi-task learning and feature learning in toxicity prediction, setting a new standard in the field. You can find their paper [here](https://arxiv.org/abs/1503.01445). 

I'm also using the data from the tox21 dataset, obtained [here](http://bioinf.jku.at/research/DeepTox/tox21.html)

# The Dataset
The Tox21 dataset contains 12,060 training samples and 647 test samples that represent chemical compounds. Moreover we have 801 dense features that represent chemical descriptors (molecular weight, Aromaticity, Bertz Complexity Index, etc.) as well as 272,776 sparse features in matrix market format that represent chemical substructures such as ECFP10.

# High-Level Approach
This approach uses deep neural networks (DNNs) to predict toxicity, leveraging DNNs' ability to automatically learn complex features. The model employs multi-task learning, allowing it to predict multiple toxicity types simultaneously by sharing representations across tasks. This is especially useful when labeled data is sparse, as it can benefit from correlations between different toxicity types.


# References

[Mayr2016] Mayr, A., Klambauer, G., Unterthiner, T., & Hochreiter, S. (2016). DeepTox: Toxicity Prediction using Deep Learning. Frontiers in Environmental Science, 3:80.

[Huang2016] Huang, R., Xia, M., Nguyen, D. T., Zhao, T., Sakamuru, S., Zhao, J., Shahane, S., Rossoshek, A., & Simeonov, A. (2016). Tox21Challenge to build predictive models of nuclear receptor and stress response pathways as mediated by exposure to environmental chemicals and drugs. Frontiers in Environmental Science, 3:85.

Unterthiner, Thomas, et al. "Toxicity Prediction Using Deep Learning." ArXiv, 2015,  https://arxiv.org/abs/1503.01445. Accessed 9 Nov. 2024.