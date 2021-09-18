# IMERG Frequency Analysis

This repository contains a (very limited amount) of my work for NASA, but it should give a taste of what my project was about. 

In short, my project focused on improving an extreme precipitation statistical modeling framework based on IMERG precipitation data. I applied an alternative methodology for the regional modeling based on the correlations of  clustering variables with the cluster homogeneity of extreme precipitation. This methodology had the effect of significantly increasing average precipitation homogeneity, while avoiding overfitting and/or double-sampling of the homogeneity statistic (avoiding issues with creating and evaluating clusters with the same test).

Next, a coordinate hyperparameter search for clustering weights was used to slightly further increase homogeneity. 

Additionally, by using precomputed L-moments (rather than MLE), the model fitting process can be completed ~50 times faster while providing higher spatial resolution and accuracy (especially in low-precipitation regions)
