# IMERG Frequency Analysis

This repository contains a (very limited amount) of my work for NASA, but it should give a taste of what my project was about. 

In short, my project focused on improving an extreme precipitation statistical modeling framework based on IMERG precipitation data. I applied an alternative methodology for the regional modeling based on the correlations of  clustering variables with the cluster homogeneity of extreme precipitation. The weights of the selected clustering variables were then tuned with coordinate search. This methodology had the effect of significantly increasing the accuracy of the fitted models while avoiding ssues with creating and evaluating clusters with the same homogeneity test.

Additionally, by using precomputed L-moments (rather than MLE), the model fitting process can be completed ~50 times faster while providing higher spatial resolution and much better results in low-precipitation regions.
