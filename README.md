# IMERG Frequency Analysis

This repository contains a (very limited amount) of my work for NASA, but it should give a taste of what my project was about. 

In short, my project focused on improving an extreme precipitation statistical modeling framework based on the [IMERG](https://gpm.nasa.gov/data/imerg) satellite-based precipitation retrievals dataset. The goal was to generate accurate predictions for the magnitude and recurrence intervals of extreme precipitation events (i.e. estimating the size of the largest storm in x region in the next y years, or the rarity of events in x region larger than y threshold). In the past, this has been done using the framework of [Extreme Value Theory](https://en.wikipedia.org/wiki/Extreme_value_theory), by fitting an extreme value distribution (such as the Generalized Extreme Value or Generalized Pareto) to the largest values at each location. 

However, it can be difficult to generate accurate estimates for especially rare events (e.g. 50, 100 or perhaps even 1000 year events) given that the length of the IMERG data record is only 20 years. The idea is that the quality of estimates at one location may be improved by utilizing data or other information from other, nearby sites. Clustering together locations in this fashion may result in improved model accuracy. However, the specific method by which the clusters are formed can have a significant impact on the quality of the final results. If clusters are too heterogeneous, the bias introduced by this process may result in poor accuracy.

The clustering methodology I proposed during my internship at NASA was based on the correlations of clustering variables with the homogeneity of extreme precipitation. The weights of the selected clustering variables were then tuned with coordinate search. This methodology had the effect of vastly improving cluster homogeneity compared to previous clustering methods, which improved the accuracy of the fitted models while continuing to avoid issues with creating and evaluating clusters with the same test.

Additionally, by using precomputed L-moments (rather than MLE) to perform model fitting, the overall process required ~50 times less compute resources while simultaneously providing higher spatial resolution. 

![](https://github.com/jxiong21029/IMERG-Frequency-Analysis/blob/main/zhengzhou_ari_plots.png)
