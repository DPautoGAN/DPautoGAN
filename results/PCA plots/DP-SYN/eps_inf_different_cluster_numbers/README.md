The DP-EM in DP-SYN chooses the number of cluster by the Calinski-Harabasz criterion. This is by default and is used in all DP-SYN experiments, including in the PCA plots.

For eps=inf, we found that points are clustered in one flat band. We also tried DP-EM by forcing a different number of clusters.
The two numbers in each text file (which are the same) are the number of clusters.

We do not find any difference even trying very large number of clusters. 

Adding noise makes DP-EM seems to select more clusters at different locations, maybe due to noisy autoencoder encoding data into a less concentrated way, or DP-EM scatters the center of clusters in a more noisy way. 
