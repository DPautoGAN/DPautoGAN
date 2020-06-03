The mu in mu-smooth KL divergence controls the balance between missing a minority class and accuracy of overall population. 
The smaller the mu, the stronger penality of missing class and less emphasis on overall accuracy.

We pick mu = exp(-1/(1-p_1)) where p_1 is the fraction of the majority class to normalize the divergence score across features to a constant around 1 when p_1 is large and synthetic data output only the majority class. This seems to balance the diversity and accuracy well.

Penalty term P is set to 1, and for other penalty values, the corresponding mu is exp(-P/(1-p_1)). This roughly set the worse-case score across features to around P.
