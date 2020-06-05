All information in this folder pertains to ADULT experiment, with an exception of DP-auto-GAN MIMIC III folder.

Epsilon values reported at eps=0.36,0.51,1.01. This corresponds to eps=0.5,0.8,1.4 using standard composition for DP-SYN compared to RDP composition of two-phase training. See the paper for details.


## Parameter tuning

In each algorithm, we tried a range of noise multiplier and pick the best one. Details can be found in the paper. Here we record the optimal noise we found for each algorithm and target epsilon.

### DP WGAN:

Noise multipliers tried: 7,9,11,13,15. For eps = 0.35,0.51, also tried 19,23,27.5,35, and for eps=0.35, also tried 40,45,50.

Epsilon | Best noise multiplier
--- | --- 
0.36 | 27.5
0.51 | 19
0.8 | 11
1.01 | 9
1.4 | 7

### DP VAE:

Noise multipliers tried: 1,1.5,...,5 and upto 6.5 for eps=0.51 and to 8.5 for eps=0.36.

Epsilon | Best noise multiplier
--- | --- 
0.36 | 5
0.51 | 4
0.8 | 2.5
1.01 | 2
1.4 | 2

### DP SYN:

Noise multipliers tried: 2,4,6,8.

Epsilon | Best noise multiplier
--- | --- 
0.5 | 4
0.8 | 4
1.4 | 4
