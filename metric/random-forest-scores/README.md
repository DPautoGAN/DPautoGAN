If DP-auto-GAN generated data only keep discrete columns to have the same formatting as DP-WGAN, the accuracy drops to 70.65%-73.98%. 
This is likely due to DP-auto-GAN being required to spread privacy budget to generate other continuous columns and also receive additional noise injection from other columns during the training.
