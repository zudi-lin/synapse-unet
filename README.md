# Synapse-unet
A three dimensional U-net for synaptic cleft detection from electron microscopy (EM) images. 

### Network Architecture
![image](https://github.com/zudi-lin/synapse-unet/raw/master/img/unet_synapse.png)

### Loss Function
We balance the extreme class imbalance by using:
<div align="center">
<img width="450" alt="" src="https://github.com/zudi-lin/synapse-unet/raw/master/img/loss_function.png" />,
</div>
where

### Dataset
Training and testing data comes from MICCAI Challenge on Circuit Reconstruction from Electron Microscopy Images ([CREMI challenge](https://cremi.org)). Three training volumes of adult *Drosophila melanogaster* brain imaged with serial section Transmission Electron Microscopy (ssTEM) are provided.
