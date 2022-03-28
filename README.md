# Marginal Representation Learning with Graph Structure Self-Adaptation

### Marginally Structured Representation Learning (MSLR)

The details can be found in the [TNNLS 2018 paper](https://ieeexplore.ieee.org/document/8128909). 

This code has been evaluated on Matlab.


#### Citation:
The program is to evaluate the MSRL algorithm and these codes are for the paper:

Z. Zhang, L. Shao, Y. Xu, L. Liu, J. Yang, Learning Marginal Visual Representation With Graph Structure Self-Adaptation, 
Accepted by IEEE Transactions on Neural Networks and Learning Systems, DOI：***, 2017.


% ===========================================
% File description:
% ===========================================

main_SMSRL: main function for classification including the supervised and semi-supervised MSRL algorithms
utility file:
EProjSimplex_new: update adaptive neighbors from Prof. Feiping Nie
L2_distance_1: NN Classifier with L1 distance from Prof. Feiping Nie
LSR: Least square regression for initialization
SMSRL: The main function of our methods
sperate_data: Seperate the training and test data

SolveHomotopy:   l1-Homotopy: http://users.ece.gatech.edu/~sasif/homotopy/

% ===========================================
% How TO RUN THE CODE
% ===========================================

For image classification:
Please derectly run the matlab file "main_SMSRL.m"

% ===========================================
% OTHERS
% ===========================================
Note: This program only presents the image classification results on the Extended YaleB database

WORK SETTING:
    This code has been compiled and tested by using matlab 7.0 and R2013a

```
@article{zhang2018marginal,
  title={Marginal representation learning with graph structure self-adaptation},
  author={Zhang, Zheng and Shao, Ling and Xu, Yong and Liu, Li and Yang, Jian},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={29},
  number={10},
  pages={4645--4659},
  year={2018},
  publisher={IEEE}
}
```
