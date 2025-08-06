# Python code for "Efficient CSI Feedback in FDD Massive MIMO"
(c) 2025 Ran Chen e-mail: ranchen12979@gmail.com

## Introduction
This repository contains the original models described in 
Ran Chen, “Efficient CSI Feedback in FDD Massive MIMO", as the result of  postgraduate individual project

## Requirements
- torch==1.12.0+cu116
- torchvision==0.13.0+cu116
- numpy==1.21.0
- scipy==1.7.0
- matplotlib==3.5.0
- pandas==1.3.0
- h5py==3.6.0
- omegaconf==2.3.0
- tqdm==4.62.0
- seaborn==0.13.2
- pyyaml==6.0.2
- fvcore==0.1.5
- thop==0.1.1
- ptflops==0.7.3
- torchinfo==1.8.0
- accelerate==1.0.1
- safetensors==0.5.3

## Steps to start

### Step1. Data Preparation
After getting the data, put the data as shown below.
```
F:\CRLSTMNet\data\cost2100\
├── indoor_20slots/
│   ├── H_user_t1_32all.mat
│   ├── H_user_t2_32all.mat
│   └── ...
└── outdoor_20slots/
    ├── H_user_t1_32all.mat
    └── ...
```
step 2:quick evaluation for the result of indoor_cr64 and indoor _cr32 pre-trained model, now, it's ready to run any *.py to get the results (i.e., CRLSTMNet in Table I of my paper).

python evaluate_model_enhanced.py --model "checkpoints\indoor_cr64_enhanced/indoor_cr1_64_enhanced_20250727_2009_final_model.pth" --config configs/indoor_cr64.yaml --create_visualizations --comprehensive

python evaluate_model_enhanced.py --model "checkpoints\indoor_cr32_enhanced/indoor_cr1_32_enhanced_20250729_1519_final_model.pth" --config configs/indoor_cr32.yaml --create_visualizations --comprehensive

for specific basic evaluation :
python evaluate_model_enhanced.py --model "model path/to/model.pth" --config configs/model.yaml --create_visualizations --comprehensive


### Step3. basic training
Train with specific compression ratio
python train_main.py --config configs/indoor_cr32.yaml

Train with default configuration (indoor cr_1/32)
python train_main.py --config configs/base.yaml

Enable complexity analysis
python train_main.py --config configs/indoor_cr64.yaml --analyze_complexity


## Result
The following results are reproduced from Table I of my paper:

|   gamma  |  Methods  | Indoor |            | Outdoor |         |
|:--------:|:---------:|:------:|:----------:|:-------:|:-------:|
|          |           |  NMSE  |     rho    |  NSME   |   rho   |
|    1/4   | LASSO     |  -7.59 |    0.91    |  -5.08  |  0.82   |
|          | BM3D-AMP  |  -4.33 |     0.8    |  -1.33  |  0.52   |
|          | CRNet     | -26.99 |        |  -8.04   |     |
|          | CsiNet    | -17.36 |   0.99   |  -8.75  |  0.91   |
|          | CRLSTMNet | -8.26 |    0.923    |  -0.45  |  0.396  |
|   1/16   | LASSO     |  -2.72 |     0.7    |  -1.01  |  0.46   |
|          | BM3D-AMP  |  0.26  |    0.16    |  0.55   |  0.11   |
|          | CRNet     |  -11.35 |        |  -5.44  |     |
|          | CsiNet    |  -8.65 |    0.93    |  -4.51  |  0.79   |
|          | CRLSTMNet |  -8.19 |    0.921    |  
|   1/32   | LASSO     |  -1.03 |    0.48    |  -0.24  |  0.27   |
|          | BM3D-AMP  |  24.72 |    0.04    |  22.66  |  0.04   |
|          | CRNet     |  -8.93 |        |  -3.51   |     |
|          | CsiNet    |  -6.24 |    0.89    |  -2.81  |   0.67  |
|          | CRLSTMNet |  **-8.03** |    **0.918**    |  -0.21  |  0.381  |
|   1/64   | LASSO     |  -0.14 |    0.22    |  -0.06  |  0.12   |
|          | BM3D-AMP  |  0.22  |    0.04    |  25.45  |  0.03   |
|          | CRNet     |  -6.49  |        |  -2.22   |    |
|          | CsiNet    |  -5.84 |    0.87    |  -1.93  |  0.59   |
|          | CRLSTMNet |  **-7.89** |    **0.916**    |  


## Remarks
1.If time permitted, prepare 100% covering energy of dataset again, which can highly increase indoor performance when Compression Ratio is high, this result in high CR is limitted caused by not good enough dataset.
2.The outdoor performance is not good ,still work on it, try indoor test have already made progress than other DL models.
3.Haven't finished all outdoor data training, I will update this model at: https://github.com/Takazacky-chan/CRLSTMNet

