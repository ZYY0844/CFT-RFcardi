# CFT-RFcardi

Official implementation of:

[From High-SNR Radar Signal to ECG: A Transfer Learning Model with Cardio-Focusing Algorithm for Scenarios with Limited Data](https://ieeexplore.ieee.org/document/11216086)

## Citation

If you find our work helpful for your research, please cite our paper:
```
@article{zhang2024radarODE,
  title={{From high-SNR radar signal to ECG: A transfer learning model with cardio-focusing algorithm for scenarios with limited data}}, 
  author={Yuanyuan Zhang and Haocheng Zhao and Sijie Xiong and Rui Yang and Eng Gee Lim and Yutao Yue },
  journal={IEEE Transactions on Mobile Computing},
  year={2025},
  publisher={IEEE},
  month={Oct.}
}
```

## Run the Model
You can find the arguments and settings in:

```shell
RFcardi_Transfer_Learning/Projects/radarODE_transfer/main.py
```

A validation example is in 
```shell
RFcardi_Transfer_Learning/Projects/radarODE_transfer/utils/visualization.ipynb
```

## Dataset Download and Preparation

Dataset for RFcardi training (including spectrogram of radar inputs (sst), ECG ground truth and sparse signal ground truth (anchor))

```shell
https://drive.google.com/file/d/1Xv03591LUCHwZmTxPxYn0zPKWKMbwK2X/view?usp=sharing
```

Dataset for the CFT reuqires the orignal radar output. A quick validation can be performed using example data
```shell
https://drive.google.com/file/d/14IG14XCYOf5oE9WRu38fSTcF6vWnfQx8/view?usp=sharing
```

The full dataset (50GB) can be downloaded from
```shell
https://pan.baidu.com/s/150R0nsRdXp1dHC12xQR8cQ?pwd=gcq4
```
with original ADC data ``.bin file``, processed FMCW data structure ``.npy file``, (chirp, frame, antenna).

##

:partying_face: Any problem please send them in Issues or Email [:email:](yuanyuan.zhang16@student.xjtlu.edu.cn).