# Stratified-Attention-Dense-Network
This repository is for SADN introduced in the following paper

[ZhiWei Liu](zwliu1982@hotmail.com),[XiaoFeng Mao](m1013272487@163.com),[Ji Huang](timesok@qq.com),[MengHan Gan](1241668480@qq.com) and[YueYuan Zhang](zyyaney1981@hotmail.com), "Stratified Attention Dense Network for Image Super-Resolution".

## Requirement
1. python >= 3.7
2. tensorflow == 1.14
3. numpy == 1.15.4


## Introduction
Stratified Attention Dense Network was proposed to reconstruct high quality HR image.
![1](https://user-images.githubusercontent.com/58931124/119915189-8f967780-bf94-11eb-814e-204c0a439982.png)
Stratified Attention Dense Network

![2](https://user-images.githubusercontent.com/58931124/119935301-de0a3d00-bfb9-11eb-99dc-a5480ad4c468.png)
Attention Dense Module

## Training
1.Download DIV2K training data(800 training + 100 validation images) from  [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

### Begin to train

**Example command is in the file 'demo.txt'.**
```bash
python main.py --train_GT_path F:/ARDN/DataSet/DIV2K_train_HR --train_LR_path F:/ARDN/DataSet/DIV2K_train_LR_bicubic/X2/ --test_GT_path F:/ARDN/DataSet/benchmark/Set5/HR/ --test_LR_path F:/ARDN/DataSet/benchmark/Set5/LR_bicubic/X2/ --test_with_train True --scale 2 --log_freq 1 --model_save_freq 10000 --max_step 10000 --n_ARDG 12 --n_ARDB 12 
```
## Test

### Quick start
1.Download benchmark datasets from [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

1. (optional) Download pretrained models for our paper.

    The Trained x2 models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1PAsqKaOb3q2A5V_Izv7UuVJKJaHswwmw)
2. Run the following scripts.
**Example command is in the file 'demo.txt'.**
```bash
python main.py --mode test --pre_trained_model ./model/ARDN_X2_64_8_8-1000000 --test_GT_path F:/ARDN/DataSet/benchmark/Urban100/HR/ --test_LR_path F:/ARDN/DataSet/benchmark/Urban100/LR_bicubic/X2/ --scale 2 --save_test_result False --test_set Urban100+ --self_ensemble True --chop_forward True
```
## Results
### Quantitative Results
![PSNR](https://user-images.githubusercontent.com/58931124/119939082-e1a0c280-bfbf-11eb-9d3b-1948aa214a6b.png)

### Visual Results
![Results](https://user-images.githubusercontent.com/58931124/119939362-5247df00-bfc0-11eb-8117-7f69fb28e01d.png)

![Results2](https://user-images.githubusercontent.com/58931124/119939818-fb8ed500-bfc0-11eb-89d4-0fce21b66d62.png)


