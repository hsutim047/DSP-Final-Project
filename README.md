# DSP-Final-Project

## Dataset
`wget https://www.csie.ntu.edu.tw/~b08201047/timit_data.zip`

## Train Config
Here are the list of changeable parameters

`window_size`

`data_prefix`: the path of the dataset

`method`: average, weight_average, left_concatenate, concatenate, right_concatenate

`seed`

`learning_rate`

`epochs`

## Quick Start
`CUDA_VISIBLE_DEVICES=0 python3 train.py --window_size 5 --method average --epochs 40`
