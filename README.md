## MultiQuant: A Novel Multi-Branch Topology Method for Arbitrary Bit-width Network Quantization
## Dependence
* See requirements.txt


## Usage 

### 2-, 4-, 6-, 8-bit settings

#### Usage of CIFAR100 

```
python main_cifar.py --data PathForCIFAR \
 --visible_gpus '0' --multiprocessing_distributed False \
 --dist_url 'tcp://127.0.0.1:23456' \
 --workers 4  --datasetsname cifar100 \
 --arch 'resnet20_quant' \
 --batch_size 128  \
 --epochs 200 --lr_m 0.1 --weight_decay 0.0001 \
 --log_dir PathForSave --bit_list 2468 --gpu 0 
```

#### Usage of ImageNet

For resnet:

```
python main_resnet.py --data PathForImageNet \
--visible_gpus '0,1,2,3' --multiprocessing_distributed True \
--dist_url 'tcp://127.0.0.1:23456' --workers 16  \
--arch 'resnet18_quant' --batch_size 256  \
--epochs 90 --lr_m 0.1 --weight_decay 0.0001 \
--log_dir PathForSave --bit_list 2468
```

or for Mv1:


```
python main_mv1.py --data PathForImageNet \
--visible_gpus '0,1,2,3' --multiprocessing_distributed True \
--dist_url 'tcp://127.0.0.1:23456' --workers 16  \
--arch 'mv1_quant' --batch_size 256  \
--epochs 90 --lr_m 0.1 --weight_decay 0.0001 \
--log_dir PathForSave --bit_list 2468
```

### 2-, 3-, and 4-bit settings

1, Open the main_resnet.py

2, Comment lines 25 and 26. Uncomment 27, 28

3, run:

```
python main_resnet.py --data PathForImageNet \
--visible_gpus '0,1,2,3' --multiprocessing_distributed True \
--dist_url 'tcp://127.0.0.1:23456' --workers 16  \
--arch 'resnet18_quant' --batch_size 256  \
--epochs 90 --lr_m 0.1 --weight_decay 0.0001 \
--log_dir PathForSave --bit_list 234
```

##  Trained models
[here](https://drive.google.com/drive/folders/1gF_t_v3ReUkhsrEtSQpkPjOHUyG_Y7E0?usp=sharing)


##  Acknowledgments

Code is implemented based on [PalQuant](https://github.com/huqinghao/PalQuant/). We are very grateful for their excellent work.
