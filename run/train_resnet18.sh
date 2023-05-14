python main_resnet.py \
--data PathForImageNet \
--visible_gpus '0,1,2,3' \
--multiprocessing_distributed True \
--dist_url 'tcp://127.0.0.1:23456' \
--workers 20  \
--arch 'resnet18_quant' \
--batch_size 512  \
--epochs 90 \
--lr_m 0.1 \
--lr_q 0.0001 \
--log_dir "./results/ResNet18_muti-bits" \
--bit_list 2345678
# --groups 2 \
# --weight_bit 2 \
# --act_bit 2 \

python main_cifar.py \
--data PathForCIFAR \
--visible_gpus '3' \
--multiprocessing_distributed False \
--dist_url 'tcp://127.0.0.1:33156' \
--workers 20  \
--arch 'resnet20_quant' \
--batch_size 512  \
--epochs 200 \
--lr_m 0.01 \
--lr_q 0.0001 \
--log_dir "./results/ResNet20_muti-bits" \
--bit_list 2 \
--gpu 0 \
--datasetsname 'cifar100'