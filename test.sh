CUDA_VISIBLE_DEVICES=3 python ./tools/train_net.py --config-file ./configs/Market1501/center-bagtricks_R18-softmax.yml --eval-only \
MODEL.WEIGHTS "./logs_final/market1501/R18-softmax_n512_o512_twofc_OldCenterbn/model_final.pth" TEST.OLDRATE 50 
