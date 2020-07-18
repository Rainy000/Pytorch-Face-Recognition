
python train.py \
    --block irse \
    --loss softmax \
    --gpu-id 2,3,4,5,6,7 \
    --batch-size 20 \
    --source-num-class 5013 \
    --feat-scale 64 \
    --cos-m 0.48 \
    --factor 5 \
    --lr 0.001 \
    --lr-step 10,20 \
    --input-size 112 \
    --max-step 30 \
    --start-epoch 0 \
    --save-period 10 \
    --save-path ./models \
    --save-name scale_1_softmax_mmd \
    --log-dir ./log/scale_1_softmax_mmd.log \
    --source-dir /data/datas/recognize/icartoon_face/personai_icartoonface_rectrain/1/train_set \
    --target-dir /data/datas/recognize/icartoon_face/personai_icartoonface_rectest/1 \
    --resume \
    --feat-model ./models/scale_1_softmax_40.tmp \
    --src-model ./models/scale_1_softmax_model_fc_40.tmp 

