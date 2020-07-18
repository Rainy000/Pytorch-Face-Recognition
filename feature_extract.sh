## activate pytorch
cd /data/sharedata/wangyang_face
source ./pytorch/bin/activate
cd /data/sharedata/wangyang_face/recognition/icartoon
## set some parameters

IMG_DIR=/data/datas/recognize/icartoon_face/personai_icartoonface_rectest/1/
ROOT_DIR=/data/datas/recognize/icartoon_face


#### glint 
python extract_features.py \
    --gpu-id 4,5,6,7 \
    --batch-size 200 \
    --block irse \
    --input-size 112 \
    --ckpt-path /data/sharedata/wangyang_face/recognition/icartoon/models/scale_1_softmax_mmd_4.tmp \
    --root-path ${IMG_DIR} \
    --save-path ${ROOT_DIR}




