GPUID=$1
CONFIG=$2
WORK_DIR=$3

for i in 10 15 20 30 35 45
do
    CUDA_VISIBLE_DEVICES=$GPUID python tools/test.py ${CONFIG} --work-dir=${WORK_DIR} ${WORK_DIR}/epoch_$i.pth --metrics pa-mpjpe mpjpe
done