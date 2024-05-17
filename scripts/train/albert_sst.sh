model="albert"
dataset="sst"
test_path="./data/dataset/sst/sst"
nclasses=2
batch_size=8
max_epoch=10
save_path="./model/${model}/${dataset}"
lr=0.000005
lr_decay=0.97
max_seq_length=256
# bert_model_name="albert-base-v1"
pretrained_dir="/home/yuzhen/nlp/pretrained/albert-base-v2"
gpu_id=1

training_start_params="--model ${model} --dataset ${dataset} \
--test_path ${test_path} \
--nclasses ${nclasses} \
--batch_size ${batch_size} \
--max_epoch ${max_epoch} \
--save_path ${save_path} \
--lr ${lr} \
--lr_decay ${lr_decay} \
--max_seq_length ${max_seq_length} \
--pretrained_dir ${pretrained_dir} \
--gpu_id ${gpu_id}"

python train_classifier.py ${training_start_params}