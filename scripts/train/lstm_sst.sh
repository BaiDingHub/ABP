python train_classifier.py --model lstm --dataset sst \
                           --test_path ./data/dataset/sst/sst \
                           --embedding_path ./data/embedding/glove.6B.200d.txt \
                           --batch_size 8 \
                           --max_epoch 70 \
                           --save_path ./data/model/WordLSTM/sst \
                           --lr 0.001 \
                           --max_seq_length 256 \
                           --gpu_id 1
