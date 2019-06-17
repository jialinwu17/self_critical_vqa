# Self_Critical_VQA
This repo contains codes for ''Self-Critical Reasoning  for Robust Visual Question Answering'' with VQA-X human textual explanations
This repo contains code modified from [here](https://github.com/SinghJasdeep/Attention-on-Attention-for-VQA), many thanks! 

## Prerequisites
Python 3.7.1 <br>
PyTorch 1.1.0 <br>
spaCy (we use en_core_web_lg spaCy model) <br>
h5py, pickle, json, cv2 <br>

## Preprocessing
Please download the detection features from this [google drive](https://drive.google.com/drive/folders/1IXTsTudZtYLqmKzsXxIZbXfCnys_Izxr?usp=sharing) and put it to 'data' folder <br>
Please run ``bash tools/download.sh`` to download other useful data files including VQA QA pairs and Glove embeddings <br>
Please run ``bash tools/preprocess.sh`` to preprocess the data <br>
``mkdir saved_models``

## Training
The training propocess is split to three stage :<br>
(1) Pretrain on VQA-CP train dataset by runnning <br>
``CUDA_VISIBLE_DEVICES=0 python main.py --load_hint -1 --use_all 1 --learning_rate 0.001 --split v2cp_train --split_test v2cp_test --max_epochs 40`` <br>
After the pretraining you will have a saved model in ``saved_models`` named by the start training time. <br><br>
(2) Pretrain using the influential strengthening loss <br>
Here, please replace the 86-th line in the ``train.py`` with your VQA-CP pretrained models. <br>
Then, please run the following line to strengthen the most influential object. <br>
``CUDA_VISIBLE_DEVICES=0 python main.py --load_hint 0 --use_all 0 --learning_rate 0.00001 --split v2cp_train_vqx --split_test v2cp_test --max_epochs 12 --hint_loss_weight 20``<br>
After the pretraining you will have anthor saved model in ``saved_models`` named by the start training time. <br><br>
(3) Training with the self-critical objectives. <br>
Here, please replace the 82-th line in the ``train.py`` with your influence strengthened pretrained models. <br>
Then, please run the following line for training. <br>
``CUDA_VISIBLE_DEVICES=0 python main.py --load_hint 1 --use_all 0 --learning_rate 0.00001 --split v2cp_train_vqx --split_test v2cp_test --max_epochs 5 --hint_loss_weight 20  --compare_loss_weight 1500``<br>
