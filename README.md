# Codes-of-Inter-RAT

Official implementation of EMNLP-main 2023 paper  "Interventional Rationalization".

To train Inter-RAT :
```
python train.py \
  --gpu_id=0 \
  --types=train_all \
  --aspect=0 \
  --lr=0.001 \
  --save_path=./output \
  --is_emb=training \
  --embed_dim=100 \
  --batch_size=256 \
  --epochs=20 \
  --model_name=InterRAT \
  --alpha_rationle=0.2 \
  --lstm_hidden_dim=200 \
  --infor_loss=0.05 \
  --regular=0.01 \
  --class_num=2 \
  --seed=42 \
  --abs=1 \

```

## Citation
```
@inproceedings{yue2023interventional,
  title={Interventional Rationalization},
  author={Yue, Linan and Liu, Qi and Wang, Li and An, Yanqing and Du, Yichao and Huang, Zhenya},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={11404--11418},
  year={2023}
}
```
