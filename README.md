# GIP
This is the repo for our paper: Enhancing Graph Self-Supervised Learning with Graph Interplay
## Basic environment settings
```
python=3.8
pytorch=2.2.0
pygcl=0.1.2
torch-geometric=2.5.3
```

##Training
```
#Test on IMDB-M dataset
cd run_code
python grace_g.py --num_layers 5 --did 2  --epoch 200  --aug_type=GIP
```
