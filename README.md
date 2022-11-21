# Adaptive Channel Mixing (ACM) GNN Framework

Revisiting Heterophily for Graph Neural Networks (Sitao Luan *et al.*, NeurIPS 2022): [https://arxiv.org/abs/2210.07606](https://arxiv.org/abs/2210.07606)

Homophily metrics             |  t-SNE visualization of GCN and ACM-GCN layers trained on Squirrel
:-------------------------:|:-------------------------:
![](https://github.com/SitaoLuan/Adaptive-Channel-Mixing-GNN/blob/main/plots/fig_bipartite.jpg)  |  ![](https://github.com/SitaoLuan/Adaptive-Channel-Mixing-GNN/blob/main/plots/ACM_output_layer.jpg)

## Repository Overview
We provide the PyTorch implementation for ACM-GNN framework here. The repository is organised as follows:

```python
|-- ACM-PyTorch # experiments on 10 small-scale datasets
    |-- experiment/ # experiment bash script 
    |-- models/ # model definition
    |-- splits/ # split files for datasets
    |-- arg_parser.py  # the argument parser code for training/hyperparameter searching script
    |-- hyperparameter_searching.py # the hyperparameter searching script
    |-- logger.py # the logger code
    |-- train.py # the model trainig script
    |-- utils.py # data process and others
|-- ACM-Geometric # experiments on the large-scale and small-scale datasets based on the data provided by LINKX
    |-- sh/ # experiment bash script
    |-- large_scale_data/ # dataset folder
    |-- train.py # the model trainig script
    |-- parse.py  # set hyper-parameters
|-- data/ # 3 old datasets, including cora, citeseer, and pubmed
|-- new-data/ # 6 new datasets, including texas, wisconsin, cornell, actor, squirrel, and chameleon
|-- synthetic-experiments # generate synthetic features and graphs with different homophily levels and train baseline models
    |-- baseline_models/ # models and layers of baseline models
    |-- feature_generation.py  # generate node features with base datasets
    |-- graph_generation.py  # generate graphs with different homophily levels
    |-- train.py # the training code
    |-- utils.py # data process and others
    |-- hyperparameter_searching.py # searching the hyperparameters of the baseline model
    |-- logger.py # logger
    |-- homophily.py # different homophily metrics
|-- plots # all experimental plots and visualizations in our paper
```

## Dependencies

The script has been tested running under Python 3.7.4, with the following packages installed (along with their dependencies):

- `dgl-cpu==0.4.3.post2`
- `dgl-gpu==0.4.3.post2`
- `ogb==1.3.1`
- `numpy==1.19.2`
- `scipy==1.4.1`
- `networkx==2.5`
- `torch==1.5.0`
- `torch-cluster==1.5.7`
- `torch-geometric==1.6.3`
- `torch-scatter==2.0.5`
- `torch-sparse==0.6.6`
- `torch-spline-conv==1.2.0`

In addition, CUDA 10.2 has been used.

```
pip install -r requirements.txt
```

## Running Experiments (ACM-PyTorch)

### Training or hyperparameter searching

```
# training with default hyperparameters (e.g. ACM-GCN+ on Texas)
python train.py --model acmgcnp --dataset_name texas

# training with user defined hyperparameters
python train.py --model acmgcnp --dataset_name texas --lr 0.06 --weight_decay 0.0006 --dropout 0.6

# hyperparameter searching (learning rate, weight_decay & dropout)
python hyperparameter_searching.py --model acmgcnp --dataset_name texas
```
The training/hyperparameter seacrhing logs are saved into the `logs/` folder located at `<your-working-directory>/Adaptive-Channel-Mixing-GNN/ACM-PyTorch/logs`.

## Running Experiments (ACM-Geometric)

Create a `results/` folder at `<your-working-directory>/Adaptive-Channel-Mixing-GNN/ACM-Geometric`. Experimental results (.csv) would be saved here.

Download the `data/` folder from the repository of [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) to `<your-working-directory>/Adaptive-Channel-Mixing-GNN/ACM-Geometric/large_scale_data`.

```
# training with default hyperparameters (e.g. ACM-GCN+ on twitch-gamer)
python train.py --model acmgcnp --dataset_name "twitch-gamer"

# training with user defined hyperparameters
python train.py --model acmgcnp --dataset_name "twitch-gamer" --lr 0.002 --weight_decay 0.0006 --dropout 0.6

# hyperparameter searching
bash sh/run_all_settings.sh

```

## Synthetic Benchmark Experiments 

### 1. Generate features & graphs

```
# generate features
python feature_generation.py --base_dataset cora

# generate (random or regular) graphs
python graph_generation.py --graph_type random --degree_intra 2 --num_graph 10
```
Generated features are saved into the `synthetic_graphs/features/` folder located at `<your-working-directory>/Adaptive-Channel-Mixing-GNN/synthetic-experiment/synthetic_graphs/`.

Generated graphs are saved into the `synthetic_graphs/<random|regular>/` folder located at `<your-working-directory>/Adaptive-Channel-Mixing-GNN/synthetic-experiment/synthetic_graphs/<random|regular>`.

### 2. Training or hyperparameter searching

```
# training with user defined hyperparameters
python train.py --model_type acmgcn --base_dataset cora --graph_type random --edge_homo 0.1 --degree_intra 2 --num_graph 10 --lr 0.05 --weight_decay 0.0005 --dropout 0.5

# hyperparameter searching (weight_decay & dropout)
python hyperparameter_searching.py --model_type acmgcn --graph_type random --base_dataset cora --edge_homo 0.1 --edge_homo 0.1 --degree_intra 2 --num_graph 10 

```
Graph generation and training logs are saved into the `logs/` folder located at `<your-working-directory>/Adaptive-Channel-Mixing-GNN/synthetic-experiment/logs`.

## Papers With Code Leaderboard

See the leaderboards for node classification on heterophilic graphs [here](https://paperswithcode.com/task/node-classification-on-non-homophilic). Feel free to add your results or create new benchmark datasets.

## Attribution
Parts of the code are based on
- [GCN](https://github.com/tkipf/pygcn)

- [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale)

## Reference
If you make advantage of the ACM framework in your research, please cite the following in your manuscript:

```
@article{luan2022revisiting,
  title={Revisiting Heterophily For Graph Neural Networks},
  author={Luan, Sitao and Hua, Chenqing and Lu, Qincheng and Zhu, Jiaqi and Zhao, Mingde and Zhang, Shuyuan and Chang, Xiao-Wen and Precup, Doina},
  journal={Conference on Neural Information Processing Systems},
  year={2022}
}
```

## License
MIT
