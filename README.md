# MHC Encoding
Special course with Morten Nielsen

Experiments are based on [DeepLigand](https://arxiv.org/abs/1703.10593).
Extensive experiments were performed as to residual network structure and LSTM structure. Other conclusions are drawn of minor and more superficial experiments. <b> Note: </b> All experiments were conducted on binding affinity data only due to computational power available. <br>
<b> Conclusions drawn </b>:
* 5-layer residual network of 'cabd' structure performs the best
* Evaluating residual network outputs in a Gaussian distribution hurts overall performance. 
* Using a fully LSTM structure helps the LSTM. 
* ReZero initialization is not helpful
* Learning rate scheduler does not help (superficial experiments only)

### Best model results

Model | AUC | PCC
--- | --- | ---
Residual Network | 0.927 | 0.803
Ensemble of Residual Network | 0.931 | 0.812
NetMHCPan 3.0 | 0.933 | 0.799

The Residual Network primarily outperforms [NetMHCPan 3.0](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-016-0288-x) on outlier alleles, while the ensembl network fairly consistently outperforms [NetMHCPan 3.0](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-016-0288-x).

## Requirements

```bash
conda create -n MHC_experiments python=3.7
conda activate MHC_experiments
# use the instructions from https://pytorch.org/
conda install pytorch=1.5.1 torchvision cudatoolkit=10.2 -c pytorch 
pip install -r requirements.txt
mkdir ./experiments/
# [Optional] Install Latex (used for the figures)
```

## Train
Example of how to train:
```bash
python main.py --seed 20200904 --lstm_nhidden 128 --lstm_nlayers 5 --full_lstm True
```

