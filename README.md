# TextGCAT

This implementation highly based on official code [yao8839836/text_gcn](<https://github.com/yao8839836/text_gcn>).

## Require

* Python 3.6
* PyTorch 1.0

## Running training and evaluation

1. `cd ./preprocess`
2. Run `python remove_words.py <dataset>`
3. Run `python build_graph.py <dataset>`
4. `cd ..`
5. Run `python train.py <dataset>`
6. Replace `<dataset>` with `20ng`, `R8`, `R52`, `ohsumed` or `mr`

## Model
Pre-tained model placed in ‘model_save’.

## Acknowledge
This work was supported by the National Key R&D Program of China under Grant No. 2020AAA0103804(Sponsor: <a  href ="https://bs.ustc.edu.cn/chinese/profile-74.html">Hefu Liu</a>). This work belongs to the University of science and technology of China.
