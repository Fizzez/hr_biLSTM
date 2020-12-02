This is an implementation of instantaneous pulse rate estimation algorithm described in:
> Xu, Ke, et al. "Deep Recurrent Neural Network for Extracting Pulse Rate Variability from Photoplethysmography During Strenuous Physical Exercise." 2019 IEEE Biomedical Circuits and Systems Conference (BioCAS). IEEE, 2019.

The dataset comes from the author's [github](https://github.com/WillionXU/CIME-PPG-dataset-2018/).

Please refer to Jupyter Notebooks in `/notebooks` for dataset pre-processing details.
They are arranged in the following order:
`data_pre_processing.ipynb` -> `label_making.ipynb`.

A multi-layer bi-directional LSTM is implemented to classify the pulse boundaries. Please refer to `train.py` for details.

TODO:
- Not sure the meaning of the output layer - softmax, add a fully connected layer before.
- Should this structure be seq2seq?
- Train model on GCP
