# Training Machine Learning or Deep Learning for each task

This folder is for training ML or DL model for each task in case using normal mathematical solution is not effective.

## Structure

- **finger_counting/** - Finger counting model
  - `dataset/` - Training data
  - `models/` - Saved model checkpoints
  - `preprocess.py` - Data preprocessing
  - `train.py` - Training script

- **gesture/** - Hand gesture classification model
  - `pretrain/` - Pretrained weights
  - `resnext.py` - ResNeXt model architecture
  - `main.py` - Training pipeline
  - `test.py` - Model evaluation

- **config.py** - Shared configuration settings
