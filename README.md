# melody-generation-RNN

## Setup
```bash
cd melody-generation-RNN/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
There are 3 models to choose from: RNN, LSTM, and GRU. To train your own model, you can specify the `model_type` and 
set `train_flag=True` in the `main.py` parameters. Hyperparameters for the model can be modified in the `hyperparameters.py` file.
Then, you can run the training script to train the model. The model will be saved in the `output` directory.
### Training and Evaluation
```bash
python3 -m scripts.main
```

## Results
The output directory contains trained models, training loss, training samples, and the test samples of the best model.\
To see implementation details as well as the results, please refer to `Final_Report.pdf`.\
For a description of the inner workings of the code, please refer to `project.ipynb`.\
