# ABC-Notated Melody Generation using RNNs

## Background
This project is a melody generation system that uses PyTorch RNN architectures to generate melodies in ABC notation.\
The dataset used for this project is 6 years old and was downloaded from Kaggle user Shivam Raj:
https://www.kaggle.com/datasets/raj5287/abc-notation-of-tunes
The dataset is a text file containing 1850 melodies in a similar format to the following example:
```
X: 1
T: The Enchanted Valley
M: 2/4
L: 1/16
B: "O'Neill's 1"
N: "Very slow" "collected by J. O'Neill"
N:
Z: "Transcribed by Norbert Paap, norbertp@bdu.uva.nl"
Z:
K:Gm
G3-A (Bcd=e) | f4 (g2dB) | ({d}c3-B) G2-E2 | F4 (D2=E^F) |
G3-A (Bcd=e) | f4 d2-f2 | (g2a2 b2).g2 | {b}(a2g2 f2).d2 |
(d2{ed}c2) B2B2 | (A2G2 {AG}F2).D2 | (GABc) (d2{ed}c>A) | G2G2 G2z ||
G | B2c2 (dcAB) | G2G2 G3G | B2d2 (gfdc) | d2g2 (g3ga) |
(bagf) (gd)d>c | (B2AG) F-D.D2 | (GABc) d2d2 | (bgfd) cA.F2 |
G2A2 (B2{cB}AG) | A3-G F2-D2 | (GABc) (d2{ed}c>A) | G2G2 G2z2 ||
```
The data is preprocessed before performing feature extraction, tokenization, data augmentation, and training.

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
The sampled melodies have high key-adherence but poor melodic structure, with the biggest improvement in performance
resulting in the implementation of data augmentation.
```
Example Output (Truncated):
T:6/8 L:1/8 K:Am | ~d B A2 B | c A G A2 || G || A B c d2 e | a g e e d c | c B A G F E | 3EGB g f e d | ...
```
Example Output
To see implementation details as well as the performance metrics, please refer to `Final_Report.pdf`.\
For a description of the inner workings of the code, please refer to `project.ipynb`.
