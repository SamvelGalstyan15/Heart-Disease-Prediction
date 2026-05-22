# Heart Disease Prediction

A binary classification project designed to predict the presence of heart disease in patients using tabular data from the Kaggle Playground Series competition.

## 📊 Project Overview
* **Dataset Size**: 630,000 rows and 14 original features.
* **Target Variable**: `Heart Disease` (`1` for Presence, `0` for Absence).
* **Feature Engineering**: Expanded the feature space from 13 input features to 104 using `PolynomialFeatures` to capture non-linear relationships.

## 🏗️ Neural Network Architecture
The model is implemented as a fully connected deep neural network using the Keras Sequential API:
* **Input Layer**: Accepts 104 engineered features.
* **Hidden Layer 1**: 128 neurons, `ReLU` activation + `Dropout(0.3)`.
* **Hidden Layer 2**: 128 neurons, `ReLU` activation + `Dropout(0.3)`.
* **Output Layer**: 1 neuron, `Sigmoid` activation (outputs probability).

### Training Configurations
* **Optimizer**: Adam (learning rate = 0.00005)
* **Loss Function**: Binary Crossentropy
* **Batch Size**: 1024
* **Epochs**: 10 (with a 20% validation split)

## ⚙️ Installation

To set up the environment and install all dependencies, run the following commands:

```text
Create virtual environment
python -m venv venv

Activate virtual environment
On Windows:
venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate

Install dependencies
pip install -r requirements.txt
```

## 📈 Results

The learning curves demonstrate steady convergence with no major signs of overfitting, as training and validation loss drop simultaneously:

```python
plt.plot(model.history.history['loss'], label='loss')
plt.plot(model.history.history['val_loss'], label='val_loss')
```

*<img width="635" height="465" alt="image" src="https://github.com/user-attachments/assets/e237cb92-6c2b-4a89-9a54-32de7590e5de" />
*
* **Validation Accuracy**: ~88.4%
* **Final Test Evaluation Metrics**:
  * **Loss**: 0.2813
  * **Accuracy**: 0.8818 (88.18%)
