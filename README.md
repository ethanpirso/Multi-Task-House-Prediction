# Multi-Task House Prediction

## Project Objective
The goal of this project is to build a multi-task learning model using PyTorch Lightning that predicts both house prices (a regression task) and house categories (a classification task). This involves using a shared bottom neural network architecture with task-specific top layers, applying advanced machine learning techniques and model management features provided by PyTorch Lightning.

## Dataset
The project utilizes the "House Prices - Advanced Regression Techniques" dataset from kaggle, which can be downloaded here https://www.kaggle.com/c/house-prices-advanced-regression-techniques. A new variable 'House Category' is created from features such as 'House Style', 'Bldg Type', 'Year Built', and 'Year Remod/Add', to classify houses into distinct categories alongside predicting their prices.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip for managing Python packages

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-task-house-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd multi-task-house-prediction
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
Run the data preprocessing script to prepare your data:
```bash
python preprocessing/data_preprocessor.py
```

### Training the Model
To train the model, execute:
```bash
python training/train_model.py
```

### Evaluating the Model
To evaluate the model on the test dataset, use:
```bash
python evaluation/evaluate_model.py
```

## Project Structure
- `data/`: Contains raw and processed data.
- `models/`: Includes the multi-task learning model definition and .pkl file.
- `preprocessing/`: Scripts for data cleaning and preparation.
- `training/`: Training and validation logic.
- `evaluation/`: Model evaluation scripts.
- `utils/`: Utilities like logging and other common tasks.
- `notebooks/`: Jupyter notebooks for analysis and experimentation.
- `reports/`: Data profile and project reports.
- `lightning_logs/`: Contains MLFlow runs and model checkpoints.
- `requirements.txt`: Project dependencies.
- `README.md`: Documentation of the project.

## Advanced Features
This project makes extensive use of PyTorch Lightning's advanced features such as:
- **Logging**: Integration with MLFlow for monitoring training progress and performance.
- **Callbacks**: Use of the ModelCheckpoint callback to save the best model during training.
- **Trainer API**: Utilizes the Trainer class from PyTorch Lightning for efficient model training and testing.

## Hyperparameter Tuning
Hyperparameter optimization is performed using PyTorch Lightning's integration with Optuna:
```bash
python training/tune_hyperparameters.py
```

## Results

After training and evaluating our multi-task learning model, we achieved promising results on our test dataset. The model's performance is summarized as follows:

- **Accuracy:** The model achieved an accuracy of approximately 83.90% in categorizing houses into one of the 8 categories. This high level of accuracy indicates that the model is quite effective in understanding and classifying the complex patterns associated with the diverse house categories.

- **RMSE:** The Root Mean Square Error (RMSE) for the house price prediction task was around 60,196.13. While there's room for improvement, this value suggests that the model has a decent grasp on predicting house prices, considering the variability and range of house prices in the dataset.

These results demonstrate the capability of our multi-task learning approach in handling the intricacies of predicting multiple outputs simultaneously. Future work will focus on further optimizing the model to improve both accuracy and RMSE, potentially through more advanced feature engineering, model architecture adjustments, and hyperparameter tuning.

## View Logged Runs in the MLFlow UI

From the `lightning_logs/` directory:

```bash
python -m mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
```

Navigate to http://localhost:8080 in your browser to view the results.

You can also click and run the `start_mlflow.bat` script on Windows to not only start the MLflow UI but also automatically open the default web browser.

## Authors
- Ethan Pirso

## Acknowledgments
- PyTorch Lightning team for providing an excellent framework for rapid prototyping of deep learning models.
