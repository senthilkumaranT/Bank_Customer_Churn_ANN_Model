# Customer Churn Prediction (ANN)

This project uses an Artificial Neural Network (ANN) to predict customer churn based on various features.

## Project Structure

The project is organized into the following directories:

- **`data/`**: Contains the dataset (`Churn_Modelling.csv`).
- **`models/`**: Stores the trained model (`model.h5`) and preprocessing objects (scalers, encoders).
- **`notebooks/`**: Contains Jupyter notebooks for exploration, training, and prediction.
  - `ANN.ipynb`: Main notebook for building and training the ANN.
  - `prediction.ipynb`: Notebook for making predictions using the trained model.
  - `experiments.ipynb`: Scratchpad for experiments.
- **`src/`**: Source code for the application.
  - `app.py`: Streamlit application for interacting with the model.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   To run the Streamlit app, execute the following command from the root directory:
   ```bash
   streamlit run src/app.py
   ```

## Model Training

To retrain the model, run the `notebooks/ANN.ipynb` notebook. Ensure that the generated model and pick files are saved to the `models/` directory.

## Usage

Open the Streamlit app in your browser, adjust the input parameters (Geography, Gender, Age, etc.), and view the churn probability prediction.
