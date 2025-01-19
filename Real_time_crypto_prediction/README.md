# Trading Price Prediction

This project predicts future trading prices based on historical trade data using machine learning. The process involves fetching real-time trade data, processing it into meaningful features, training a predictive model, and using that model to make predictions on future prices.

## Project Overview

The project is structured to facilitate data fetching, feature extraction, model training, and prediction generation. The key components include:

- **Data Fetching**: The project integrates with the Kraken API to fetch the latest trade data. Users must configure API access and collect data to train the model.
- **Feature Processing**: The fetched data is processed into features such as percentage price change, 5-trade Simple Moving Average (SMA), 10-trade SMA, and sum of volumes from the last 5 trades. These features help the model to learn price trends.
- **Model Training**: Once the data is collected, a machine learning model is trained to predict the next trading price based on historical trade data. Users are required to run the training script to train the model on their collected data.
- **Prediction**: After training the model, it can predict the next price based on the incoming trade data, which is continuously retrieved from the API.

The system is designed for real-time predictions, meaning the model updates its predictions continuously as new trade data becomes available.

## Requirements

To run the project, you'll need the following dependencies. You can install them using the provided `requirements.txt`:

- Data processing libraries
- Machine learning frameworks
- API integration tools

## Configuration

Before running the project, you need to configure the necessary parameters in the `config.py` file. This includes setting up your Kraken API keys and adjusting model-specific settings.

### Notes:
- The project does not include pre-trained models or API keys. You need to collect data and train the model yourself.
- API key and data collection configuration are handled in `config.py`.

## Usage

The project is designed to automatically fetch trade data, process it, and predict the next trading price once the model is trained. However, users must first collect data through the API and train the model by running the training script.

You can run the following individual scripts:

- `fetch_trades.py`: Fetches the trade data from Kraken API.
- `process_features.py`: Processes the fetched data into features used for training.
- `train_model.py`: Trains a machine learning model on the processed data.
- `check_data.py`: Optionally used to verify the fetched data before training.

Once the model is trained, it can predict the next price based on real-time incoming data.
