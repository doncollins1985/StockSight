# StockSight: The Profit Prophet - Stock Price Prediction Using News Sentiment Analysis

This project implements the **FinBERT-LSTM model** for predicting stock prices. The model combines **financial news sentiment analysis** with **historical stock price data** using deep learning techniques like Long Short-Term Memory (LSTM) networks and the FinBERT NLP model. The project is inspired by the research paper [FinBERT-LSTM: Deep Learning based stock price prediction using News Sentiment Analysis](https://arxiv.org/abs/2211.07392).

---

## 📚 **Features**

- **Stock Price Data**: Collects historical stock price data using the Yahoo Finance API.
- **News Sentiment Analysis**: Uses the FinBERT NLP model to compute sentiment scores for financial news.
- **Deep Learning Model**: Combines stock prices and sentiment data into a FinBERT-LSTM model for accurate predictions.
- **Performance Metrics**: Evaluates model performance using metrics like MAE (Mean Absolute Error) and MAPE (Mean Absolute Percentage Error).

---

## 📂 **Directory Structure**

```plaintext
StockSight/
├── data/
│   ├── price_data.csv                               # Stock price data
│   ├── news_sentiment.csv                           # Sentiment scores from FinBERT
│   ├── merged_price_data_with_sentiment.csv         # Combined stock prices and sentiment data
├── notebooks/
│   ├── data_preprocessing.ipynb                     # Data collection and preprocessing
│   ├── model_training.ipynb                         # Model training and evaluation
├── models/
│   ├── best_model.keras                             # Trained FinBERT-LSTM model
│   ├── checkpoints/                                 # Checkpoints for intermediate training states
├── utils/
│   ├── stock.py                                     # Script to fetch stock prices
│   ├── sentiment.py                                 # Script to compute sentiment scores
│   ├── data.py                                      # Script to merge data
│   ├── train_model.py                               # Script for training the FinBERT-LSTM model
│   ├── evaluate_model.py                            # Script for evaluating and visualizing results
│   ├── predict_future_price.py                      # Script for prediction
├── requirements.txt                                 # List of dependencies
├── README.md                                        # Project documentation
```

---

## 🚀 **How to Run the Project**

### **1. Clone the Repository**

```bash
git clone https://github.com/doncollins1985/StockSight.git
cd StockSight
```

### **2. Install Dependencies**:

Make sure you have Python 3.7+ installed. Install the required packages using:

```bash
pip install -r requirements.txt
```

### **3. Collect Stock Prices**:

Run `stocksight stock` to fetch stock price data.

```bash
python stocksight stock -s START_DATE -t TICKER_SYMBOL
```

### **4. Collect News Data**:

Run `stocksight news` to fetch financial news articles.

```bash
python stocksight news -s START_DATE
```

### **5. Compute Sentiments**:

Run `stocksight data (optional -p for parallel)` to calculate sentiment scores using FinBERT.

```bash
python stocksight data -p
```

### **6. Train the Model**:

Run `stocksight train` to train the FinBERT-LSTM model.

```bash
python stocksight train
```

### **7. Evaluate the Model**:

Run `stocksight evaluate` to evaluate and visualize predictions.

```bash
python stocksight evaluate
```

### **8. Predict Next Day Stock Price**:

Run `stocksight predict (optional -n)` to see next day prediction.

```bash
python stocksight predict -n NUMBER_OF_DAYS
```

---

## 🔧 **Key Components**

1. **FinBERT**: A financial sentiment analysis model based on BERT.
2. **LSTM**: A recurrent neural network architecture for time-series prediction.
3. **Yahoo Finance API**: For fetching historical stock price data.
4. **New York Times API**: For collecting financial news articles.

---

## 📈 **Performance**

The FinBERT-LSTM model achieves:

- **Lower MAE and MAPE** compared to traditional LSTM and MLP models.
- Enhanced accuracy by integrating news sentiment into stock price prediction.

---

## 🛠️ **Technologies Used**

- **Python**: Core programming language.
- **TensorFlow/Keras**: For building and training deep learning models.
- **Transformers (Hugging Face)**: For FinBERT sentiment analysis.
- **Pandas/Numpy**: Data manipulation and preprocessing.
- **Matplotlib**: Visualization.

---

## 📝 **Future Work**

- Expand the dataset with global financial news sources.
- Experiment with additional deep learning architectures (e.g., CNN-LSTM).
- Develop a real-time prediction system for stock trading.

---

## 📜 **References**

- Halder, S. (2022). FinBERT-LSTM: Deep Learning based stock price prediction using News Sentiment Analysis. [arXiv](https://arxiv.org/abs/2211.07392).

---

## 🤝 **Contributions**

Feel free to fork this project and submit pull requests. Contributions are always welcome!

---

## 📧 **Contact**

For any inquiries, please email: doncollins1985@gmail.com
