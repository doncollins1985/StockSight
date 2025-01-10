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
FinBERT_LSTM/
├── data/
│   ├── stock_prices.csv                       # Stock price data
│   ├── news_sentiment.csv               # Sentiment scores from FinBERT
│   ├── merged_data.csv                     # Combined stock prices and sentiment data
├── notebooks/
│   ├── data_preprocessing.ipynb     # Data collection and preprocessing
│   ├── model_training.ipynb             # Model training and evaluation
├── models/
│   ├── finbert_lstm_model.keras     # Trained FinBERT-LSTM model
│   ├── checkpoints/                            # Checkpoints for intermediate training states
├── scripts/
│   ├── stock.py                                    # Script to fetch stock prices
│   ├── news.py                                     # Script to fetch financial news
│   ├── sentiment.py                            # Script to compute sentiment scores
│   ├── data.py                                      # Script to merge data
│   ├── train.py                                     # Script for training the FinBERT-LSTM model
│   ├── evaluate.py                              # Script for evaluating and visualizing results
│   ├── predict.py                                # Script for prediction
├── requirements.txt                         # List of dependencies
├── README.md                                 # Project documentation
```

---

## 🚀 **How to Run the Project**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/StockSight.git
cd StockSight
```

### **2. Install Dependencies**:
Make sure you have Python 3.7+ installed. Install the required packages using:
```bash
pip install -r requirements.txt
```

### **3. Collect Stock Prices**:
Run `stock.py` to fetch stock price data.
```bash
python scripts/stock.py
```

### **4. Collect News Data**:
Run `news.py` to fetch financial news articles.
```bash
python scripts/news.py
```

### **5. Compute Sentiments**:
Run `sentiment.py` to calculate sentiment scores using FinBERT.
```bash
python scripts/sentiments.py
```

### **6. Prepare Data**:
Run `data.py` to create rolling window sequences for LSTM training.
```bash
python scripts/data.py
```

### **7. Train the Model**:
Run `train.py` to train the FinBERT-LSTM model.
```bash
python scripts/train.py
```

### **9. Evaluate the Model**:
Run `evaluate.py` to evaluate and visualize predictions.
```bash
python scripts/evaluate.py
```

### **10. Predict Next Day Stock Price**:
Run `predict.py` to see next day prediction.
```bash
python scripts/predict.py
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
