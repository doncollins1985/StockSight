# StockSight: The Profit Prophet - Stock Price Prediction Using News Sentiment & CTM

**StockSight** is an advanced stock price prediction pipeline that integrates **financial news sentiment analysis** with a **Continuous Thought Machine (CTM)** neural network architecture. By combining historical price data with sentiment scores derived from the **FinBERT** NLP model, StockSight aims to capture both market trends and public sentiment.

---

## 📚 **Features**

- **Stock Price Data**: Fetches historical data using the **Yahoo Finance** API and calculates technical indicators (SMA, EMA, RSI, MACD, etc.) using `talib`.
- **News Sentiment Analysis**: Uses the **FinBERT** model (via Hugging Face Transformers) to classify financial news headlines as Positive, Negative, or Neutral.
- **Advanced Architecture**: Implements a **Continuous Thought Machine (CTM)**, a dynamic neural architecture inspired by cognitive modeling, to process time-series data.
- **Robust Prediction**: Predicts **Log Returns** to ensure stationarity and reconstructs price levels for realistic future forecasts.
- **Data Scaling**: Utilizes **StandardScaler** to handle market volatility and data distribution effectively.

---

## 📂 **Directory Structure**

```plaintext
StockSight/
├── data/
│   ├── price_data.csv                    # Historical stock price data
│   ├── news_data.csv                     # Raw news headlines
│   ├── news_sentiments.csv               # Sentiment scores from FinBERT
│   ├── merged_price_and_sentiment.csv    # Merged dataset with technical indicators
│   ├── sequences.npz                     # Processed numpy sequences for training
│   └── scalers/                          # Saved StandardScaler objects
├── models/
│   ├── final_model.pth                   # Trained PyTorch model
│   └── checkpoints/                      # Training checkpoints
├── stocksight/
|   ├── cli.py                            # CLI entry point script
│   ├── stock.py                          # Fetches stock prices & calculates indicators
│   ├── sentiments.py                     # Computes sentiment scores using FinBERT
│   ├── data.py                           # Merges data and creates sequences
│   ├── models.py                         # Defines the CTM architecture
│   ├── train_model.py                    # Trains the CTM model
│   ├── evaluate_model.py                 # Evaluates performance & plots results
│   ├── predict_future_price.py           # Generates future price predictions
│   └── utils.py                          # Helper functions (logging, config, etc.)
├── config.json                           # Configuration parameters
├── requirements.txt                      # Python dependencies
├── pyproject.toml                        # Python installation file
└── README.md                             # Project documentation
```

---

## 🚀 **How to Run the Project**

### **1. Clone the Repository**

```bash
git clone https://github.com/doncollins1985/StockSight.git
cd StockSight
```

### **2. Install Dependencies**

Ensure you have Python 3.8+ installed. Install the required packages:

```bash
pip install -r requirements.txt
```

### **3. Configuration**

Edit `config.json` to adjust parameters like `ticker`, `start_date`, `model` hyperparameters, or `target_column`.

### **4. Pipeline Execution**

StockSight uses a convenient CLI tool `stocksight` to manage the pipeline.

**Step A: Fetch Stock Data**
```bash
python stocksight stock -t ^GSPC -s 1985-01-01
```

**Step B: Fetch News Data**
```bash
# Fetches news from NYT API (Requires API Key in config.json)
python stocksight news -s 2024-01-01
```

**Step C: Process Data**
Computes sentiment (if needed), merges datasets, and creates training sequences.
```bash
python stocksight data
```
*Optional: Use `-p` for parallel sentiment processing.*

**Step D: Train the Model**
Trains the CTM model on the processed data.
```bash
python stocksight train
```

**Step E: Evaluate**
Evaluates the model on test data, calculating metrics (MAE, R²) and generating plots.
```bash
python stocksight evaluate
```

**Step F: Predict**
Predicts stock prices for the next `N` days.
```bash
python stocksight predict -n 5
```

---

## 🔧 **Key Components**

1.  **FinBERT**: A pre-trained NLP model specifically optimized for financial sentiment analysis.
2.  **Continuous Thought Machine (CTM)**: A custom PyTorch model that dynamically processes temporal information using "Neural Language Models" (NLMs) and "Synapses" to maintain a hidden state over time.
3.  **Log Returns**: The model predicts the logarithmic return of the asset, which is then converted back to a price price for actionable insights.

---

## 📈 **Performance**

The system evaluates performance using:
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

Evaluation results are saved in `logs/evaluation/` as JSON metrics and visualization plots.

---

## 🛠️ **Technologies Used**

- **Python**: Core language.
- **PyTorch**: Deep learning framework for the CTM model.
- **Transformers (Hugging Face)**: FinBERT model.
- **TA-Lib**: Technical analysis library.
- **Pandas/NumPy**: Data manipulation.
- **Matplotlib**: Visualization.

---

## 📝 **Future Work**

- Incorporate additional news sources beyond NYT.
- Implement Reinforcement Learning for automated trading execution.
- Add support for intraday data and real-time streaming.

---

## 📜 **References**

- **FinBERT**: [ArXiv:1908.10063](https://arxiv.org/abs/1908.10063)
- **Continuous Thought Machine**: Inspired by cognitive architectures for sequence processing.

---

## 📧 **Contact**

For inquiries, please contact the repository owner.
