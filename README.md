# 🤖 AutoML Classification System

A **Streamlit-based AutoML web application** that automates the complete machine learning workflow for classification tasks - from data upload to model deployment and real-time prediction.

## ✨ Features

- 📁 **CSV Upload** with automatic encoding detection (UTF-8, Latin-1)
- 🔍 **Automated EDA** with missing value analysis, outlier detection, correlation matrices
- ⚠️ **Data Quality Detection** (class imbalance, high cardinality, constant features)
- 🛠️ **User-Controlled Preprocessing** - you approve all fixes before they're applied
- 🤖 **7 Classification Algorithms**: Logistic Regression, KNN, Decision Tree, Naive Bayes, Random Forest, SVM, Rule-Based
- ⚡ **Hyperparameter Tuning** with Grid Search or Randomized Search
- 📊 **Comprehensive Comparison** with Accuracy, Precision, Recall, F1, ROC-AUC
- 📄 **Detailed Reports** in HTML and Markdown format
- 🎯 **Real-Time Prediction** with the best trained model

## 🚀 Quick Start

### Local Development

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd AutoMl-FutureCashCow
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open browser** at `http://localhost:8501`

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with default settings (auto-detects `streamlit_app.py`)

## 📁 Project Structure

```
├── streamlit_app.py              # Main entry point
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── pages/
│   ├── 01_Upload_And_Info.py     # Dataset upload & metadata
│   ├── 02_EDA_And_Issues.py      # EDA & issue detection
│   ├── 03_Preprocess_And_Split.py # Preprocessing configuration
│   ├── 04_Train_And_Tune.py      # Model training
│   ├── 05_Compare_Report_Deploy.py # Comparison & reporting
│   └── 06_Prediction.py          # Real-time prediction
├── ml/
│   ├── data_loader.py            # CSV parsing & validation
│   ├── eda.py                    # Exploratory data analysis
│   ├── issue_detector.py         # Data quality detection
│   ├── preprocessing.py          # Data preprocessing
│   ├── models.py                 # Classification models
│   ├── tuning.py                 # Hyperparameter optimization
│   ├── evaluation.py             # Model evaluation
│   └── report_generator.py       # Report generation
└── utils/
    ├── session_manager.py        # Session state helpers
    ├── validators.py             # Input validation
    └── visualizations.py         # Plot utilities
```

## 🔧 Workflow

1. **Upload Dataset** - Upload CSV file (max 200 MB)
2. **Explore Data** - View automated EDA and detected issues
3. **Approve Fixes** - Review and approve data quality fixes
4. **Configure Preprocessing** - Set scaling, encoding, train/test split
5. **Train Models** - Train 7 classifiers with hyperparameter tuning
6. **Compare Results** - View metrics, confusion matrices, ROC curves
7. **Generate Report** - Download comprehensive HTML/Markdown report
8. **Make Predictions** - Use the best model for new data

## 📊 Supported Algorithms

| Algorithm              | Description                             |
| ---------------------- | --------------------------------------- |
| Logistic Regression    | Linear classifier for binary/multiclass |
| K-Nearest Neighbors    | Distance-based classification           |
| Decision Tree          | Rule-based tree classifier              |
| Naive Bayes            | Probabilistic classifier (Gaussian)     |
| Random Forest          | Ensemble of decision trees              |
| Support Vector Machine | Margin-based classifier                 |
| Rule-Based             | Interpretable decision rules            |

## 📈 Evaluation Metrics

- **Accuracy** - Overall correctness
- **Precision** - Positive predictive value
- **Recall** - Sensitivity / True positive rate
- **F1-Score** - Harmonic mean of precision and recall (primary ranking metric)
- **ROC-AUC** - Area under ROC curve (binary classification only)
- **Confusion Matrix** - Detailed prediction breakdown

## ⚙️ Configuration

### File Constraints

- **Format**: CSV only
- **Max Size**: 200 MB
- **Encoding**: Auto-detected (UTF-8, Latin-1)

### Performance Targets

- **EDA**: ≤30 seconds for 100k rows
- **Training**: ≤15 minutes for 50k rows
- **Prediction**: ≤2 seconds response time

## 🎓 Target Users

- **Students** - Learning ML through hands-on experimentation
- **Researchers** - Quick model prototyping and comparison
- **Data Analysts** - Building classification models without coding
- **ML Practitioners** - Rapid baseline model evaluation

## 📝 License

This project is developed as part of CS-245 Machine Learning course.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

**Built with ❤️ using Streamlit and scikit-learn**
