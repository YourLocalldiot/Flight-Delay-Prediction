# Flight Delay Prediction - Deployment Guide

Hướng dẫn hoàn chỉnh để save, load và deploy machine learning models.

## 📁 File Structure

```
├── app/
│   ├── model_utils.py              # Helper functions for save/load
│   ├── streamlit_app.py            # Streamlit web application
│   └── requirements.txt            # Python dependencies
├── models/
│   ├── linear_regression_arrdelay.joblib
│   ├── random_forest_arrdelay.joblib
│   └── ...
├── notebooks/
│   ├── 04_linear_regression.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_model_comparison.ipynb
│   └── 07_model_deployment.ipynb   # Deployment workflow
└── data/
    └── processed_flight_data.csv
```

## 🔧 Setup

### 1. Install Dependencies

```bash
cd app/
pip install -r requirements.txt
```

### 2. Train Models

Chạy các notebook để huấn luyện models:

```bash
# Linear Regression
jupyter notebook notebooks/04_linear_regression.ipynb

# Random Forest
jupyter notebook notebooks/05_random_forest.ipynb
```

Notebooks sẽ tự động save models vào `models/` folder bằng joblib.

### 3. Save Trained Models (Optional)

Nếu models chưa được save, bạn có thể save bằng script Python:

```python
from pathlib import Path
import sys
sys.path.append('app')
from model_utils import save_model

# Save Linear Regression model
save_model(
    model=lr_model,
    feature_names=['DepDelay', 'CRSElapsedTime', 'Distance', 'DepHour', 
                   'IsWeekend', 'IsRushHour', 'Origin_Freq', 'Dest_Freq', 
                   'Month', 'DayOfWeek'],
    scaler=scaler,
    model_name="linear_regression_arrdelay",
    model_dir=Path("models")
)

# Save Random Forest model (không cần scaler)
save_model(
    model=rf_model,
    feature_names=['DepDelay', 'CRSElapsedTime', 'Distance', 'DepHour', 
                   'IsWeekend', 'IsRushHour', 'Origin_Freq', 'Dest_Freq', 
                   'Month', 'DayOfWeek'],
    scaler=None,
    model_name="random_forest_arrdelay",
    model_dir=Path("models")
)
```

## 🚀 Run Streamlit App

```bash
cd app/
streamlit run streamlit_app.py
```

App sẽ mở ở http://localhost:8501

### Features

- ✈️ **Model Selection**: Chọn giữa Linear Regression và Random Forest
- 🎚️ **Interactive Input**: Nhập flight information bằng sliders
- 📊 **Real-time Prediction**: Nhận dự đoán delay ngay lập tức
- ℹ️ **Model Info**: Xem chi tiết model được sử dụng

## 💾 Serialization Workflow

### Saving a Model

```python
from model_utils import save_model

# Package model với scaler và metadata
model_path = save_model(
    model=trained_model,
    feature_names=feature_list,
    scaler=scaler_object,
    model_name="my_model",
    model_dir=Path("../models")
)
```

**Output**: `models/my_model.joblib` (compressed)

### Loading a Model

```python
from model_utils import load_model

# Load model package
model_package = load_model(Path("../models/my_model.joblib"))

# Extract components
model = model_package['model']
features = model_package['feature_names']
scaler = model_package['scaler']
model_type = model_package['model_type']
```

### Making Predictions

```python
from model_utils import predict_with_preprocessing

# Simple way
input_data = {
    'DepDelay': 10,
    'CRSElapsedTime': 120,
    'Distance': 800,
    'DepHour': 12,
    'IsWeekend': 0,
    'IsRushHour': 0,
    'Origin_Freq': 0.01,
    'Dest_Freq': 0.01,
    'Month': 6,
    'DayOfWeek': 3
}

prediction = predict_with_preprocessing(
    model_package=model_package,
    input_data=input_data,
    feature_names=model_package['feature_names']
)

print(f"Predicted Arrival Delay: {prediction:.1f} minutes")
```

## 🔑 Key Concepts

### Serialization

Quá trình lưu Python objects thành file để tái sử dụng:

- ✅ Lưu state của model
- ✅ Không cần retrain
- ✅ Share qua mạng/servers
- ✅ Reproducibility

### Joblib vs Pickle

| Feature | Joblib | Pickle |
|---------|--------|--------|
| Compression | ✅ Yes | ❌ No |
| Speed | ✅ Fast | ❌ Slow |
| File Size | ✅ Small | ❌ Large |
| Sklearn | ✅ Optimal | ❌ Slow |

**Kết luận**: Joblib tốt hơn cho sklearn models.

## 📋 Production Workflow

```
Training Phase (Notebook)
  ↓
1. Load & preprocess data
2. Train model
3. Evaluate performance
4. Save model (joblib)
  ↓
Deployment Phase (Streamlit/API)
  ↓
1. Load model from joblib
2. Receive user input
3. Preprocess (scale if needed)
4. Make prediction
5. Return result
```

## 🐛 Troubleshooting

### Model not found error

```
FileNotFoundError: Model file not found: models/linear_regression_arrdelay.joblib
```

**Solution**: Run notebooks to train and save models first.

### Version compatibility error

```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Scaler mismatch

Nếu Linear Regression prediction không đúng, kiểm tra:
1. Scaler được save và load đúng
2. Input features được scale bằng cùng scaler
3. Feature order đúng

## 📊 Model Comparison

| Aspect | Linear Regression | Random Forest |
|--------|------------------|---------------|
| Training Time | ⚡ Fast | 🐌 Slow |
| Prediction Time | ⚡ Fast | ⚡ Fast |
| Interpretability | 📈 Excellent | 📉 Low |
| Accuracy | 📊 Good | 📊 Better |
| Non-linear Fit | ❌ No | ✅ Yes |
| Hyperparameter | Few | Many |

## 🎯 Next Steps

1. Collect more training data
2. Engineer new features
3. Try advanced models (XGBoost, LightGBM)
4. Implement model versioning
5. Monitor prediction drift in production
6. Set up retraining pipeline

## 📝 References

- [Joblib Documentation](https://joblib.readthedocs.io/)
- [Scikit-learn Model Persistence](https://scikit-learn.org/stable/modules/model_persistence.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 📧 Support

Có thắc mắc? Tham khảo `notebooks/07_model_deployment.ipynb` để hiểu chi tiết.

---

**Tạo bởi**: Flight Delay Prediction Project  
**Ngày**: May 24, 2026
