# 📦 E-commerce Time-Series Forecast — Revenue & COGS

Dự báo **Doanh thu (Revenue)** và **Chi phí hàng bán (COGS)** theo ngày trong lĩnh vực E-commerce, sử dụng mô hình **XGBoost** kết hợp với chiến lược **Recursive Forecasting** và tối ưu siêu tham số bằng **Optuna**.

---

## 📁 Cấu trúc thư mục

```
├── train_model.ipynb          # Notebook huấn luyện chính (toàn bộ pipeline)
├── sales.csv                 # Dữ liệu đầu vào (bắt buộc)
└── submission.csv             # Kết quả dự báo đầu ra (sinh sau khi chạy)
```

> **Lưu ý:** Toàn bộ pipeline nằm trong một file notebook duy nhất `train_model.ipynb`. Không có file script phụ hay module ngoài.

---

## 📊 Dữ liệu đầu vào

| Thuộc tính | Chi tiết |
|---|---|
| File | `sales.csv` |
| Cột bắt buộc | `date`, `Revenue`, `COGS` |
| Định dạng cột date | `YYYY-MM-DD` |
| Tập huấn luyện | 04/07/2012 → 31/12/2022 |
| Tần suất | Hàng ngày (daily) |

---

## 📤 Kết quả đầu ra

| Thuộc tính | Chi tiết |
|---|---|
| File | `submission.csv` |
| Số cột | 3: `date`, `revenue`, `cogs` |
| Khoảng dự báo | 01/01/2023 → 01/07/2024 |
| Số dòng | ~548 ngày |

**Ví dụ:**
```
date,revenue,cogs
2023-01-01,12345678.00,8765432.00
2023-01-02,11234567.00,7654321.00
...
```

---

## ⚙️ Pipeline Huấn Luyện

Pipeline gồm **12 bước** thực thi tuần tự trong notebook:

### Bước 1 — Cố định Random Seed
Đảm bảo kết quả tái lặp hoàn toàn trên mọi máy và mọi lần chạy.
```
SEED = 42
PYTHONHASHSEED, random, numpy, XGBoost random_state, Optuna TPESampler
OMP_NUM_THREADS = 1  →  tắt parallel non-determinism của XGBoost
```

### Bước 2 — Import & Cấu hình
Khai báo toàn bộ thư viện và các hằng số cấu hình:

| Tham số | Giá trị |
|---|---|
| `TRAIN_START` | 2012-07-04 |
| `TRAIN_END` | 2022-12-31 |
| `FORECAST_START` | 2023-01-01 |
| `FORECAST_END` | 2024-07-01 |
| `OPTIMAL_LAGS` | [1, 3, 7, 29, 30, 365] |
| `OPTIMAL_WINDOWS` | [3, 7] |

### Bước 3 — Tải dữ liệu
Đọc `sales.csv`, set `date` làm index, tách riêng 2 series:
- `train_series` — Revenue từ 2012 đến 2022
- `train_cogs` — COGS từ 2012 đến 2022

### Bước 4 — Xây dựng Feature Matrix

Hàm `make_calendar_features()` tạo **24 features** từ DatetimeIndex, chia làm 4 nhóm:

**Calendar features (14 features)**
| Feature | Mô tả |
|---|---|
| `dayofweek` | Thứ trong tuần (0=T2, 6=CN) |
| `quarter` | Quý (1–4) |
| `month` | Tháng (1–12) |
| `year` | Năm |
| `dayofyear` | Ngày thứ mấy trong năm (1–366) |
| `dayofmonth` | Ngày trong tháng (1–31) |
| `weekofyear` | Tuần ISO trong năm |
| `is_weekend` | 1 nếu là T7/CN |
| `is_month_start` | 1 nếu là ngày đầu tháng |
| `is_month_end` | 1 nếu là ngày cuối tháng |
| `is_quarter_start` | 1 nếu là ngày đầu quý |
| `is_quarter_end` | 1 nếu là ngày cuối quý |
| `is_year_start` | 1 nếu là ngày đầu năm |
| `is_year_end` | 1 nếu là ngày cuối năm |

**Cyclical features (4 features)**  
Mã hóa vòng tròn để mô hình nhận biết tính tuần hoàn của thời gian:

| Feature | Công thức |
|---|---|
| `sin_dayofweek` | sin(2π × dayofweek / 7) |
| `cos_dayofweek` | cos(2π × dayofweek / 7) |
| `sin_month` | sin(2π × month / 12) |
| `cos_month` | cos(2π × month / 12) |

**Holiday features — Việt Nam (4 features)**

| Feature | Mô tả |
|---|---|
| `is_holiday` | 1 nếu là ngày lễ quốc gia Việt Nam (thư viện `holidays.VN`) |
| `is_tet_season` | 1 nếu ngày lễ đó là Tết Nguyên Đán |
| `days_to_tet` | Số ngày còn lại đến Tết gần nhất (âm nếu đã qua) |
| `is_pre_tet_3d` | 1 nếu trong 3 ngày trước Tết |

> Ngày Tết Âm lịch được hard-code chính xác cho từng năm 2012–2024.

**Lag features (6 features mỗi target)**  
Lấy giá trị thực tế của `t - lag` ngày trước, tránh data leakage bằng cách chỉ dùng quá khứ:

| Feature | Ý nghĩa |
|---|---|
| `revenue_lag_1` | Doanh thu hôm qua |
| `revenue_lag_3` | 3 ngày trước |
| `revenue_lag_7` | 1 tuần trước |
| `revenue_lag_29` | ~1 tháng trước |
| `revenue_lag_30` | 1 tháng trước |
| `revenue_lag_365` | Cùng ngày năm ngoái (seasonal) |

> COGS có bộ lag tương tự: `cogs_lag_1`, `cogs_lag_3`, ..., `cogs_lag_365`

**Rolling window features (8 features mỗi target)**  
Tính trên cửa sổ 3 ngày và 7 ngày, luôn `shift(1)` để tránh leakage:

| Feature | Mô tả |
|---|---|
| `revenue_roll_mean_3` | Trung bình 3 ngày gần nhất |
| `revenue_roll_std_3` | Độ lệch chuẩn 3 ngày |
| `revenue_roll_max_3` | Max 3 ngày |
| `revenue_roll_min_3` | Min 3 ngày |
| `revenue_roll_mean_7` | Trung bình 7 ngày gần nhất |
| `revenue_roll_std_7` | Độ lệch chuẩn 7 ngày |
| `revenue_roll_max_7` | Max 7 ngày |
| `revenue_roll_min_7` | Min 7 ngày |

> **Tổng số features mỗi model: 24 (calendar) + 6 (lag) + 8 (rolling) = 38 features**

### Bước 5 & 6 — Hyperparameter Tuning (Optuna)

Tối ưu bằng **Optuna TPE Sampler** với 30 trials, validation bằng **TimeSeriesSplit 3 folds** (không dùng KFold thông thường để tôn trọng thứ tự thời gian). Metric tối ưu hóa: **RMSE**.

Không gian tìm kiếm:

| Hyperparameter | Khoảng tìm kiếm |
|---|---|
| `learning_rate` | [1e-4, 0.1] (log scale) |
| `max_depth` | [3, 10] |
| `min_child_weight` | [1, 10] |
| `subsample` | [0.5, 1.0] |
| `colsample_bytree` | [0.5, 1.0] |
| `gamma` | [1e-8, 1.0] (log scale) |
| `reg_lambda` | [1e-8, 10.0] (log scale) |
| `reg_alpha` | [1e-8, 10.0] (log scale) |
| `n_estimators` | 1000 (với early stopping 50 rounds) |

> `n_estimators` cuối cùng = trung bình `best_iteration` qua 3 folds.  
> Quá trình tuning được thực hiện **độc lập** cho Revenue và COGS.

### Bước 7 & 8 — Huấn luyện Final Model

Sau khi tìm được best params, train lại trên **toàn bộ tập train (2012–2022)** với `random_state=42`.

Đánh giá in-sample (sanity check) bằng 4 chỉ số:

| Chỉ số | Mô tả |
|---|---|
| **MAE** | Mean Absolute Error — sai số tuyệt đối trung bình |
| **RMSE** | Root Mean Squared Error — phạt nặng outlier |
| **MAPE** | Mean Absolute Percentage Error — sai số tương đối (%) |
| **R²** | Hệ số xác định — 1.0 = hoàn hảo |

### Bước 9 & 10 — Recursive Forecasting

Chiến lược **Recursive (one-step-ahead)**:
- Khởi tạo `history` = toàn bộ dữ liệu train thực tế
- Mỗi ngày forecast: tính lag/rolling từ `history` → predict → **nạp kết quả vừa dự báo vào `history`** → tiếp tục ngày kế tiếp
- Đảm bảo **không có data leakage** vì lag luôn dùng quá khứ
- Kết quả được clamp về 0 (không cho ra giá trị âm)

Revenue và COGS được forecast **độc lập song song** theo cùng chiến lược.

### Bước 11 — Lưu kết quả

Gộp Revenue + COGS vào **một file duy nhất** `submission.csv` với 3 cột: `date`, `revenue`, `cogs`.

### Bước 12 — Visualization

6 biểu đồ được tạo ra:
1. **Revenue**: Thực tế (2021–2022) + Dự báo (2023–2024) — line chart
2. **Revenue**: Dự báo theo tháng — bar chart + MA-30 overlay
3. **COGS**: Thực tế (2021–2022) + Dự báo (2023–2024) — line chart
4. **COGS**: Dự báo theo tháng — bar chart + MA-30 overlay
5. **Revenue vs COGS**: So sánh theo tháng + vùng Gross Profit
6. **SHAP + Feature Importance**: (xem bên dưới)

### Bước 13 & 14 — SHAP / Feature Importance

Giải thích mô hình bằng `shap.TreeExplainer` (nhanh, chính xác với tree-based models).  
Mỗi model (Revenue & COGS) sinh ra:

| Biểu đồ | Mô tả |
|---|---|
| **SHAP Summary (Beeswarm)** | Phân phối tác động của từng feature lên từng dự báo |
| **SHAP Bar (mean \|SHAP\|)** | Ranking feature importance tổng hợp |
| **SHAP Dependence Plots** | Mối quan hệ giữa giá trị feature và SHAP value (Top 5) |
| **XGBoost Gain Importance** | Native importance theo gain — Revenue vs COGS cạnh nhau |

---

## 🔧 Thư viện sử dụng

| Thư viện | Mục đích |
|---|---|
| `xgboost` | Mô hình gradient boosting chính |
| `optuna` | Tối ưu siêu tham số tự động (TPE Sampler) |
| `shap` | Giải thích mô hình (TreeExplainer) |
| `pandas` | Xử lý dữ liệu time-series |
| `numpy` | Tính toán số học |
| `scikit-learn` | TimeSeriesSplit, metrics (MAE, RMSE, MAPE, R²) |
| `matplotlib` | Visualization |
| `holidays` | Ngày lễ Việt Nam (`holidays.VN`) |
| `openpyxl` | Đọc file `.xlsx` |

---

## ▶️ Hướng dẫn chạy lại kết quả

### Chạy trên Kaggle (khuyến nghị)

**Bước 1:** Tạo notebook mới trên [kaggle.com/code](https://www.kaggle.com/code)

**Bước 2:** Upload dữ liệu
- Vào tab **Data** → **Add Data** → **Upload** → chọn file `train.xlsx`
- Kaggle sẽ mount file vào `/kaggle/input/<tên-dataset>/train.xlsx`
- Sửa dòng config trong notebook:
```python
DATA_PATH = "/kaggle/input/<tên-dataset>/train.xlsx"
```

**Bước 3:** Upload notebook
- Chọn **File** → **Import Notebook** → upload `train_model.ipynb`

**Bước 4:** Chạy toàn bộ
- **Run All** (`Shift + Enter` từng cell, hoặc menu **Run** → **Run All**)
- Thời gian ước tính: **15–30 phút** (tùy Optuna 30 trials × 2 models)

**Bước 5:** Tải kết quả
- File `submission.csv` xuất hiện trong tab **Output** → nhấn **Download**

---

### Chạy trên máy local

**Bước 1:** Cài đặt thư viện
```bash
pip install xgboost optuna shap pandas numpy scikit-learn matplotlib holidays openpyxl
```

**Bước 2:** Đặt file dữ liệu cùng thư mục với notebook
```
your-folder/
├── train_model.ipynb
└── train.xlsx
```

**Bước 3:** Đảm bảo `DATA_PATH` trong notebook là:
```python
DATA_PATH = "train.xlsx"
```

**Bước 4:** Chạy notebook
```bash
jupyter notebook train_model.ipynb
# hoặc
jupyter lab train_model.ipynb
```
Sau đó **Run All Cells**.

**Bước 5:** File `submission.csv` sẽ được tạo trong cùng thư mục.

---

## 🔁 Đảm bảo Reproducibility

Mọi nguồn ngẫu nhiên đã được cố định với `SEED = 42`:

| Thành phần | Cách cố định |
|---|---|
| Python built-in | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| Hash của Python | `os.environ["PYTHONHASHSEED"] = "42"` |
| XGBoost | `random_state=42`, `OMP_NUM_THREADS=1` |
| Optuna | `TPESampler(seed=42)` cho cả 2 study |
| SHAP sampling | `.sample(random_state=42)` |

> ⚠️ Kết quả số có thể lệch nhỏ nếu **phiên bản thư viện khác nhau**. Để đảm bảo 100%, hãy dùng cùng môi trường Kaggle (Python 3.10+, XGBoost 2.x).