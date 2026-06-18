# Boston Housing 特徵選擇與回歸圖表產生系統開發規格書

**專案名稱**：AI Boston Housing Price Prediction and Feature Selection System  
**中文名稱**：AI Boston 房價預測與特徵選擇分析系統  
**文件版本**：v1.0  
**目標圖表**：`feature_selection_performance_allinone.png`  
**適用問題類型**：Supervised Learning / Regression Problem  
**CRISP-DM 階段**：Step 4 Modeling + Step 5 Evaluation  
**主要任務**：使用 Boston Housing 資料集，實作 9 種特徵選擇方法，逐步評估不同特徵數量下的模型表現，並輸出一張整合型圖表。

---

## 1. 專案目標

本專案要將原本的房價預測專題修改為 **Boston Housing 回歸分析問題**，並使用 Boston Housing 的實際資料產生一張整合型分析圖表。

圖表內容包含：

1. 不同特徵數量下的 **Test R-squared 分數**
2. 不同特徵數量下的 **Test MSE 分數**
3. 9 種特徵選擇方法的 **特徵排序表**
4. 標示最佳特徵數量甜蜜點，例如 `Sweet Spot (k=3)`
5. 輸出檔案名稱固定為：

```text
feature_selection_performance_allinone.png
```

---

## 2. 專題定位

### 2.1 原始方向

原本專案若包含以下內容：

- 台灣房價
- 實價登錄
- 居住品質分析
- 主題式推薦
- AQI / 氣象 / TDX
- 台灣地圖視覺化

本次要全部改成 Boston Housing 資料分析問題。

### 2.2 修改後方向

新專案主軸為：

> 使用 Boston Housing Dataset，分析不同特徵選擇方法對房價預測模型表現的影響，並找出最適合的特徵數量與關鍵特徵組合。

---

## 3. 資料集規格

### 3.1 資料來源

建議使用本機 CSV 檔案，不建議使用 `sklearn.datasets.load_boston()`，因為新版 scikit-learn 已移除此函式。

建議放置位置：

```text
data/boston_housing.csv
```

### 3.2 資料欄位

| 欄位 | 說明 | 類型 | 是否為特徵 |
|---|---|---:|---:|
| CRIM | per capita crime rate by town | numeric | yes |
| ZN | proportion of residential land zoned for large lots | numeric | yes |
| INDUS | proportion of non-retail business acres per town | numeric | yes |
| CHAS | Charles River dummy variable | binary | yes |
| NOX | nitric oxides concentration | numeric | yes |
| RM | average number of rooms per dwelling | numeric | yes |
| AGE | proportion of owner-occupied units built before 1940 | numeric | yes |
| DIS | weighted distances to employment centers | numeric | yes |
| RAD | index of accessibility to radial highways | numeric | yes |
| TAX | property tax rate | numeric | yes |
| PTRATIO | pupil-teacher ratio by town | numeric | yes |
| B | legacy demographic variable; use only with ethical warning | numeric | optional |
| LSTAT | percentage of lower status population | numeric | yes |
| MEDV | median value of owner-occupied homes | numeric | target |

### 3.3 欄位標準化

程式內統一將欄位轉成小寫，方便圖表顯示：

```text
CRIM → crim
ZN → zn
INDUS → indus
CHAS → chas
NOX → nox
RM → rm
AGE → age
DIS → dis
RAD → rad
TAX → tax
PTRATIO → ptratio
B → b
LSTAT → lstat
MEDV → medv
```

### 3.4 倫理提醒

Boston Housing 是經典教學資料集，但包含過時且具爭議的社會變數，例如 `B` 與 `LSTAT`。本專案只能作為機器學習教學、特徵選擇演算法比較與圖表產生練習，不應作為真實房價決策或商業決策依據。

---

## 4. 需要產生的圖表

### 4.1 圖表名稱

```text
CRISP-DM Step 4: 9 Feature Selection Algorithms Stepwise Evaluation (Boston Housing)
```

### 4.2 圖表尺寸

建議輸出：

```text
寬度：20 inch
高度：11 inch
DPI：150 或 200
格式：PNG
```

輸出路徑：

```text
reports/figures/feature_selection_performance_allinone.png
```

### 4.3 圖表版面

整張圖分成三個區域：

```text
上方左側：Test R-squared Score by Feature Subset Size
上方右側：Test MSE by Feature Subset Size
下方整列：Feature Ranking Table
```

建議使用 `matplotlib.gridspec.GridSpec`：

```python
fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(
    nrows=2,
    ncols=2,
    height_ratios=[2.1, 1.6],
    width_ratios=[1, 1]
)
ax_r2 = fig.add_subplot(gs[0, 0])
ax_mse = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :])
```

---

## 5. 特徵選擇演算法規格

本專案需實作 9 種特徵選擇方法。

| 編號 | 圖例名稱 | 方法名稱 | 實作方式 |
|---:|---|---|---|
| 1 | Pearson Corr | Pearson 相關係數 | `abs(corr(feature, target))` |
| 2 | Spearman Corr | Spearman 等級相關 | `spearmanr(feature, target)` |
| 3 | F-test Reg | F-test regression | `sklearn.feature_selection.f_regression` |
| 4 | Mutual Info | Mutual information regression | `sklearn.feature_selection.mutual_info_regression` |
| 5 | RFE | Recursive Feature Elimination | `sklearn.feature_selection.RFE` |
| 6 | SFS (Forward) | Sequential Forward Selection | `sklearn.feature_selection.SequentialFeatureSelector` |
| 7 | SBS (Backward) | Sequential Backward Selection | `SequentialFeatureSelector(direction="backward")` |
| 8 | Lasso (L1) | Lasso coefficient ranking | `sklearn.linear_model.LassoCV` |
| 9 | Random Forest | Feature importance | `RandomForestRegressor.feature_importances_` |

---

## 6. 評估流程規格

### 6.1 資料切分

固定使用：

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 6.2 特徵數量範圍

Boston Housing 有 13 個特徵，因此逐步評估：

```text
k = 1, 2, 3, ..., 13
```

### 6.3 每個演算法的流程

每一種特徵選擇方法都要先產生完整特徵排序。

接著針對每個 k：

```text
取出該方法排名前 k 名特徵
↓
使用相同模型訓練
↓
在 test set 上預測
↓
計算 R²
↓
計算 MSE
↓
記錄結果
```

### 6.4 評估模型

為了公平比較特徵選擇方法，所有方法預設使用相同模型：

```python
LinearRegression()
```

可選擇加入第二組實驗：

```python
RandomForestRegressor(random_state=42)
```

但主圖預設只用 `LinearRegression()`，避免不同模型造成比較失真。

### 6.5 評估指標

| 指標 | 說明 |
|---|---|
| R² | 模型解釋力，越高越好 |
| MSE | 平均平方誤差，越低越好 |

程式使用：

```python
from sklearn.metrics import r2_score, mean_squared_error
```

---

## 7. Sweet Spot 規格

### 7.1 預設標示

圖表需標示：

```text
Sweet Spot (k=3)
```

並在兩張折線圖上都畫出紅色垂直虛線。

```python
ax.axvline(x=3, color="red", linestyle=":", linewidth=2, alpha=0.7)
```

### 7.2 自動判斷規則

預設可先固定 k=3。  
進階版可自動判斷：

```text
在 R² 接近最高值 95% 以上的情況下，選擇特徵數量最少的 k。
```

範例：

```python
best_r2 = max(best_frontier_r2)
threshold = best_r2 * 0.95
sweet_k = min(k for k, r2 in results if r2 >= threshold)
```

---

## 8. Best Frontier 規格

圖中除了 9 種方法之外，還要額外繪製：

```text
Best (Frontier)
```

這不是第 10 種演算法，而是每個 k 對所有方法取最佳表現。

### 8.1 R² Frontier

```python
best_r2_at_k = max(r2_scores_by_method[method][k] for method in methods)
```

### 8.2 MSE Frontier

```python
best_mse_at_k = min(mse_scores_by_method[method][k] for method in methods)
```

---

## 9. 特徵排序表規格

### 9.1 表格欄位

底部表格需包含：

```text
Rank
Pearson
Spearman
F-test
Mutual Info
RFE
SFS (Fwd)
SBS (Bwd)
Lasso (L1)
Random Forest
```

### 9.2 表格列數

共有 13 列：

```text
Rank 1
Rank 2
...
Rank 13
```

### 9.3 表格內容

每一格顯示該方法在該排名下的特徵名稱，例如：

```text
Rank 1: lstat
Rank 2: rm
Rank 3: ptratio
```

### 9.4 表格視覺樣式

| 區域 | 樣式 |
|---|---|
| Header | 深藍底、白字、粗體 |
| Rank 欄 | 淺灰底、粗體 |
| 內容列 | 淺藍灰交錯底色 |
| 格線 | 淺藍灰 |
| 字體 | DejaVu Sans 或 Arial |
| 尺寸 | 9~10 pt |

---

## 10. 專案資料夾結構

建議調整為：

```text
boston-housing-feature-selection/
│
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── boston_housing.csv
│   └── README_Boston_Housing_Data.md
│
├── notebooks/
│   ├── 01_boston_data_understanding.ipynb
│   ├── 02_boston_feature_selection.ipynb
│   └── 03_boston_model_evaluation.ipynb
│
├── scripts/
│   └── generate_feature_selection_chart.py
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── model_evaluation.py
│   └── visualization.py
│
├── reports/
│   ├── Boston_Housing_CRISP_DM_Report.md
│   └── figures/
│       └── feature_selection_performance_allinone.png
│
└── docs/
    ├── Boston_Housing_Project_SPEC.md
    ├── Boston_Housing_Data_Dictionary.md
    ├── Boston_Housing_Feature_Selection_Methods.md
    └── Boston_Housing_Ethical_Note.md
```

---

## 11. Markdown 檔案改名規格

如果原專案中有台灣房價、居住品質、推薦系統相關 `.md`，請全部改名。

### 11.1 建議改名表

| 原本可能檔名 | 新檔名 |
|---|---|
| `AI_Taiwan_Housing_Quality_Recommendation_SPEC.md` | `Boston_Housing_Feature_Selection_Project_SPEC.md` |
| `Taiwan_Housing_Project_Report.md` | `Boston_Housing_CRISP_DM_Report.md` |
| `Taiwan_Housing_Data_Dictionary.md` | `Boston_Housing_Data_Dictionary.md` |
| `Taiwan_Housing_Recommendation_System.md` | `Boston_Housing_Regression_System.md` |
| `Housing_Quality_Analysis.md` | `Boston_Housing_Feature_Analysis.md` |
| `Theme_Based_Recommendation.md` | `Boston_Housing_Feature_Selection_Methods.md` |
| `Environment_Monitoring.md` | `Boston_Housing_Data_Understanding.md` |

### 11.2 Markdown 內容修改規則

所有 `.md` 檔案需掃描並替換：

| 舊內容 | 新內容 |
|---|---|
| 台灣房價 | Boston Housing 房價 |
| 實價登錄 | Boston Housing Dataset |
| 居住品質推薦 | 房價預測與特徵選擇 |
| 主題式推薦 | 特徵選擇方法比較 |
| AQI / 氣象 / TDX | CRIM / RM / LSTAT / PTRATIO 等特徵 |
| 分類問題 | 回歸問題 |
| 推薦結果 | 預測結果與特徵重要性 |
| 使用者偏好 | 特徵子集合 |
| 推薦分數 | R² / MSE |

---

## 12. 核心程式模組規格

### 12.1 `data_loader.py`

功能：

- 讀取 `data/boston_housing.csv`
- 欄位轉小寫
- 檢查是否包含 `medv`
- 回傳 `X`, `y`, `feature_names`

必要函式：

```python
def load_boston_csv(path: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    pass
```

---

### 12.2 `preprocessing.py`

功能：

- 檢查缺失值
- 數值欄位轉型
- train/test split
- StandardScaler

必要函式：

```python
def split_data(X, y, test_size=0.2, random_state=42):
    pass

def build_scaled_model(model):
    pass
```

---

### 12.3 `feature_selection.py`

功能：

- 產生 9 種方法的完整特徵排序
- 回傳格式統一為 dictionary

必要函式：

```python
def rank_by_pearson(X_train, y_train) -> list[str]:
    pass

def rank_by_spearman(X_train, y_train) -> list[str]:
    pass

def rank_by_f_test(X_train, y_train) -> list[str]:
    pass

def rank_by_mutual_info(X_train, y_train, random_state=42) -> list[str]:
    pass

def rank_by_rfe(X_train, y_train) -> list[str]:
    pass

def rank_by_sfs_forward(X_train, y_train) -> list[str]:
    pass

def rank_by_sbs_backward(X_train, y_train) -> list[str]:
    pass

def rank_by_lasso(X_train, y_train, random_state=42) -> list[str]:
    pass

def rank_by_random_forest(X_train, y_train, random_state=42) -> list[str]:
    pass

def get_all_feature_rankings(X_train, y_train) -> dict[str, list[str]]:
    pass
```

---

### 12.4 `model_evaluation.py`

功能：

- 根據每個方法的 top-k 特徵訓練模型
- 計算 R² 與 MSE
- 產生 Best Frontier

必要函式：

```python
def evaluate_feature_subset(
    X_train,
    X_test,
    y_train,
    y_test,
    selected_features,
    model=None
) -> dict:
    pass

def evaluate_all_methods_stepwise(
    X_train,
    X_test,
    y_train,
    y_test,
    rankings,
    max_k=13
) -> dict:
    pass

def compute_best_frontier(results) -> dict:
    pass
```

---

### 12.5 `visualization.py`

功能：

- 繪製 R² 折線圖
- 繪製 MSE 折線圖
- 繪製特徵排名表
- 匯出 `feature_selection_performance_allinone.png`

必要函式：

```python
def plot_feature_selection_allinone(
    results,
    rankings,
    sweet_k=3,
    output_path="reports/figures/feature_selection_performance_allinone.png"
) -> None:
    pass
```

---

### 12.6 `generate_feature_selection_chart.py`

功能：

- 主程式入口
- 可從命令列指定資料路徑與輸出路徑

CLI 使用方式：

```bash
python scripts/generate_feature_selection_chart.py \
  --data data/boston_housing.csv \
  --output reports/figures/feature_selection_performance_allinone.png \
  --sweet-k 3
```

---

## 13. 主要 Python 套件

`requirements.txt`：

```text
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scipy>=1.10
scikit-learn>=1.2
joblib>=1.3
```

可選：

```text
mlxtend>=0.23
```

如果不用 `mlxtend`，請直接使用 scikit-learn 的 `SequentialFeatureSelector` 實作 forward 與 backward selection。

---

## 14. 圖表視覺規範

### 14.1 折線圖共同規格

- X 軸：`Number of Features in Model`
- 左圖 Y 軸：`Test R-squared (R²)`
- 右圖 Y 軸：`Test Mean Squared Error (MSE)`
- 需顯示 grid
- 需顯示 legend
- 需顯示 marker
- 需標示 Sweet Spot

### 14.2 線條與 marker 建議

| 方法 | Marker |
|---|---|
| Pearson Corr | `o` |
| Spearman Corr | `v` |
| F-test Reg | `^` |
| Mutual Info | `>` |
| RFE | `s` |
| SFS (Forward) | `D` |
| SBS (Backward) | `*` |
| Lasso (L1) | `d` |
| Random Forest | `p` |
| Best (Frontier) | `D`，線條較粗 |

### 14.3 標題字體

```python
fig.suptitle(
    "CRISP-DM Step 4: 9 Feature Selection Algorithms Stepwise Evaluation (Boston Housing)",
    fontsize=18,
    fontweight="bold"
)
```

---

## 15. 報告輸出規格

除了 PNG 圖表之外，建議額外輸出：

```text
reports/feature_selection_results.csv
reports/feature_rankings.csv
reports/Boston_Housing_CRISP_DM_Report.md
```

### 15.1 `feature_selection_results.csv`

欄位：

```text
method,k,features,r2,mse
```

### 15.2 `feature_rankings.csv`

欄位：

```text
rank,pearson,spearman,f_test,mutual_info,rfe,sfs_forward,sbs_backward,lasso_l1,random_forest
```

---

## 16. 開發步驟

### Step 1：資料準備

1. 建立 `data/` 資料夾
2. 放入 `boston_housing.csv`
3. 確認欄位包含 `MEDV`
4. 程式讀取後轉成小寫欄位

### Step 2：資料理解

1. 顯示資料 shape
2. 顯示欄位名稱
3. 檢查缺失值
4. 檢查 target `medv` 分布
5. 顯示特徵相關係數

### Step 3：資料前處理

1. 切分 `X` 與 `y`
2. train/test split
3. 建立 scaling pipeline

### Step 4：特徵選擇

1. 執行 9 種特徵選擇方法
2. 取得完整 feature ranking
3. 將結果儲存成 `feature_rankings.csv`

### Step 5：逐步模型評估

1. 對每種方法從 k=1 到 k=13 評估
2. 計算 R²
3. 計算 MSE
4. 儲存成 `feature_selection_results.csv`

### Step 6：產生圖表

1. 建立 all-in-one figure
2. 左圖畫 R²
3. 右圖畫 MSE
4. 下方畫 ranking table
5. 標示 Sweet Spot
6. 儲存 PNG

### Step 7：產生報告

1. 自動插入圖表
2. 解釋主要發現
3. 說明最佳特徵數量
4. 說明重要特徵
5. 加上倫理提醒

---

## 17. 驗收標準

### 17.1 功能驗收

執行以下指令後：

```bash
python scripts/generate_feature_selection_chart.py \
  --data data/boston_housing.csv \
  --output reports/figures/feature_selection_performance_allinone.png
```

必須成功產生：

```text
reports/figures/feature_selection_performance_allinone.png
```

### 17.2 圖表驗收

圖表必須包含：

- [ ] 大標題：CRISP-DM Step 4...
- [ ] 左側 R² 折線圖
- [ ] 右側 MSE 折線圖
- [ ] 9 種方法線條
- [ ] Best Frontier 線條
- [ ] Sweet Spot 紅色虛線
- [ ] 下方 13 rank 特徵排序表
- [ ] 圖表文字清楚可讀
- [ ] PNG 解析度足夠放入報告或簡報

### 17.3 資料驗收

- [ ] 使用 Boston Housing CSV 實際資料
- [ ] 不可用假資料
- [ ] 不可用 AI 圖片生成
- [ ] 結果必須可重現
- [ ] 所有 random_state 固定為 42

### 17.4 文件驗收

- [ ] 所有 `.md` 檔案已改成 Boston Housing 主題
- [ ] 不再出現台灣實價登錄、AQI、TDX 等非本專案內容
- [ ] 報告有放入 `feature_selection_performance_allinone.png`
- [ ] README 可清楚說明如何執行程式

---

## 18. README.md 應包含內容

`README.md` 必須更新為：

```markdown
# AI Boston Housing Price Prediction and Feature Selection System

This project uses the Boston Housing dataset to compare 9 feature selection algorithms for a regression problem.

## Goal

Predict MEDV, the median house value, and evaluate how different feature subset sizes affect model performance.

## Main Output

![Feature Selection Performance](reports/figures/feature_selection_performance_allinone.png)

## Workflow

1. Load Boston Housing dataset
2. Clean and inspect data
3. Run 9 feature selection algorithms
4. Evaluate feature subsets from k=1 to k=13
5. Compare Test R² and Test MSE
6. Generate an all-in-one performance chart

## Run

```bash
pip install -r requirements.txt
python scripts/generate_feature_selection_chart.py
```

## Ethical Note

The Boston Housing dataset is outdated and contains problematic variables. It should only be used for educational practice.
```

---

## 19. Codex / Gemini 開發提示語

請將以下提示語交給開發 Agent：

```text
You are a senior Python machine learning engineer.

Modify the existing project into a Boston Housing regression project.

The goal is to use the actual Boston Housing dataset to generate an all-in-one chart named:

feature_selection_performance_allinone.png

The chart must reproduce the structure of the reference image:
1. Top-left line chart: Test R-squared score by feature subset size.
2. Top-right line chart: Test MSE by feature subset size.
3. Bottom table: feature rankings for 9 feature selection algorithms.
4. Add a red vertical dotted line at k=3 labeled "Sweet Spot (k=3)".
5. Add a Best Frontier line that takes the best R² or lowest MSE across methods at each k.

Use the following 9 feature selection methods:
- Pearson correlation
- Spearman correlation
- F-test regression
- Mutual information regression
- RFE
- Sequential Forward Selection
- Sequential Backward Selection
- Lasso L1 coefficients
- Random Forest feature importance

Use the Boston Housing dataset from data/boston_housing.csv.
Do not use fake data.
Do not use image generation.
Do not use sklearn.datasets.load_boston because it has been removed in newer scikit-learn versions.
The target variable is MEDV.
Convert feature names to lowercase for display.

Use train_test_split with test_size=0.2 and random_state=42.
Use LinearRegression as the default evaluation model.
Evaluate top-k features for k=1 to k=13.
Compute R² and MSE on the test set.

Create the following files:
- scripts/generate_feature_selection_chart.py
- src/data_loader.py
- src/preprocessing.py
- src/feature_selection.py
- src/model_evaluation.py
- src/visualization.py
- reports/figures/feature_selection_performance_allinone.png
- reports/feature_selection_results.csv
- reports/feature_rankings.csv
- reports/Boston_Housing_CRISP_DM_Report.md

Also scan all Markdown files and rename or rewrite them so they fit the Boston Housing regression problem.
Remove Taiwan housing, real price registration, AQI, weather, TDX, and recommendation-system content unless placed in a clearly marked Future Extension section.

At the end, provide a summary of:
1. Files created.
2. Markdown files renamed.
3. Charts generated.
4. Main findings from the feature selection experiment.
```

---

## 20. 注意事項

1. 這張圖必須由 Python 根據 Boston Housing 資料計算後產生。
2. 不可用 AI 生成圖片代替資料視覺化。
3. 每次執行結果應盡量一致。
4. 圖表的重點不是追求最高模型分數，而是比較不同特徵選擇方法在不同特徵數量下的表現。
5. 若實際數值與參考圖不完全相同是正常的，因為會受到資料版本、切分方式、模型、scaler、演算法參數影響。
6. 但圖表結構、方法數量、評估流程與輸出格式必須一致。

---

## 21. 完成定義 Definition of Done

本專案完成時，應能做到：

```text
輸入：data/boston_housing.csv
處理：9 種特徵選擇方法 + top-k stepwise evaluation
輸出：feature_selection_performance_allinone.png
報告：Boston_Housing_CRISP_DM_Report.md
```

且圖表需清楚呈現：

```text
哪些特徵選擇方法表現最好？
使用幾個特徵時已經達到足夠好的模型表現？
哪些特徵最常被選為重要特徵？
是否能用少量特徵達到接近完整模型的預測效果？
```
