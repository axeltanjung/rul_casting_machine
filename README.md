# RUL Prediction – Continuous Casting Machine

## Project Overview

This project focuses on **Remaining Useful Life (RUL) prediction** for a **continuous casting machine** in a steel manufacturing process.

The goal is to estimate the **remaining operational life** of key machine components based on:
- process parameters
- thermal and cooling signals
- metallurgical composition
- casting configuration
- production and operational context

Accurate RUL estimation enables:
- predictive maintenance
- reduced unplanned downtime
- improved equipment utilization
- data-driven maintenance scheduling

---

## Problem Definition

**Remaining Useful Life (RUL)** represents the amount of operational time or usage left before a machine or component requires maintenance or replacement.

In this project:
- **Target variable:** `RUL` (continuous value)
- **Task type:** Regression (time-to-failure / degradation modeling)
- **Domain:** Heavy industry – steel continuous casting

---

## Dataset Description

The dataset contains **time-stamped operational, physical, and chemical features** collected during the continuous casting process.

### Feature Groups

#### 1. Temporal & Production Context
| Feature | Description |
|------|------------|
| `date` | Timestamp of the production record |
| `cast_in_row` | Casting sequence number |
| `kind` | Casting or production category |
| `quantity, tonn` | Quantity produced |
| `sample_time_continuous_caster` | Sampling time at the caster |
| `sleeve` | Sleeve or component identifier |
| `num_crystallizer` | Crystallizer unit number |
| `num_stream` | Casting stream number |

---

#### 2. Workpiece & Geometry
| Feature | Description |
|------|------------|
| `workpiece_weight, tonn` | Actual workpiece weight |
| `steel_weight_theoretical, tonn` | Theoretical steel weight |
| `steel_weight, tonn` | Measured steel weight |
| `workpiece_slice_geometry` | Geometry of the cast slice |
| `technical_trim, tonn` | Trim losses during processing |
| `resistance, tonn` | Mechanical resistance during casting |

---

#### 3. Casting & Mechanical Parameters
| Feature | Description |
|------|------------|
| `swing_frequency, amount/minute` | Oscillation frequency |
| `crystallizer_movement, mm` | Crystallizer movement amplitude |
| `alloy_speed, meter/minute` | Casting speed |
| `grab1_num`, `grab2_num` | Grab operation counters |

---

#### 4. Thermal Measurements
| Feature | Description |
|------|------------|
| `steel_temperature_grab1, Celsius deg.` | Steel temperature at grab 1 |
| `time_temperature_measurement1` | Timestamp of first temperature measurement |
| `temperature_measurement1, Celsius deg.` | First measured temperature |
| `time_temperature_measurement2` | Timestamp of second temperature measurement |
| `temperature_measurement2, Celsius deg.` | Second measured temperature |

---

#### 5. Cooling & Water System
| Feature | Description |
|------|------------|
| `water_consumption, liter/minute` | Total water consumption |
| `water_temperature_delta, Celsius deg.` | Temperature difference of cooling water |
| `water_consumption_secondary_cooling_zone_num1, liter/minute` | Secondary cooling zone 1 |
| `water_consumption_secondary_cooling_zone_num2, liter/minute` | Secondary cooling zone 2 |
| `water_consumption_secondary_cooling_zone_num3, liter/minute` | Secondary cooling zone 3 |

---

#### 6. Metallurgical Composition (%)
Chemical composition of the steel, expressed as percentages:

`Ce`, `C`, `Si`, `Mn`, `S`, `P`, `Cr`, `Ni`, `Cu`, `As`, `Mo`, `Nb`,  
`Sn`, `Ti`, `V`, `Al`, `Ca`, `N`, `Pb`, `Mg`, `Zn`

These features strongly influence:
- solidification behavior
- thermal stress
- wear and degradation rate of casting equipment

---

#### 7. Material & Alloy Context
| Feature | Description |
|------|------------|
| `steel_type` | Steel grade classification |
| `alloy_type` | Alloy category |
| `doc_requirement` | Process or quality requirement |

---

#### 8. Residuals & Losses
| Feature | Description |
|------|------------|
| `slag_weight_close_grab1, tonn` | Slag weight after grab 1 |
| `metal_residue_grab1, tonn` | Residual metal after grab 1 |
| `residuals_grab2, tonn` | Residuals after grab 2 |

---

#### 9. Target Variable
| Feature | Description |
|------|------------|
| **`RUL`** | Remaining Useful Life of the casting machine/component |

---

## Modeling Objective

Build a model that can:
- predict **continuous RUL values**
- learn degradation patterns from multivariate signals
- generalize across steel grades, alloys, and operating regimes

---

## Challenges & Considerations

- Strong multicollinearity between process variables
- Mixed data types (numerical, categorical, temporal)
- Nonlinear degradation behavior
- Operational regime shifts (different steel/alloy types)
- Potential censoring or truncation in RUL labels

---

## Planned Modeling Approach

- Exploratory Data Analysis (EDA)
- Feature engineering:
  - thermal deltas
  - rolling statistics
  - regime-based aggregation
- Baseline models:
  - Linear Regression
  - Random Forest / Gradient Boosting
- Advanced models:
  - XGBoost / LightGBM
  - Survival analysis (optional)
- Evaluation metrics:
  - MAE
  - RMSE
  - RUL-specific error tolerance

---

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis is conducted to understand **process behavior, degradation patterns, and data quality issues** before modeling RUL.

### 1. Data Quality & Integrity Checks
Key checks performed:
- Missing value analysis across process, thermal, and chemical features
- Consistency between:
  - `steel_weight_theoretical, tonn` vs `steel_weight, tonn`
  - temperature measurements at different timestamps
- Detection of physically impossible values (e.g., negative weights, extreme temperatures)

These checks ensure that observed patterns are **physically plausible**, not data artifacts.

---

### 2. Temporal Behavior & Degradation Trends
RUL is analyzed against:
- `date`
- `cast_in_row`
- `sample_time_continuous_caster`

Key observations:
- RUL shows **gradual degradation patterns**, not abrupt drops
- Certain casting sequences exhibit faster degradation, indicating **operational regime effects**
- Time-based grouping reveals non-stationary behavior across production periods

This confirms that RUL prediction is a **degradation modeling problem**, not simple regression.

---

### 3. Correlation & Multicollinearity Analysis
Correlation analysis highlights:
- Strong relationships among:
  - thermal variables (steel temperature, cooling water delta)
  - mechanical variables (swing frequency, crystallizer movement, casting speed)
- High correlation between chemical composition elements due to alloy design

Implications:
- Linear models may suffer from multicollinearity
- Tree-based or regularized models are preferred
- Feature grouping and dimensionality reduction may be beneficial

---

### 4. Operational Regime Analysis
EDA is segmented by:
- `steel_type`
- `alloy_type`
- `num_crystallizer`
- `num_stream`

Findings:
- Different steel grades exhibit distinct RUL decay patterns
- Certain alloys accelerate wear due to thermal and mechanical stress
- Equipment-specific behavior exists between crystallizers and streams

This motivates **regime-aware modeling** instead of a single global model.

---

### 5. Distribution & Outlier Analysis
- RUL distribution is right-skewed
- Long-tail behavior observed for low degradation scenarios
- Outliers often correspond to:
  - abnormal cooling conditions
  - extreme chemical compositions
  - unusual casting speeds

Outliers are analyzed case-by-case rather than blindly removed.

---

## Feature Engineering

Feature engineering focuses on **capturing degradation dynamics**, not just raw sensor values.

Key feature categories:
- **Thermal deltas**  
  - temperature differences between measurements
  - cooling water temperature gradients
- **Rate-based features**  
  - change in temperature per unit time
  - casting speed normalized by steel weight
- **Rolling statistics** (time-aware)
  - rolling mean, std, and slope of critical signals
- **Regime encodings**
  - steel and alloy types as categorical indicators
  - interaction terms between composition and thermal variables

All feature transformations are designed to be:
- physically interpretable
- consistent across training and inference

---

## Modeling Strategy

### 1. Problem Framing
- **Type:** Supervised regression
- **Target:** Continuous RUL
- **Prediction goal:** Estimate remaining usable life under current operating conditions

---

### 2. Baseline Models
Initial benchmarks include:
- Linear Regression (with regularization)
- Ridge / Lasso Regression

Purpose:
- establish baseline error
- understand linear contributions
- validate feature relevance

---

### 3. Tree-Based Models
Primary modeling focus:
- Random Forest Regressor
- Gradient Boosting Machines (XGBoost / LightGBM)

Advantages:
- handle nonlinear degradation behavior
- robust to multicollinearity
- capture interaction between thermal, mechanical, and chemical features

---

### 4. Advanced & Optional Approaches
- Survival analysis for time-to-failure modeling
- Quantile regression for uncertainty-aware RUL prediction
- Regime-specific models per steel or alloy group

---

### 5. Train–Validation Strategy
- Time-aware split to avoid data leakage
- Cross-validation within operational regimes
- No random shuffling across time sequences

This ensures realistic performance estimates.

---

## Model Evaluation

Evaluation focuses on **maintenance relevance**, not just statistical accuracy.

Metrics used:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- RUL-specific tolerance bands (early vs late prediction penalty)

Key considerations:
- Overestimating RUL is more costly than underestimating
- Error is analyzed as a function of true RUL value
- Performance is reported per operating regime

---

## Model Interpretability

Interpretability is critical for industrial adoption.

Methods used:
- Feature importance (tree-based)
- Partial dependence plots
- Local explanations for individual predictions

Insights from interpretability are validated against:
- process physics
- metallurgical knowledge
- SME feedback

---

## Key Insights from Analysis

- Thermal management is a dominant driver of degradation
- Alloy composition significantly modulates wear behavior
- Cooling efficiency directly impacts RUL
- Operational regimes must be explicitly modeled

---

## Limitations & Future Work

- Dataset may contain censored RUL values
- Long-term temporal dependencies require sequence models
- Integration with maintenance logs would improve labeling

Planned improvements:
- sequence-based models (LSTM / temporal boosting)
- uncertainty quantification
- cost-aware optimization for maintenance scheduling

## Disclaimer

This dataset represents **industrial process data** and is intended for:
- predictive maintenance research
- machine learning experimentation
- educational and benchmarking purposes

It does not represent any specific real-world plant or proprietary system.

---

## Author

Developed as part of a predictive maintenance and reliability analytics initiative.
