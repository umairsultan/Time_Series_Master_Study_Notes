# Time Series Analysis & Prediction — Master Study Notes
### Final Exam Revision | UET Taxila, MS DS | Dr. Farooq Ali

> **Scope:** Whole course (Weeks 1–12). Deep Learning (RNN/LSTM/GRU) is **excluded** — it was taught as a separate subject.
> **Exam DNA (from the mid-term):** part *concept-recognition MCQs*, part *ACF/PACF plot reading*, part *by-hand numerical calculation*. The most repeated single skill is **reading ACF/PACF to name a model** — prioritise it.

---

## PART A — Foundations (Weeks 1–3)

### A1. What is a Time Series?
A sequence of observations recorded at successive, ordered time intervals: **{Xₜ}, t = 1, 2, 3, …**
The defining feature is **temporal dependency** — the current value can depend on past values, and **the ordering is meaningful** (you cannot shuffle the data).

### A2. Time Series vs Cross-Sectional Data

| Aspect | Time Series | Cross-Sectional |
|---|---|---|
| Time dependency | Present | Absent |
| Order matters | Yes | No |
| Example | Stock price over days | Survey of many people at one moment |
| Focus | Trends, patterns over time | Group comparisons |

### A3. Types of Time Series
- **Discrete** (fixed intervals: monthly sales, yearly GDP) vs **Continuous** (recorded continuously: ECG, sensor data).
- **Univariate** (one variable over time: daily temperature) vs **Multivariate** (several variables together: temperature + humidity + pressure — lets you model interdependencies).

### A4. The Four Components
**Xₜ = Trend + Seasonality + Cyclic + Irregular**

- **Trend (T):** long-term upward/downward movement; smooth, persistent (e.g. rising population).
- **Seasonality (S):** regular repeating pattern at a **fixed, known period** (e.g. sales peak every December).
- **Cyclic (C):** long-term wave-like fluctuation with **no fixed period**, driven by external/economic factors (business cycles).
- **Irregular (I):** random, unpredictable "noise" (disasters, shocks, measurement error).

**⚠️ Exam trap — Seasonal vs Cyclic:**

| | Seasonal | Cyclic |
|---|---|---|
| Period | **Fixed** | **Variable** |
| Predictability | High | Low |
| Cause | Calendar-related | Economic / external |

> Seasonality is **easier to estimate** than cyclic (fixed, predictable period). *(Answers mid-term Q1.2.)*

### A5. Additive vs Multiplicative Models

| Model | Formula | Use when… |
|---|---|---|
| **Additive** | Xₜ = Tₜ + Sₜ + Cₜ + Iₜ | Seasonal swings are **constant** in size over time |
| **Multiplicative** | Xₜ = Tₜ × Sₜ × Cₜ × Iₜ | Seasonal swings **grow/shrink with the trend** (common in finance) |

### A6. Decomposition
Splitting a series into Trend + Seasonal + Residual. Helps understand structure, choose a model, and improve forecasts. Conceptual steps: estimate trend → remove trend to expose seasonality → isolate residual.

### A7. EDA & Visualisation
- **Time plot:** values vs time — spot trend, outliers, volatility, rough stationarity.
- **Lag plot:** Xₜ vs Xₜ₋₁ — a clear pattern ⇒ autocorrelation present.
- **Seasonal plot:** overlay periods to expose repeating shape.

### A8. Python Toolkit
NumPy (numerical) · Pandas (time-indexed data) · Matplotlib (plots) · **Statsmodels** (ARIMA, ADF, ACF/PACF) · Scikit-learn (ML + metrics).

---

## PART B — Stationarity (Week 4)

### B1. Definition
A series is **stationary** if its statistical properties don't change over time. Required by AR, MA, ARMA, ARIMA.

**Strict stationarity:** the *entire joint distribution* is time-invariant — very strong, rare in practice.

**Weak (covariance) stationarity** — the practical one — needs **three** conditions:
1. **Constant mean:** E(Yₜ) = μ
2. **Constant variance:** Var(Yₜ) = σ²
3. **Autocovariance depends only on the lag k, not on time t:** Cov(Yₜ, Yₜ₊ₖ) = γ(k)

> **⚠️ Exam trap (Q1.4):** "Time series is **Gaussian**" is **NOT** a required condition for weak stationarity. Constant mean, |s−t|-only autocovariance, and finite variance *are* required.

### B2. Signs of Non-Stationarity
Trend (drifting mean), seasonality, or **changing variance** (e.g. volatility clustering). Quick check: **rolling mean / rolling variance** — if they wander over time, it's non-stationary.

### B3. Transformations to Achieve Stationarity
- **Differencing** — removes trend: **Y′ₜ = Yₜ − Yₜ₋₁** (apply twice if needed). Most common; this is the "I" in ARIMA. *(e.g. 10,15,20,25,30 → differences all = 5, now flat.)*
- **Log transformation** — stabilises **growing variance / exponential growth**: Y′ₜ = log(Yₜ). *(e.g. 100,200,400,800,1600 doubling → log makes growth linear.)*
- **Detrending** — fit a regression line, subtract fitted values.
- **Moving average** — smooths to expose/remove trend.

### B4. Unit Root & Random Walk
**Random walk:** Yₜ = Yₜ₋₁ + εₜ. Shocks have **permanent** effects, variance grows ⇒ non-stationary (it "contains a unit root"). Differencing gives Yₜ − Yₜ₋₁ = εₜ, which is stationary.

### B5. ADF (Augmented Dickey–Fuller) Test
Formal stationarity test.
- **H₀:** series has a unit root ⇒ **non-stationary**
- **H₁:** series is **stationary**
- **Decision:** **p ≤ 0.05 → reject H₀ → stationary.** p > 0.05 → non-stationary.
- More **negative** ADF statistic = stronger evidence of stationarity.

**Workflow:** plot → spot trend/variance → difference if needed → ADF test → confirm before AR/MA modelling.

---

## PART C — Autocorrelation: ACF & PACF (Week 5) ★ HIGH-YIELD ★

### C1. Core definitions
- **Autocorrelation** ρₖ = correlation between Yₜ and its lag-k self, **Yₜ₋ₖ**.
  Formula: **ρₖ = Cov(Yₜ, Yₜ₋ₖ) / Var(Yₜ)**. Properties: −1 ≤ ρₖ ≤ 1, and ρ₀ = 1.
- **Lag (k):** "how many steps back am I looking." Lag 1 = yesterday, Lag 7 = one week ago.
- **Autocovariance** measures **linear dependence between two points on the *same* series at *different* times**. *(Answers mid-term Q1.3.)*

### C2. ACF vs PACF
- **ACF** = **total** correlation at lag k (direct **+** indirect ripple effects through intermediate lags). Plotted as a **correlogram**.
- **PACF** = **direct** correlation at lag k with the in-between lags' influence **removed**.

### C3. Reading the plots — two shapes
- **Cut off:** drops suddenly to ≈0 (inside the confidence band) after some lag and stays there. *Sharp cliff.*
- **Tail off:** shrinks gradually toward 0 over many lags. *Slow decay.*

### C4. ★ THE MODEL-IDENTIFICATION TABLE (memorise) ★

| Model | ACF | PACF |
|---|---|---|
| **AR(p)** | tails off | **cuts off after lag p** |
| **MA(q)** | **cuts off after lag q** | tails off |
| **ARMA(p,q)** | tails off | tails off |

**Memory hook:** the plot that **CUTS OFF** gives you the order — count the significant spikes before the cliff.
- **PACF cuts off → gives p** (AR order).
- **ACF cuts off → gives q** (MA order).
- Both tail off → **ARMA**.

Extra clues: a **slowly-decaying ACF** ⇒ **non-stationary** (trend present, difference it). **Near-zero everywhere** ⇒ white noise.

### C5. Confidence bands
Plotted at **±1.96 / √n** (n = number of observations). A spike **outside** the band is statistically significant; **inside** = ignore it.

---

## PART D — Classical Forecasting & Numerical Methods (Week 6) ★ CALCULATIONS ★

### D1. Simple Moving Average (SMA)
Average of the last *w* observations. 3-period MA at time t = (Yₜ + Yₜ₋₁ + Yₜ₋₂)/3.
> **Q1.6 worked:** demand 100(Oct), 200(Nov), 300(Dec), 400(Jan). 3-month SMA forecast for **Feb** = last 3 = (200+300+400)/3 = **300**.
> Code: `series.rolling(window=3).mean()`.

### D2. Weighted Moving Average (WMA)
Recent periods get bigger weights (weights sum to 1).
> Example (weights 0.4 most-recent, 0.4, 0.2): forecast = 0.4·Yₜ + 0.4·Yₜ₋₁ + 0.2·Yₜ₋₂.

### D3. Centered Moving Average (CMA) — for even windows
An even-order MA (e.g. 4-period) falls *between* periods, so you **average two consecutive MAs** to re-centre on a whole period. Used to estimate **trend** in seasonal data.
> **Q4B worked** — sales 40,44,48,52,60,64,68,72:
> - 4-period MAs: 46, 51, 56, 61, 66
> - Centered MAs (trend, periods 3–6): **(46+51)/2=48.5, (51+56)/2=53.5, (56+61)/2=58.5, (61+66)/2=63.5**

### D4. Simple Exponential Smoothing (SES)
Weights decay smoothly into the past via a smoothing constant **α** (0<α<1).
**Fₜ₊₁ = α·Yₜ + (1−α)·Fₜ**  (higher α = more reactive to recent data).

### D5. Holt & Holt–Winters
- **Holt's linear trend:** SES **+ a trend term** (two equations: level + trend) — for data with trend, no seasonality.
- **Holt–Winters:** adds a **seasonal** term (level + trend + seasonal) — for trend **and** seasonality. Comes in additive/multiplicative flavours.

### D6. Least-Squares Trend Fitting
Fit a straight line **Ŷ = a + bX** (X = time index) by least squares.
- Slope **b** = average change per period (the "rate of increase").
- Trend values = plug each X into the line. "Eliminate trend" = look at residuals (Actual − Trend), leaving **seasonal + cyclic + irregular**.

### D7. Seasonal Indices (quarterly decomposition)
Standard **ratio-to-trend** recipe (multiplicative):
1. Fit the trend line (least squares).
2. For each period, ratio = (Actual / Trend) × 100.
3. Average the ratios **by season** (all Q1s together, all Q2s, …) → **seasonal index** for each quarter.
4. (Normalise so the 4 indices sum to 400.)
5. **Estimate** a quarter = Trend × Index/100.
6. **% variation** = (Actual − Estimate)/Estimate × 100.
*(Full worked version of the exam's XYZ Ltd. problem is in the Appendix.)*

---

## PART E — AR / MA / ARMA / ARIMA (Weeks 7–8)

### E1. AR(p) — AutoRegressive
Present value = function of **past values**. Yₜ = c + φ₁Yₜ₋₁ + … + φₚYₜ₋ₚ + εₜ.
Signature: **ACF tails off, PACF cuts off at p.**

### E2. MA(q) — Moving Average
Present value = function of **past forecast errors**. Yₜ = c + εₜ + θ₁εₜ₋₁ + … + θqεₜ₋q.
Signature: **ACF cuts off at q, PACF tails off.**

### E3. ARMA(p,q)
Combines AR + MA (stationary data only). **Both ACF and PACF tail off.**

### E4. ARIMA(p, d, q)
**A**uto**R**egressive **I**ntegrated **M**oving **A**verage — for **non-stationary** data.
- **p** = AR terms (from PACF) · **d** = number of differences to reach stationarity · **q** = MA terms (from ACF).

**Modelling procedure:**
1. Collect data → 2. Check stationarity (plot + ADF) → 3. Difference to fix (sets **d**) → 4. Identify **p, q** from PACF/ACF → 5. Estimate coefficients (MLE / least squares) → 6. Diagnostics on residuals → 7. Forecast.

> **Worked identifications:**
> - Q1.8: ACF tails off, PACF cuts off at lag 1 → **AR(1), MA(0)**.
> - Q4A: stationary after **1st** differencing, ACF gradual, PACF cuts off at **lag 2** → **ARIMA(2, 1, 0)**.

### E5. ★ Reading the ARIMA `summary()` output ★ (from Dr. Farooq's notebook)
- **AIC / BIC:** model comparison — **lower is better** (reward fit, penalise complexity).
- **coef (ar.L1, ma.L1):** the estimated φ and θ.
- **P>|z| (p-value):** coefficient significance — **< 0.05 = significant/keep it.**
- **Ljung–Box (Q):** tests for leftover autocorrelation in residuals — **p > 0.05 = good** (residuals are white noise).
- **Jarque–Bera (JB):** residual normality — high p = normal (desirable).
- **Heteroskedasticity (H):** constant residual variance? **low p = variance NOT constant** (a warning; often means you need SARIMA / variance modelling).

### E6. Residual diagnostics (model validation)
Good residuals are **random, mean ≈ 0, no pattern**, and their **ACF has no significant spikes**. If the residual ACF shows spikes (especially at seasonal lags 12, 24…), the model is inadequate → adjust p,q or go **seasonal (SARIMA)**.

---

## PART F — SARIMA & Forecast Evaluation (Weeks 9–10)

### F1. SARIMA(p, d, q)(P, D, Q)ₛ
Extends ARIMA to handle a **repeating seasonal wave** ARIMA can't capture.
- lowercase **(p,d,q)** = non-seasonal part (as before).
- uppercase **(P,D,Q)** = the same three ideas applied at the **seasonal lag** (seasonal AR, seasonal differencing, seasonal MA).
- **s = season length:** 12 (monthly), 4 (quarterly), 7 (daily w/ weekly cycle).
- **When to use:** a fitted ARIMA's **residual ACF still spikes at seasonal lags** → switch to SARIMA.
- Code: `SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,12))`.

### F2. Forecast Accuracy Metrics (lower = better)
Given actual yₜ and forecast ŷₜ:

| Metric | Formula | Note |
|---|---|---|
| **MAE** | mean of \|yₜ − ŷₜ\| | avg error size, original units |
| **MSE** | mean of (yₜ − ŷₜ)² | squares errors — **big misses hurt more** |
| **RMSE** | √MSE | original units; penalises large errors |
| **MAPE** | mean of (\|yₜ − ŷₜ\| / \|yₜ\|) × 100% | **percentage**, scale-free |

---

## PART G — Multivariate Time Series (Week 11)

### G1. Cross-Correlation
Measures similarity between **two different series** as one is shifted (lagged) relative to the other; identifies **lead/lag relationships**. Ranges −1 to 1. (Autocorrelation = a series with *itself*; cross-correlation = *two* series.) Used in finance, engineering, neuroscience.

### G2. VAR — Vector AutoRegression
Generalises AR to **several interdependent series at once**: each variable is regressed on the **past values of itself AND all the other variables**. Captures two-way dynamics (requires stationary series). Order = number of lags included.

### G3. Granger Causality
A statistical test: series X "**Granger-causes**" Y if **past values of X improve the prediction of Y** beyond Y's own past. It's about **predictive precedence, not true causation** — always state this caveat.

---

## PART H — Machine Learning for Time Series (Week 12)

### H1. Time Series as Supervised Learning
Reframe forecasting as regression: **input = past lags, target = next value**. This lets you use any ML regressor on time series.

### H2. Lag Features & Feature Engineering
Turn the series into a feature table: **lag features** (Yₜ₋₁, Yₜ₋₂, …), rolling means/std, and **date parts** (month, day-of-week, quarter). This engineered table is what the ML model learns from.
> **Careful with ordering:** split train/test **chronologically** (no shuffling) to avoid leaking the future.

### H3. Regression & Tree-Based Models (incl. LightGBM)
- **Linear regression** on lag features = a simple, interpretable baseline.
- **Tree-based / gradient boosting (LightGBM)** captures **non-linear** patterns and interactions, handles many engineered features well, and is fast — a strong modern baseline for tabular/time-series forecasting.
- **Classical (ARIMA/SARIMA) vs ML:** classical models are built on explicit statistical assumptions (stationarity) and are great for clear trend/seasonality; ML models are flexible, assumption-light, and shine with many features/exogenous variables — but need careful feature engineering and validation.

---

# QUICK-REFERENCE APPENDIX

### ★ Model identification at a glance
| Pattern | Model |
|---|---|
| PACF cuts off at p, ACF tails off | **AR(p)** |
| ACF cuts off at q, PACF tails off | **MA(q)** |
| Both tail off | **ARMA(p,q)** |
| ACF decays *very slowly* | **Non-stationary → difference** |
| Residual ACF spikes at seasonal lags | **Use SARIMA** |

### Key formulas
- Autocorrelation: ρₖ = Cov(Yₜ, Yₜ₋ₖ) / Var(Yₜ)
- First difference: Y′ₜ = Yₜ − Yₜ₋₁
- Confidence band: ±1.96/√n
- SES: Fₜ₊₁ = αYₜ + (1−α)Fₜ
- Trend line: Ŷ = a + bX
- RMSE = √MSE ; MAPE = mean(|error|/|actual|)×100%

### Decision rules
- **ADF:** p ≤ 0.05 → stationary.
- **Coefficient p-value:** < 0.05 → significant.
- **Ljung–Box p:** > 0.05 → residuals OK (good model).
- **AIC/BIC:** lower → better.

### Common MCQ traps
- Gaussian is **not** required for weak stationarity.
- Seasonality (fixed period) is **easier** to estimate than cyclic (variable period).
- A monthly up-in-Jan/down-in-Dec pattern **is** seasonality (repeats at a fixed period).
- Naïve, exponential smoothing, and moving average are **all** time-series methods.
- Forecasting bookings / sales / calls for a future horizon are **all** time-series problems.

---

# APPENDIX — Worked Mid-Term Answers

### Q1 — MCQs (with reasoning)
1. *Not a time-series model?* → **D) None of the above** (naïve, exp. smoothing, MA are all TS methods).
2. *Easier to estimate?* → **A) Seasonality**.
3. *Autocovariance measures?* → **D) linear dependence between two points on the same series at different times**.
4. *Not needed for weak stationarity?* → **D) Time series is Gaussian**.
5. *ACF shape → AR or MA?* → if the ACF **tails off** → **AR**; if it **cuts off** → **MA** (read your plot's shape).
6. *3-month SMA for Feb 2017* → (200+300+400)/3 = **A) 300**.
7. *Lag-1 sample autocorrelation* → **C) 0.13** (computed).
8. *AR/MA terms from ACF tail-off + PACF cutoff@1* → **A) AR(1) MA(0)**.
9. *Jan–Mar up, Nov–Dec down — seasonality?* → **A) TRUE**.
10. *Which are TS problems?* → **E) 1, 2 and 3**.

### Q2 — Which detrended ACF for ARMA?
Prefer the series whose **ACF cuts off / dies quickly**. Justification: a **slowly-decaying ACF signals leftover non-stationarity** (trend not fully removed); ARMA requires stationary input, so choose the sharp-cutoff version.

### Q4A — Identify the model
d = 1 (first differencing), ACF gradual (tails off), PACF cuts off at lag 2 (p = 2, q = 0) → **ARIMA(2, 1, 0)**.

### Q4B — Centered moving averages
4-period MAs: 46, 51, 56, 61, 66 → **Centered MAs (trend): 48.5, 53.5, 58.5, 63.5** (periods 3–6).

### Q3 — Quarterly trend + seasonal indices (XYZ Ltd.)
> **Assumption:** multiplicative *ratio-to-trend* method (state this if your teacher used a different convention — e.g. additive, or trend fitted on yearly averages — the mechanics are the same).

**(i) Trend (least squares, X = 1…16 quarters):**  **Ŷ = 57.45 + 3.682·X**

**Seasonal indices** (avg ratio-to-trend, ≈ sum to 400):
Q1 ≈ **55.7**, Q2 ≈ **89.7**, Q3 ≈ **170.4**, Q4 ≈ **84.3**

**2004 trend values** (X = 13…16): 105.3, 109.0, 112.7, 116.4

**(ii) Estimated 2004 sales** = Trend × Index/100:
Q1 ≈ 58.7 · Q2 ≈ 97.7 · Q3 ≈ 192.0 · Q4 ≈ 98.1

**(iii) % variation** = (Actual − Est)/Est × 100 (actuals 54, 78, 184, 106):
Q1 ≈ **−8.0%** · Q2 ≈ **−20.2%** · Q3 ≈ **−4.2%** · Q4 ≈ **+8.1%**

---

*End of notes. Good luck — you've got this.* 📈
