# IGZO Reservoir Computing for Variable Star Classification

A reservoir computing model for classifying astronomical variable-star light curves using a simulated IGZO (indium gallium zinc oxide) neuromorphic device response.

The project investigates whether the temporal dynamics of an experimentally characterized IGZO device can provide useful reservoir representations for astronomical time-series classification. Performance using the reservoir representation is compared directly against classification of the original processed light curves.

## Variable Star Classes

The model distinguishes between four types of variable stars:

* **CEP** — Cepheid variables
* **EB** — Eclipsing binaries
* **RRAB** — RR Lyrae type AB (fundamental mode)
* **RRC** — RR Lyrae type C (first overtone)

## Project Pipeline

```text
Raw light-curve CSV files
          │
          ▼
        PS.py
          │
          ├── Data cleaning
          ├── Lomb–Scargle period estimation
          ├── Phase folding
          ├── Phase alignment
          ├── Normalization
          └── Stratified train/test split
          │
          ▼
stars_reservoir_dataset.npz
          │
          ▼
        RC.py
          │
          ├───────────────────────┐
          ▼                       ▼
   IGZO Reservoir            Raw Light Curve
          │                       │
          ▼                       ▼
        MLP                     MLP
          │                       │
          └───────────┬───────────┘
                      ▼
             Performance Comparison
```

## Dataset Preparation

`PS.py` converts raw astronomical light-curve CSV files into a balanced dataset suitable for reservoir computing.

For each light curve, the preprocessing pipeline:

1. Loads time and flux measurements
2. Removes invalid observations and extreme measurement outliers
3. Estimates the dominant period using a **Lomb–Scargle periodogram**
4. Phase-folds the time series onto one periodic cycle
5. Resamples the curve to **100 evenly spaced phase points**
6. Aligns the dominant minimum to a common phase reference
7. Normalizes the light curve to the range **[0, 1]**
8. Generates deterministic, stratified training and test indices

By default, the script selects **200 valid light curves per class**, producing a balanced four-class dataset.

### Expected Dataset Structure

The raw data directory should follow the structure:

```text
STARS/
├── CEP_LCs/
│   └── *.csv
├── EB_LCs/
│   └── *.csv
└── RR_LCs/
    ├── RRAB/
    │   └── *.csv
    └── RRC/
        └── *.csv
```

Each light-curve CSV must contain at least:

```text
time, flux
```

## IGZO Reservoir Model

The reservoir is implemented in `RC.py` and is based on experimentally measured IGZO device response parameters, including rise and decay time constants.

Each normalized light curve contains 100 phase points. The input is expanded using **6 masked virtual nodes**, producing a 600-dimensional reservoir representation.

The reservoir state evolves through exponential relaxation dynamics with separate fast and slow components. The resulting current response acts as a nonlinear, history-dependent transformation of the original light curve.

This produces:

```text
Raw input:        100 features
Reservoir input:  600 features
```

## Classification

A one-hidden-layer multilayer perceptron (MLP) is used as the classifier for both experiments:

```text
Input
  │
  ▼
Hidden Layer (30 neurons, tanh)
  │
  ▼
4-Class Output
  │
  ▼
Softmax
```

The same classification approach is applied to:

* the **IGZO reservoir states**, and
* the **original phase-folded light curves**

so that their classification performance can be compared.

The MLP is trained using multiclass cross-entropy loss, Adam optimization, weight decay, and early stopping.

## Model Evaluation

The outer training data are further divided into fit and validation subsets using a stratified split.

Validation accuracy is used for model selection and early stopping. The parameters from the epoch with the highest validation accuracy are restored before final evaluation.

The held-out test set is **not used for model selection** and is evaluated only after training is complete.

Reported metrics include:

* Fit accuracy
* Validation accuracy
* Final test accuracy
* Reservoir vs. no-reservoir test improvement
* Best validation epoch

## Visualizations

`RC.py` generates several visualizations for analyzing the reservoir and classifier:

* **LDA projection** of the reservoir representations
* **Fit and validation accuracy** versus training epoch
* **Fit and validation cross-entropy loss**
* **Test-set confusion matrix**
* Representative **phase-folded light curves**
* Corresponding **IGZO virtual-node reservoir states**

The main result figures are saved as:

```text
reservoir_results_clean_evaluation.png
reservoir_states_by_class.png
```

## Requirements

* Python 3
* NumPy
* SciPy
* Matplotlib

Install the required packages using:

```bash
pip install numpy scipy matplotlib
```

## Running the Project

### 1. Prepare the Dataset

Run `PS.py` first:

```bash
python PS.py
```

By default, this creates:

```text
stars_reservoir_dataset.npz
stars_reservoir_dataset.csv
```

The `.npz` file contains the processed light curves, class labels, source identifiers, and train/test indices.

The companion `.csv` file records the source star, class, and dataset split for each processed sample.

Optional preprocessing parameters can also be specified:

```bash
python PS.py \
    --stars-dir STARS \
    --output stars_reservoir_dataset.npz \
    --per-class 200 \
    --length 100 \
    --seed 42
```

### 2. Run the Reservoir Classification

After generating the dataset, run `RC.py`:

```bash
python RC.py
```

The program will:

1. Generate IGZO reservoir states
2. Train the MLP using the reservoir representation
3. Train an equivalent MLP using the raw processed curves
4. Select each model using validation performance
5. Evaluate the final models on the held-out test set
6. Print the classification results
7. Generate the analysis figures

## Output Files

```text
stars_reservoir_dataset.npz
stars_reservoir_dataset.csv
reservoir_results_clean_evaluation.png
reservoir_states_by_class.png
```

## Research Objective

The goal of this project is to explore the use of experimentally characterized neuromorphic device dynamics for astronomical time-series processing.

By comparing classification using the IGZO reservoir states against classification using the original light curves, the project evaluates whether the device's nonlinear temporal response provides a useful representation for distinguishing different classes of variable stars.
