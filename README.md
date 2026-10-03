# AI-Powered Digital Twin for Predictive Maintenance

## Machine Health Scoring, Failure Risk Prediction, Remaining Useful Life Prediction, and Anomaly Detection

This project implements a dataset-driven Digital Twin for predictive maintenance using the NASA C-MAPSS turbofan engine degradation dataset.

The system uses machine-learning models to estimate Remaining Useful Life (RUL), predict near-term failure risk, identify anomalous operating behavior, calculate a Machine Health Score, classify the current machine condition, and generate maintenance recommendations through an interactive Streamlit dashboard.

The complete project is software-based and does not require physical sensors or IoT hardware.

---

## Project Objective

Unexpected equipment failures can cause downtime, maintenance costs, production losses, and safety risks.

The objective of this project is to build an AI-powered Digital Twin that can:

- Monitor machine telemetry
- Estimate Remaining Useful Life
- Predict near-term failure risk
- Detect anomalous sensor behavior
- Calculate a Machine Health Score
- Classify machine condition
- Generate maintenance recommendations
- Visualize machine degradation trends
- Provide an interactive Digital Twin dashboard

---

## Dataset

### NASA C-MAPSS Turbofan Engine Dataset

This project uses the FD001 subset of the NASA C-MAPSS dataset.

Each telemetry record contains:

- Engine ID
- Operating cycle
- 3 operational settings
- 21 anonymized sensor channels

The project uses the following files:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

### Dataset Statistics

```text
Training rows    : 20,631
Testing rows     : 13,096
Training engines : 100
Testing engines  : 100
```

The NASA C-MAPSS sensor channels are anonymized. Therefore, this project keeps their original names such as `sensor_2`, `sensor_3`, and `sensor_4` instead of assigning unsupported physical labels.

The downloaded NASA dataset is not committed to this repository.

Place the required dataset files inside:

```text
dataset/
└── CMAPSSData/
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```

---

## System Architecture

```text
NASA C-MAPSS FD001
        |
        v
Data Loading and Validation
        |
        v
RUL Target Generation
        |
        +-----------------------------+
        |                             |
        v                             v
Random Forest                  Random Forest
RUL Regressor                  Failure Classifier
        |                             |
        +--------------+--------------+
                       |
                       v
                Isolation Forest
                Anomaly Detection
                       |
                       v
               Machine Health Score
                       |
                       v
                 Digital Twin State
                       |
                       v
              Maintenance Alert Logic
                       |
                       v
               Streamlit Dashboard
```

---

## Machine Learning Models

### 1. Remaining Useful Life Prediction

A Random Forest Regressor is used to estimate the number of operating cycles remaining before engine failure.

The training RUL target is calculated as:

```text
RUL = Maximum Cycle of Engine - Current Cycle
```

### Features Used

```text
cycle
setting_1
setting_2
sensor_2
sensor_3
sensor_4
sensor_6
sensor_7
sensor_8
sensor_9
sensor_11
sensor_12
sensor_13
sensor_14
sensor_15
sensor_17
sensor_20
sensor_21
```

### Model Performance

Engine-grouped validation:

```text
MAE  : 23.81 cycles
RMSE : 31.35 cycles
R2   : 0.772
```

NASA FD001 official test evaluation:

```text
MAE  : 19.85 cycles
RMSE : 26.88 cycles
```

### 2. Failure Risk Prediction

A Random Forest Classifier estimates whether the current telemetry resembles an engine operating within its final 30 cycles.

The target is:

```text
Failure Soon = 1 if RUL <= 30
Failure Soon = 0 if RUL > 30
```

Validation performance:

```text
Accuracy  : 95.2%
Precision : 78.6%
Recall    : 94.5%
F1 Score  : 85.8%
```

The displayed failure probability estimates whether current telemetry belongs to the final 30-cycle risk condition. It should not be interpreted as a guaranteed physical probability of immediate failure.

### 3. Anomaly Detection

An Isolation Forest is used for unsupervised anomaly detection.

The model is trained using relatively healthy engine telemetry where:

```text
RUL > 120 cycles
```

Training information:

```text
Healthy baseline rows : 8,531
Contamination setting : 3%
Baseline anomaly rate : 3.00%
```

The model returns:

```text
NORMAL
ANOMALY
```

Decision score interpretation:

```text
Positive score -> More normal behavior
Negative score -> More anomalous behavior
```

The anomaly score is not a probability.

---

## Machine Health Score

The project introduces a Machine Health Score from 0 to 100.

It combines:

```text
70% Failure-Risk Health Component
30% Normalized Predicted-RUL Component
```

Conceptually:

```text
Machine Health Score
=
0.70 x Failure-Risk Health Component
+
0.30 x Normalized RUL Component
```

Higher values indicate a healthier estimated condition. Lower values indicate greater degradation or failure risk.

The Machine Health Score is a project-defined heuristic and is not an official NASA C-MAPSS target variable.

---

## Machine Status Logic

### CRITICAL

Triggered when:

```text
Failure Probability >= 80%
OR
Predicted RUL <= 15 cycles
```

Recommendation:

```text
Immediate maintenance inspection recommended.
```

### HIGH RISK

Triggered when:

```text
Failure Probability >= 60%
OR
Predicted RUL <= 30 cycles
```

Recommendation:

```text
Schedule maintenance as soon as possible.
```

### WARNING

Triggered when:

```text
Failure Probability >= 30%
OR
Predicted RUL <= 60 cycles
```

Recommendation:

```text
Monitor closely and prepare maintenance.
```

### HEALTHY

Used when none of the above risk conditions are triggered.

Recommendation:

```text
No immediate maintenance action required.
```

---

## Streamlit Dashboard

The interactive dashboard provides:

- Engine selection
- Operating-cycle selection
- Machine Health Score
- Failure Probability
- Predicted Remaining Useful Life
- Machine Status
- Anomaly Status
- Isolation Forest Decision Score
- Maintenance Advisory
- Current Sensor Readings
- Machine Health Trend
- Predicted RUL Trend
- Failure Probability Trend
- Individual Sensor Trend
- Recent Alerts
- Model Validation Information
- Technical Digital Twin State
- System Architecture

---

## Example Results

### Healthy Engine

```text
Engine ID              : 1
Operating Cycle        : 31
Predicted RUL          : 170.04 cycles
True RUL               : 112 cycles
Failure Probability    : 0.00%
Machine Health Score   : 95.76%
Machine Status         : HEALTHY
Anomaly Status         : NORMAL
Anomaly Decision Score : 0.1008
```

### Critical Engine

```text
Engine ID            : 34
Operating Cycle      : 203
Predicted RUL        : 7.13 cycles
Failure Probability  : 99.99%
Machine Health Score : 1.09%
Machine Status       : CRITICAL
Anomaly Status       : ANOMALY
```

---

## Project Structure

```text
Digital-Twin-for-Predictive-Maintenance-using-AI-and-Simulation/
│
├── app/
│   └── final_dashboard.py
│
├── dataset/
│   └── CMAPSSData/
│       ├── train_FD001.txt
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
│
├── models/
│   ├── rul_rf_model.joblib
│   ├── failure_risk_model.joblib
│   └── anomaly_model.joblib
│
├── src/
│   ├── data_loader.py
│   ├── train_rul_model.py
│   ├── failure_risk_model.py
│   ├── train_anomaly_model.py
│   ├── digital_twin.py
│   ├── test_risk_states.py
│   └── final_test.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Plotly
- Git
- GitHub

---

## How to Run the Project

### 1. Open the project directory

```powershell
cd Digital-Twin-for-Predictive-Maintenance-using-AI-and-Simulation
```

### 2. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Start the Streamlit dashboard

```powershell
streamlit run app\final_dashboard.py
```

If Streamlit does not open automatically, visit:

```text
http://localhost:8501
```

To stop the application, press `Ctrl + C`.

---

## Model Training

### Train RUL Model

```powershell
python src\train_rul_model.py
```

### Train Failure Risk Model

```powershell
python src\failure_risk_model.py
```

### Train Anomaly Detection Model

```powershell
python src\train_anomaly_model.py
```

---

## Final Project Testing

Run:

```powershell
python src\final_test.py
```

Expected result:

```text
ALL FINAL TESTS PASSED
```

The final test validates dataset loading, RUL calculation, trained model availability, healthy and critical Digital Twin states, and anomaly detection output.

---

## Digital Twin Concept

A traditional monitoring dashboard mainly displays sensor values. This project goes further by maintaining an AI-derived software representation of the machine's current condition.

For every operating cycle, the Digital Twin combines:

```text
Current Telemetry
        +
Predicted Remaining Useful Life
        +
Near-Term Failure Risk
        +
Anomaly Detection
        +
Machine Health Score
        +
Machine Condition
        +
Maintenance Recommendation
```

---

## Academic Scope

This project is an academic software implementation of predictive maintenance and Digital Twin concepts.

Historical NASA C-MAPSS records are replayed as telemetry to simulate machine operation.

The project does not currently use:

- Physical sensors
- Real industrial machinery
- Live IoT devices
- Automatic machine shutdown
- Automatic maintenance execution

---

## Limitations

- Only the NASA C-MAPSS FD001 subset is used.
- Historical telemetry is replayed instead of streamed from real hardware.
- The Machine Health Score is a project-defined heuristic.
- RUL predictions can have larger errors for some early-life states.
- Random Forest models do not explicitly model long temporal sequences.
- Isolation Forest scores are not calibrated probabilities.
- NASA C-MAPSS sensor channels are anonymized.
- Dataset performance does not guarantee identical real-world performance.
- Maintenance recommendations are advisory only.

---

## Future Enhancements

- LSTM or Transformer-based RUL prediction
- Autoencoder-based anomaly detection
- Real-time IoT sensor ingestion
- MQTT or OPC-UA integration
- Database storage for Digital Twin history
- Cloud deployment
- Email or Microsoft Teams alerts
- Automated maintenance-ticket generation
- SHAP-based explainable AI
- Multi-machine fleet monitoring
- Physics-informed Digital Twins

---

## Research Relevance

This project demonstrates concepts from:

- Artificial Intelligence
- Machine Learning
- Predictive Maintenance
- Digital Twins
- Condition Monitoring
- Remaining Useful Life Prediction
- Prognostics
- Anomaly Detection
- Industrial AI
- Industry 4.0
- Maintenance Decision Support

---

## Attribution

This project was developed by extending and substantially modifying the following public GitHub repository as a starting codebase:

https://github.com/srijan300/Digital-Twin-for-Predictive-Maintenance-using-AI-and-Simulation

The final implementation adds or replaces major components including:

- NASA FD001 raw-data loading
- Dataset validation
- Training RUL calculation
- Random Forest RUL prediction
- Random Forest 30-cycle failure-risk classification
- Isolation Forest anomaly detection
- Composite Machine Health Score
- Machine-condition classification
- Maintenance recommendation logic
- Dataset-driven Digital Twin state generation
- Interactive Streamlit dashboard
- Health, RUL, failure-risk, and sensor trend visualization
- Recent alert generation
- Model-validation display
- End-to-end automated testing

The original repository did not contain a `LICENSE` file in the local repository copy used during development. No separate license is claimed here.

---

## Conclusion

This project demonstrates an end-to-end AI-powered Digital Twin for predictive maintenance.

The system transforms historical machine telemetry into maintenance intelligence by combining Remaining Useful Life prediction, failure-risk classification, anomaly detection, Machine Health Scoring, machine-condition classification, maintenance recommendations, and interactive Digital Twin visualization.
