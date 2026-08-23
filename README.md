# 🏥 CareFlow AI

### Hospital Operations & Patient Flow Intelligence

CareFlow AI is a hospital operations analytics project I built to explore how machine learning and local AI can help teams understand patient demand, capacity pressure, and staffing requirements.

The goal is not to replace clinical decision-making, but to provide an operational view of what may need attention.

🌐 **Live Demo:** https://careflow-ai-ltezpynsctncgkzff4nisf.streamlit.app/

## 🚀 What It Does

- 📊 Tracks patient arrivals, admissions, occupancy, wait times, and available beds
- 🔮 Forecasts Emergency Department demand for the next 7 days
- 📈 Evaluates forecast performance using MAE and RMSE
- 🛏️ Projects future bed capacity and potential shortfalls
- 👩‍⚕️ Simulates staffing requirements under different demand scenarios
- 🚨 Identifies departments with higher capacity pressure
- 🤖 Uses a local Llama model through Ollama to explain operational results

## 🧠 How It Works

```text
Hospital Operations Data
          ↓
    Data Preparation
          ↓
   Demand Forecasting
          ↓
   Capacity Analysis
          ↓
 Staffing Simulation
          ↓
 Operational Risk
          ↓
 Local AI Explanation

 The numerical calculations and forecasts are performed by Python and the machine-learning components. The local LLM is used as an explanation layer rather than as the source of the underlying metrics.

🛠️ Tech Stack
Python
Pandas
NumPy
Scikit-learn
Plotly
Streamlit
Ollama
Llama 3.2
Git & GitHub
📁 Project Structure
careflow-ai/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── hospital_operations.csv
│
└── src/
    ├── ai_analyst.py
    ├── forecasting.py
    └── generate_data.py
▶️ Run Locally

Clone the repository:

git clone https://github.com/spikevicky91-hub/careflow-ai.git
cd careflow-ai

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Start the application:

streamlit run app.py
🤖 Local AI Setup

CareFlow AI uses Ollama with Llama 3.2 for the local operations analyst.

Install Ollama and then run:

ollama pull llama3.2:3b

Make sure Ollama is running before using the AI analyst in the dashboard.

📊 Forecasting

The Emergency Department forecasting model uses historical demand patterns and engineered time-series features to predict near-term patient arrivals.

The current model is evaluated using:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)

The dashboard also compares historical demand with model predictions before showing the future 7-day forecast.

🏥 Capacity Intelligence

The capacity model combines:

Forecasted Emergency Department arrivals
Historical admission rates
Historical discharge rates
Current available beds

This produces projected net bed demand, future available capacity, and potential bed shortfalls.

👩‍⚕️ Staffing Simulator

The staffing simulator allows users to explore how changes in:

Expected patient arrivals
Available staff
Patients-per-staff assumptions

could affect staffing pressure.

📸 Dashboard
Hospital Overview

Emergency Demand Forecast

Inventory Intelligence

What-If Simulator

⚠️ Important Note

This project uses a synthetic hospital operations dataset created for demonstration and learning purposes.

CareFlow AI is an operational analytics prototype and is not a clinical decision-support system. Forecasts and recommendations should not be used for real patient-care decisions.

🎯 Why I Built This

I wanted to build a project that connects machine learning predictions to a practical business problem.

Instead of stopping at:

"How many patients might arrive?"

CareFlow AI asks:

"What could that demand mean for hospital capacity and staffing?"

That connection between prediction, simulation, and operational explanation is the main idea behind the project.