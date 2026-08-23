import ollama


MODEL_NAME = "llama3.2:3b"


def get_ai_analysis(
    current_occupancy,
    available_beds,
    capacity_risk,
    forecast_mae,
    forecast_rmse,
    forecast_average,
    staffing_risk,
    staff_gap,
    latest_date,
):
    """
    Generate a business-friendly hospital operations
    analysis using the local Ollama model.
    """

    prompt = f"""
You are a hospital operations analyst.

Analyze the following hospital operational data and
provide a concise business-oriented explanation.

Do NOT diagnose patients.
Do NOT recommend medical treatments.
Focus only on hospital operations, capacity,
staffing pressure, and demand planning.

Operational data:

Latest data date:
{latest_date}

Current bed occupancy:
{current_occupancy:.1f}%

Available beds:
{available_beds}

Current capacity risk:
{capacity_risk}

Average forecasted Emergency Department arrivals:
{forecast_average:.1f} patients/day

Forecast MAE:
{forecast_mae:.2f} patients

Forecast RMSE:
{forecast_rmse:.2f} patients

Staffing pressure:
{staffing_risk}

Current staff gap:
{staff_gap}

Provide the response using exactly these sections:

### Situation

Explain the current operational situation.

### Demand Outlook

Explain what the demand forecast suggests.

### Capacity Impact

Explain what the forecast may mean for hospital capacity.

### Staffing Impact

Explain the staffing situation.

### Recommended Operational Action

Suggest practical operational actions such as
capacity review, discharge planning, scheduling,
or staffing coverage.

### Caveat

Mention that the forecast is an estimate and that
operational decisions should be reviewed by
appropriate hospital personnel.

Keep the response concise and easy for a
non-technical manager to understand.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]