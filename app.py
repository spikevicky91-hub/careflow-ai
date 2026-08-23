import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.forecasting import train_forecasting_model
from src.ai_analyst import get_ai_analysis

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="CareFlow AI",
    page_icon="🏥",
    layout="wide",
)


# -----------------------------
# Load Data
# -----------------------------

DATA_PATH = "data/hospital_operations.csv"

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

# -----------------------------
# Demand Forecasting
# -----------------------------

forecast_result = train_forecasting_model(df)

forecast_df = forecast_result["forecast"]
evaluation_df = forecast_result["evaluation"]

forecast_mae = forecast_result["mae"]
forecast_rmse = forecast_result["rmse"]

# -----------------------------
# Header
# -----------------------------

st.title("🏥 CareFlow AI")

st.subheader(
    "Hospital Operations & Patient Flow Intelligence"
)

st.write(
    "Monitor hospital demand, capacity, patient flow, "
    "and operational pressure using data and machine learning."
)


# -----------------------------
# Sidebar Filters
# -----------------------------

st.sidebar.header("🔎 Filters")

departments = st.sidebar.multiselect(
    "Department",
    options=sorted(df["department"].unique()),
    default=sorted(df["department"].unique()),
)


filtered_df = df[
    df["department"].isin(departments)
].copy()


# -----------------------------
# Business Overview
# -----------------------------

st.markdown("---")

st.markdown("## 🏥 Hospital Overview")

total_arrivals = filtered_df[
    "patient_arrivals"
].sum()

total_admissions = filtered_df[
    "admissions"
].sum()

average_occupancy = (
    filtered_df["occupancy_rate"].mean()
    * 100
)

average_wait = (
    filtered_df["average_wait_minutes"].mean()
)

average_available_beds = (
    filtered_df["available_beds"].mean()
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Patient Arrivals",
        f"{total_arrivals:,}",
    )


with col2:

    st.metric(
        "Admissions",
        f"{total_admissions:,}",
    )


with col3:

    st.metric(
        "Avg Bed Occupancy",
        f"{average_occupancy:.1f}%",
    )


with col4:

    st.metric(
        "Avg ED Wait",
        f"{average_wait:.0f} min",
    )


with col5:

    st.metric(
        "Avg Available Beds",
        f"{average_available_beds:.0f}",
    )


# -----------------------------
# Patient Demand Trend
# -----------------------------

st.markdown("---")

st.markdown("## 📈 Patient Demand Trend")

daily_demand = (
    filtered_df
    .groupby("date")["patient_arrivals"]
    .sum()
    .reset_index()
)

fig_demand = px.line(
    daily_demand,
    x="date",
    y="patient_arrivals",
    title="Daily Patient Arrivals",
    markers=False,
)

fig_demand.update_layout(
    xaxis_title="Date",
    yaxis_title="Patient Arrivals",
)

st.plotly_chart(
    fig_demand,
    use_container_width=True,
)


# -----------------------------
# Department Performance
# -----------------------------

st.markdown("---")

st.markdown("## 🏥 Department Performance")

department_summary = (
    filtered_df
    .groupby("department")
    .agg(
        patient_arrivals=(
            "patient_arrivals",
            "sum",
        ),
        admissions=(
            "admissions",
            "sum",
        ),
        average_occupancy=(
            "occupancy_rate",
            "mean",
        ),
        average_wait=(
            "average_wait_minutes",
            "mean",
        ),
        average_available_beds=(
            "available_beds",
            "mean",
        ),
    )
    .reset_index()
)

latest_date = filtered_df["date"].max()

latest_data = filtered_df[
    filtered_df["date"] == latest_date
]

department_summary[
    "average_occupancy"
] = (
    department_summary[
        "average_occupancy"
    ] * 100
)


col1, col2 = st.columns(2)


with col1:

    fig_department = px.bar(
        department_summary,
        x="department",
        y="patient_arrivals",
        title="Patient Arrivals by Department",
    )

    fig_department.update_layout(
        xaxis_title="Department",
        yaxis_title="Patient Arrivals",
    )

    st.plotly_chart(
        fig_department,
        use_container_width=True,
    )


with col2:

    fig_occupancy = px.bar(
        department_summary,
        x="department",
        y="average_occupancy",
        title="Average Bed Occupancy",
    )

    fig_occupancy.update_layout(
        xaxis_title="Department",
        yaxis_title="Occupancy (%)",
    )

    st.plotly_chart(
        fig_occupancy,
        use_container_width=True,
    )

# -----------------------------
# Demand Forecasting
# -----------------------------

st.markdown("---")

st.markdown("## 🔮 Emergency Demand Forecast")

st.write(
    "Machine learning forecast of expected "
    "Emergency Department patient arrivals."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "MAE",
        f"{forecast_mae:.2f} patients",
    )

with col2:
    st.metric(
        "RMSE",
        f"{forecast_rmse:.2f} patients",
    )

# -----------------------------
# Model Evaluation Chart
# -----------------------------

evaluation_chart = evaluation_df.copy()

evaluation_chart["date"] = pd.to_datetime(
    evaluation_chart["date"]
)

fig_evaluation = px.line(
    evaluation_chart,
    x="date",
    y=[
        "patient_arrivals",
        "predicted_arrivals",
    ],
    markers=True,
    title="Actual vs Predicted Emergency Arrivals",
)

fig_evaluation.update_layout(
    xaxis_title="Date",
    yaxis_title="Patient Arrivals",
    legend_title="",
)

fig_evaluation.for_each_trace(
    lambda trace: trace.update(
        name=(
            "Actual Arrivals"
            if trace.name == "patient_arrivals"
            else "Predicted Arrivals"
        )
    )
)

st.plotly_chart(
    fig_evaluation,
    use_container_width=True,
)


# -----------------------------
# Future Forecast Chart
# -----------------------------

fig_future = px.line(
    forecast_df,
    x="date",
    y="predicted_arrivals",
    markers=True,
    title="Next 7 Days Emergency Demand Forecast",
)

fig_future.update_layout(
    xaxis_title="Date",
    yaxis_title="Expected Patient Arrivals",
)

st.plotly_chart(
    fig_future,
    use_container_width=True,
)

# -----------------------------
# Forecast Table
# -----------------------------

forecast_display = forecast_df[
    [
        "date",
        "predicted_arrivals",
    ]
].copy()

forecast_display["date"] = (
    pd.to_datetime(
        forecast_display["date"]
    ).dt.strftime("%d %b %Y")
)

forecast_display = forecast_display.rename(
    columns={
        "date": "Date",
        "predicted_arrivals": "Expected Arrivals",
    }
)

st.dataframe(
    forecast_display,
    use_container_width=True,
    hide_index=True,
)


# -----------------------------
# Future Capacity Forecast
# -----------------------------

st.markdown("---")

st.caption(
    "Hospital-wide capacity projection based on "
    "forecasted Emergency Department demand."
)

st.write(
    "Estimated operational pressure based on "
    "forecasted Emergency Department demand."
)


# ---------------------------------
# Historical Emergency Rates
# ---------------------------------

emergency_history = df[
    df["department"] == "Emergency"
].copy()


historical_admission_rate = (
    emergency_history["admissions"].sum()
    / emergency_history["patient_arrivals"].sum()
)


historical_discharge_rate = (
    emergency_history["discharges"].sum()
    / emergency_history["patient_arrivals"].sum()
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Historical Admission Rate",
        f"{historical_admission_rate * 100:.1f}%",
    )


with col2:

    st.metric(
        "Historical Discharge Rate",
        f"{historical_discharge_rate * 100:.1f}%",
    )


# ---------------------------------
# Current Available Capacity
# ---------------------------------

current_available_beds = int(
    latest_data["available_beds"].sum()
)


# ---------------------------------
# Project Future Bed Pressure
# ---------------------------------

future_capacity = forecast_df.copy()


future_capacity["projected_admissions"] = (
    future_capacity["predicted_arrivals"]
    * historical_admission_rate
).round(0)


future_capacity["projected_discharges"] = (
    future_capacity["predicted_arrivals"]
    * historical_discharge_rate
).round(0)


future_capacity["net_bed_demand"] = (
    future_capacity["projected_admissions"]
    - future_capacity["projected_discharges"]
)


future_capacity["cumulative_net_demand"] = (
    future_capacity["net_bed_demand"]
    .cumsum()
)


future_capacity["projected_available_beds"] = (
    current_available_beds
    - future_capacity["cumulative_net_demand"]
)

future_capacity["projected_bed_shortfall"] = (
    future_capacity["projected_available_beds"]
    .clip(upper=0)
    .abs()
)

# ---------------------------------
# Capacity Risk Classification
# ---------------------------------

def classify_future_risk(available_beds):

    if available_beds <= 0:
        return "🔴 Critical"

    elif available_beds <= 10:
        return "🟠 High"

    elif available_beds <= 25:
        return "🟡 Medium"

    else:
        return "🟢 Low"


future_capacity["capacity_risk"] = (
    future_capacity["projected_available_beds"]
    .apply(classify_future_risk)
)


# ---------------------------------
# Display Forecast
# ---------------------------------

future_display = future_capacity[
    [
        "date",
        "predicted_arrivals",
        "projected_admissions",
        "projected_discharges",
        "net_bed_demand",
        "projected_available_beds",
        "projected_bed_shortfall",
        "capacity_risk",
    ]
].copy()


future_display.columns = [
    "Date",
    "Expected ED Arrivals",
    "Projected Admissions",
    "Projected Discharges",
    "Net Bed Demand",
    "Projected Available Beds",
    "Bed Shortfall",
    "Capacity Risk",
]


future_display["Date"] = (
    future_display["Date"]
    .dt.strftime("%d %b %Y")
)


future_display["Projected Available Beds"] = (
    future_display["Projected Available Beds"]
    .round(0)
)


st.dataframe(
    future_display,
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# Future Capacity Warning
# -----------------------------

critical_days = future_capacity[
    future_capacity["capacity_risk"] == "🔴 Critical"
]

high_days = future_capacity[
    future_capacity["capacity_risk"] == "🟠 High"
]


if not critical_days.empty:

    st.error(
        "🚨 Critical capacity pressure is projected "
        "within the next 7 days. Review bed capacity, "
        "discharge planning, and staffing coverage."
    )

elif not high_days.empty:

    st.warning(
        "⚠️ High capacity pressure is projected "
        "during the forecast period."
    )

else:

    st.success(
        "✅ No high or critical capacity pressure "
        "is currently projected."
    )


# -----------------------------
# Staffing Pressure Simulator
# -----------------------------

st.markdown("---")

st.markdown("## 👩‍⚕️ Staffing Pressure Simulator")

st.write(
    "Test how changes in patient demand and staffing "
    "may affect operational pressure."
)


col1, col2, col3 = st.columns(3)


with col1:

    simulated_arrivals = st.number_input(
        "Expected Patient Arrivals",
        min_value=1,
        max_value=300,
        value=int(
            forecast_df["predicted_arrivals"].mean()
        ),
        step=1,
    )


with col2:

    simulated_staff = st.number_input(
        "Available Staff",
        min_value=1,
        max_value=100,
        value=8,
        step=1,
    )


with col3:

    patients_per_staff = st.number_input(
        "Patients per Staff Member",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
    )


# -----------------------------
# Staffing Calculation
# -----------------------------

required_staff = int(
    np.ceil(
        simulated_arrivals
        / patients_per_staff
    )
)

staff_gap = (
    required_staff
    - simulated_staff
)


if staff_gap > 2:

    staffing_risk = "🔴 Critical"

elif staff_gap > 0:

    staffing_risk = "🟠 High"

elif staff_gap == 0:

    staffing_risk = "🟡 Medium"

else:

    staffing_risk = "🟢 Low"


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Required Staff",
        required_staff,
    )


with col2:

    st.metric(
        "Staff Gap",
        max(staff_gap, 0),
    )


with col3:

    st.metric(
        "Staffing Pressure",
        staffing_risk,
    )


if staff_gap > 0:

    st.warning(
        f"⚠️ The scenario requires approximately "
        f"{required_staff} staff members. "
        f"Current staffing is {simulated_staff}."
    )

else:

    st.success(
        "✅ Current staffing is sufficient "
        "for this simulated demand."
    )


# -----------------------------
# Operational Risk
# -----------------------------

st.markdown("---")

st.markdown("## 🚨 Operational Capacity")

latest_date = filtered_df["date"].max()

latest_data = filtered_df[
    filtered_df["date"] == latest_date
]


current_occupancy = (
    latest_data["occupied_beds"].sum()
    / latest_data["bed_capacity"].sum()
    * 100
)

available_beds = (
    latest_data["available_beds"].sum()
)

if current_occupancy >= 90:

    risk = "🔴 Critical"

elif current_occupancy >= 80:

    risk = "🟠 High"

elif current_occupancy >= 70:

    risk = "🟡 Medium"

else:

    risk = "🟢 Low"


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Current Occupancy",
        f"{current_occupancy:.1f}%",
    )


with col2:

    st.metric(
        "Available Beds",
        f"{available_beds}",
    )


with col3:

    st.metric(
        "Capacity Risk",
        risk,
    )


st.caption(
    f"Latest operational data: "
    f"{latest_date.strftime('%d %B %Y')}"
)


# -----------------------------
# Capacity Risk by Department
# -----------------------------

st.markdown("---")

st.markdown("## 🛏️ Capacity Risk by Department")

capacity_df = (
    latest_data[
        [
            "department",
            "bed_capacity",
            "occupied_beds",
            "available_beds",
            "occupancy_rate",
        ]
    ]
    .copy()
)

capacity_df["occupancy_rate"] = (
    capacity_df["occupancy_rate"] * 100
)


def classify_capacity_risk(occupancy):

    if occupancy >= 90:
        return "🔴 Critical"

    elif occupancy >= 80:
        return "🟠 High"

    elif occupancy >= 70:
        return "🟡 Medium"

    else:
        return "🟢 Low"


capacity_df["capacity_risk"] = (
    capacity_df["occupancy_rate"]
    .apply(classify_capacity_risk)
)


capacity_display = capacity_df[
    [
        "department",
        "bed_capacity",
        "occupied_beds",
        "available_beds",
        "occupancy_rate",
        "capacity_risk",
    ]
].copy()


capacity_display.columns = [
    "Department",
    "Bed Capacity",
    "Occupied Beds",
    "Available Beds",
    "Occupancy %",
    "Capacity Risk",
]


capacity_display["Occupancy %"] = (
    capacity_display["Occupancy %"]
    .round(1)
)


st.dataframe(
    capacity_display,
    use_container_width=True,
    hide_index=True,
)


critical_departments = capacity_df[
    capacity_df["capacity_risk"] == "🔴 Critical"
]["department"].tolist()

high_departments = capacity_df[
    capacity_df["capacity_risk"] == "🟠 High"
]["department"].tolist()


if critical_departments:

    st.error(
        "🚨 Immediate capacity pressure in: "
        + ", ".join(critical_departments)
        + "."
    )

elif high_departments:

    st.warning(
        "⚠️ Elevated capacity pressure in: "
        + ", ".join(high_departments)
        + "."
    )

else:

    st.success(
        "✅ No departments are currently classified "
        "as high or critical capacity risk."
    )

# -----------------------------
# AI Hospital Operations Analyst
# -----------------------------

st.markdown("---")

st.markdown("## 🤖 AI Hospital Operations Analyst")

st.write(
    "Use the local Llama model to summarize current "
    "capacity, demand, and staffing pressure."
)


if st.button("🤖 Analyze Hospital Operations"):

    with st.spinner(
        "Analyzing hospital operations..."
    ):

        forecast_average = (
            forecast_df["predicted_arrivals"].mean()
        )

        ai_analysis = get_ai_analysis(
            current_occupancy=current_occupancy,
            available_beds=available_beds,
            capacity_risk=risk,
            forecast_mae=forecast_mae,
            forecast_rmse=forecast_rmse,
            forecast_average=forecast_average,
            staffing_risk=staffing_risk,
            staff_gap=max(staff_gap, 0),
            latest_date=latest_date.strftime(
                "%d %B %Y"
            ),
        )

    st.markdown("### 📋 AI Operations Analysis")

    st.markdown(ai_analysis)