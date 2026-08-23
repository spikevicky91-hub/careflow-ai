import numpy as np
import pandas as pd


np.random.seed(42)


# -----------------------------
# Configuration
# -----------------------------

DAYS = 365

START_DATE = "2025-01-01"

DEPARTMENTS = [
    "Emergency",
    "General Medicine",
    "Surgery",
    "Cardiology",
    "Orthopedics",
    "Pediatrics",
]


BED_CAPACITY = {
    "Emergency": 25,
    "General Medicine": 120,
    "Surgery": 80,
    "Cardiology": 60,
    "Orthopedics": 70,
    "Pediatrics": 50,
}


# -----------------------------
# Generate daily hospital data
# -----------------------------

dates = pd.date_range(
    start=START_DATE,
    periods=DAYS,
    freq="D",
)


records = []


for date in dates:

    day_of_week = date.dayofweek

    # Weekends generally have slightly different demand
    weekend_factor = 0.90 if day_of_week >= 5 else 1.0

    # Seasonal variation
    seasonal_factor = (
        1
        + 0.10
        * np.sin(
            2 * np.pi * date.dayofyear / 365
        )
    )

    for department in DEPARTMENTS:

        base_demand = {
            "Emergency": 75,
            "General Medicine": 42,
            "Surgery": 25,
            "Cardiology": 18,
            "Orthopedics": 22,
            "Pediatrics": 28,
        }[department]

        demand = (
            base_demand
            * weekend_factor
            * seasonal_factor
        )

        arrivals = max(
            1,
            int(
                np.random.normal(
                    demand,
                    demand * 0.12,
                )
            ),
        )

        admission_rate = {
            "Emergency": 0.28,
            "General Medicine": 0.65,
            "Surgery": 0.72,
            "Cardiology": 0.68,
            "Orthopedics": 0.55,
            "Pediatrics": 0.35,
        }[department]

        admissions = np.random.binomial(
            arrivals,
            admission_rate,
        )

        discharge_rate = {
            "Emergency": 0.20,
            "General Medicine": 0.55,
            "Surgery": 0.60,
            "Cardiology": 0.58,
            "Orthopedics": 0.52,
            "Pediatrics": 0.30,
        }[department]

        discharges = max(
            0,
            int(
                np.random.normal(
                    arrivals * discharge_rate,
                    max(1, arrivals * 0.05),
                )
            ),
        )

        bed_capacity = BED_CAPACITY[
            department
        ]

        occupancy = np.clip(
            np.random.normal(
                0.78
                + (
                    admissions
                    / max(arrivals, 1)
                ) * 0.10,
                0.06,
            ),
            0.45,
            0.98,
        )

        occupied_beds = int(
            bed_capacity * occupancy
        )

        available_beds = (
            bed_capacity
            - occupied_beds
        )

        average_wait = max(
            10,
            np.random.normal(
                45
                + (
                    arrivals
                    / max(base_demand, 1)
                ) * 20,
                8,
            ),
        )

        average_length_of_stay = max(
            1,
            np.random.normal(
                {
                    "Emergency": 0.8,
                    "General Medicine": 4.5,
                    "Surgery": 5.2,
                    "Cardiology": 4.8,
                    "Orthopedics": 4.2,
                    "Pediatrics": 2.8,
                }[department],
                0.6,
            ),
        )

        staff_count = max(
            2,
            int(
                np.random.normal(
                    {
                        "Emergency": 12,
                        "General Medicine": 18,
                        "Surgery": 14,
                        "Cardiology": 10,
                        "Orthopedics": 11,
                        "Pediatrics": 9,
                    }[department],
                    2,
                )
            ),
        )

        records.append(
            {
                "date": date,
                "department": department,
                "patient_arrivals": arrivals,
                "admissions": admissions,
                "discharges": discharges,
                "bed_capacity": bed_capacity,
                "occupied_beds": occupied_beds,
                "available_beds": available_beds,
                "occupancy_rate": round(
                    occupied_beds
                    / bed_capacity,
                    3,
                ),
                "average_wait_minutes": round(
                    average_wait,
                    1,
                ),
                "average_length_of_stay": round(
                    average_length_of_stay,
                    1,
                ),
                "staff_count": staff_count,
            }
        )


df = pd.DataFrame(records)


# -----------------------------
# Save dataset
# -----------------------------

output_path = "data/hospital_operations.csv"

df.to_csv(
    output_path,
    index=False,
)


print(
    f"Generated {len(df):,} hospital operation records."
)

print(
    f"Saved to: {output_path}"
)

print("\nDepartments:")

print(
    df["department"]
    .value_counts()
)


print("\nDataset shape:")

print(df.shape)