import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------
# Page Settings
# ---------------------------------------

st.set_page_config(
    page_title="City Service Demand Prediction",
    layout="wide"
)


# ---------------------------------------
# Title
# ---------------------------------------

st.title("City Service Demand Prediction")

st.write(
    "Predict the expected number of municipal service requests "
    "for a selected hour."
)


# ---------------------------------------
# Load Dataset
# ---------------------------------------

df = pd.read_csv("service-requests-real-a03a7a.csv")

df["created_date"] = pd.to_datetime(df["created_date"])

# Extract hour
df["hour"] = df["created_date"].dt.hour


# ---------------------------------------
# Calculate Hourly Demand
# ---------------------------------------

hourly_demand = (
    df.groupby("hour")
    .size()
    .reset_index(name="demand")
)

# Make sure all 24 hours are present
all_hours = pd.DataFrame({"hour": range(24)})

hourly_demand = all_hours.merge(
    hourly_demand,
    on="hour",
    how="left"
)

hourly_demand["demand"] = hourly_demand["demand"].fillna(0)


# ---------------------------------------
# Train Model
# ---------------------------------------

X = hourly_demand[["hour"]]
y = hourly_demand["demand"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)


# ---------------------------------------
# Predict Demand for All Hours
# ---------------------------------------

hourly_demand["predicted"] = model.predict(X)


# ---------------------------------------
# User Input
# ---------------------------------------

st.subheader("Predict Service Demand")

selected_hour = st.slider(
    "Select Hour",
    min_value=0,
    max_value=23,
    value=10
)


# ---------------------------------------
# Prediction
# ---------------------------------------

if st.button("Predict Demand"):

    prediction = model.predict(
        [[selected_hour]]
    )[0]

    prediction = round(prediction)

    st.success(
        f"📊 Predicted Service Demand at "
        f"{selected_hour:02d}:00 → {prediction} requests"
    )


# ---------------------------------------
# Find Selected Hour
# ---------------------------------------

selected_row = hourly_demand[
    hourly_demand["hour"] == selected_hour
]

selected_actual = selected_row["demand"].values[0]

selected_prediction = selected_row["predicted"].values[0]


# ---------------------------------------
# Hourly Demand Graph
# ---------------------------------------

st.subheader("Hourly Service Demand")

fig, ax = plt.subplots(figsize=(12, 5))

# Actual demand
ax.plot(
    hourly_demand["hour"],
    hourly_demand["demand"],
    marker="o",
    label="Actual Demand"
)

# Predicted demand
ax.plot(
    hourly_demand["hour"],
    hourly_demand["predicted"],
    marker="x",
    linestyle="--",
    label="Predicted Demand"
)

# Highlight selected hour
ax.scatter(
    selected_hour,
    selected_prediction,
    s=150,
    marker="*",
    label=f"Selected Hour ({selected_hour}:00)"
)

# Vertical line for selected hour
ax.axvline(
    selected_hour,
    linestyle=":",
    alpha=0.7
)

# Labels
ax.set_xlabel("Hour")
ax.set_ylabel("Number of Service Requests")

ax.set_title(
    f"Service Demand at Selected Hour: {selected_hour}:00"
)

ax.set_xticks(range(24))

ax.legend()

ax.grid(True)

st.pyplot(fig)


# ---------------------------------------
# Selected Hour Information
# ---------------------------------------

st.subheader("Selected Hour Details")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Actual Requests",
        int(selected_actual)
    )

with col2:
    st.metric(
        "Predicted Requests",
        round(selected_prediction)
    )


# ---------------------------------------
# Hourly Data
# ---------------------------------------

st.subheader("📋 Hourly Demand Data")

display_data = hourly_demand.copy()

display_data["hour"] = display_data["hour"].apply(
    lambda x: f"{x:02d}:00"
)

display_data["demand"] = display_data["demand"].astype(int)

display_data["predicted"] = display_data["predicted"].round().astype(int)

display_data.columns = [
    "Hour",
    "Actual Demand",
    "Predicted Demand"
]

st.dataframe(
    display_data,
    use_container_width=True
)