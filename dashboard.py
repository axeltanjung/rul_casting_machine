import pathlib

import pandas as pd
import streamlit as st


DATA_PATH = pathlib.Path("data") / "raw" / "ccm_rul_dataset.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Basic cleaning
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep only rows where we have a target RUL value
    if "RUL" in df.columns:
        df = df.dropna(subset=["RUL"])

    return df


def main() -> None:
    st.set_page_config(
        page_title="Casting Machine RUL Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Remaining Useful Life (RUL) Dashboard")
    st.caption(
        "Interactive exploration of predicted RUL for continuous casting machine workpieces."
    )

    if not DATA_PATH.exists():
        st.error(f"Data file not found at `{DATA_PATH}`. Please check the path.")
        return

    df = load_data()

    if df.empty or "RUL" not in df.columns:
        st.error("The dataset does not contain any rows with a valid `RUL` column.")
        return

    # Sidebar controls
    st.sidebar.header("Filters")

    steel_types = sorted(df["steel_type"].dropna().unique()) if "steel_type" in df.columns else []
    selected_steel_types = (
        st.sidebar.multiselect("Steel type", steel_types, default=steel_types)
        if steel_types
        else []
    )

    alloy_types = sorted(df["alloy_type"].dropna().unique()) if "alloy_type" in df.columns else []
    selected_alloy_types = (
        st.sidebar.multiselect("Alloy type", alloy_types, default=alloy_types)
        if alloy_types
        else []
    )

    num_streams = sorted(df["num_stream"].dropna().unique()) if "num_stream" in df.columns else []
    selected_num_streams = (
        st.sidebar.multiselect("Number of streams", num_streams, default=num_streams)
        if num_streams
        else []
    )

    filtered_df = df.copy()
    if selected_steel_types:
        filtered_df = filtered_df[filtered_df["steel_type"].isin(selected_steel_types)]
    if selected_alloy_types:
        filtered_df = filtered_df[filtered_df["alloy_type"].isin(selected_alloy_types)]
    if selected_num_streams:
        filtered_df = filtered_df[filtered_df["num_stream"].isin(selected_num_streams)]

    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Observations", f"{len(filtered_df):,}")
    with col2:
        st.metric("Mean RUL", f"{filtered_df['RUL'].mean():.1f}")
    with col3:
        st.metric("Min RUL", f"{filtered_df['RUL'].min():.1f}")
    with col4:
        st.metric("Max RUL", f"{filtered_df['RUL'].max():.1f}")

    st.markdown("---")

    left, right = st.columns([2, 3])

    with left:
        st.subheader("RUL Distribution")
        st.bar_chart(filtered_df["RUL"])

        if "date" in filtered_df.columns:
            st.subheader("RUL Over Time")
            time_df = filtered_df.sort_values("date").set_index("date")
            st.line_chart(time_df["RUL"])

    with right:
        st.subheader("RUL vs. Key Process Variables")

        numeric_candidates = [
            c
            for c in filtered_df.columns
            if filtered_df[c].dtype != "object" and c not in {"RUL"}
        ]

        if numeric_candidates:
            x_axis = st.selectbox(
                "Select variable for X-axis", numeric_candidates, index=0
            )
            st.scatter_chart(filtered_df[[x_axis, "RUL"]], x=x_axis, y="RUL")
        else:
            st.info(
                "No numeric process variables available for scatter plot besides `RUL`."
            )

    st.markdown("---")
    st.subheader("Raw Data")
    st.dataframe(filtered_df)


if __name__ == "__main__":
    main()


