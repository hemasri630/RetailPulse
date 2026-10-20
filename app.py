import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


@st.cache_data
def load_small_csv(filename):
    return pd.read_csv(os.path.join(DATA_DIR, filename))


@st.cache_data
def load_sales():
    path = os.path.join(DATA_DIR, "sales_transactions.csv")

    df = pd.read_csv(
        path,
        usecols=[
            "date",
            "receipt_id",
            "store_id",
            "sku_id",
            "customer_id",
            "quantity",
            "unit_price",
            "total_value",
            "channel",
        ],
        dtype={
            "receipt_id": "category",
            "store_id": "category",
            "sku_id": "category",
            "customer_id": "category",
            "channel": "category",
        },
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["quantity"] = pd.to_numeric(
        df["quantity"], errors="coerce"
    ).fillna(0)
    df["total_value"] = pd.to_numeric(
        df["total_value"], errors="coerce"
    ).fillna(0)

    return df.dropna(subset=["date"])


st.title("📊 RetailPulse")
st.caption("AI-Powered Customer Analytics & Demand Forecasting Platform")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Sales Analytics",
        "Customer Segmentation",
        "Customer Churn",
        "Inventory Optimization",
        "Product Analytics",
        "Demand Forecast",
        "Settings",
    ],
)

st.sidebar.divider()

try:
    sales = load_sales()
except Exception as e:
    st.error(f"Unable to load sales data: {e}")
    st.stop()

try:
    stores = load_small_csv("store_master.csv")
    products = load_small_csv("sku_master.csv")
    customers = load_small_csv("customer_master.csv")
    inventory = load_small_csv("inventory_snapshot.csv")
except Exception as e:
    st.error(f"Unable to load reference data: {e}")
    st.stop()

sales["store_id"] = sales["store_id"].astype(str)
sales["sku_id"] = sales["sku_id"].astype(str)

products["sku_id"] = products["sku_id"].astype(str)
stores["store_id"] = stores["store_id"].astype(str)

sales = sales.merge(
    products[["sku_id", "sku_name", "category", "cost_price"]],
    on="sku_id",
    how="left",
)

sales = sales.merge(
    stores[["store_id", "store_name"]],
    on="store_id",
    how="left",
)

st.sidebar.subheader("Global Filters")

min_date = sales["date"].min().date()
max_date = sales["date"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

store_options = sorted(sales["store_name"].dropna().astype(str).unique())
selected_stores = st.sidebar.multiselect(
    "Stores",
    store_options,
    default=store_options,
)

channel_options = sorted(sales["channel"].dropna().astype(str).unique())
selected_channels = st.sidebar.multiselect(
    "Sales Channels",
    channel_options,
    default=channel_options,
)

category_options = sorted(sales["category"].dropna().astype(str).unique())
selected_categories = st.sidebar.multiselect(
    "Product Categories",
    category_options,
    default=category_options,
)

filtered = sales[
    (sales["date"].dt.date >= start_date)
    & (sales["date"].dt.date <= end_date)
    & (sales["store_name"].astype(str).isin(selected_stores))
    & (sales["channel"].astype(str).isin(selected_channels))
    & (sales["category"].astype(str).isin(selected_categories))
].copy()

if page == "Executive Dashboard":
    st.header("Executive Dashboard")

    revenue = filtered["total_value"].sum()
    transactions = filtered["receipt_id"].nunique()
    units = filtered["quantity"].sum()
    customers_count = filtered["customer_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"₹{revenue:,.2f}")
    c2.metric("Transactions", f"{transactions:,}")
    c3.metric("Units Sold", f"{units:,.0f}")
    c4.metric("Customers", f"{customers_count:,}")

    st.subheader("Daily Sales Trend")
    daily = filtered.groupby(filtered["date"].dt.date)["total_value"].sum()
    st.line_chart(daily)

    left, right = st.columns(2)

    with left:
        st.subheader("Revenue by Store")
        store_sales = (
            filtered.groupby("store_name")["total_value"]
            .sum()
            .sort_values(ascending=False)
        )
        st.bar_chart(store_sales)

    with right:
        st.subheader("Revenue by Channel")
        channel_sales = filtered.groupby("channel")["total_value"].sum()
        st.bar_chart(channel_sales)

elif page == "Sales Analytics":
    st.header("Sales Analytics")

    c1, c2 = st.columns(2)
    c1.metric("Revenue", f"₹{filtered['total_value'].sum():,.2f}")
    c2.metric(
        "Average Transaction Value",
        f"₹{filtered.groupby('receipt_id')['total_value'].sum().mean():,.2f}"
        if not filtered.empty
        else "₹0.00",
    )

    st.subheader("Sales Over Time")
    daily = filtered.groupby(filtered["date"].dt.date)["total_value"].sum()
    st.line_chart(daily)

    st.subheader("Sales by Store and Channel")
    st.dataframe(
        filtered.groupby(
            ["store_name", "channel"], observed=True
        )
        .agg(
            Revenue=("total_value", "sum"),
            Units=("quantity", "sum"),
        )
        .reset_index(),
        use_container_width=True,
    )

elif page == "Customer Segmentation":
    st.header("Customer Segmentation")

    customer_sales = (
        filtered.groupby("customer_id", observed=True)
        .agg(
            Total_Spending=("total_value", "sum"),
            Total_Units=("quantity", "sum"),
            Transactions=("receipt_id", "nunique"),
            Last_Purchase=("date", "max"),
        )
        .reset_index()
    )

    if not customer_sales.empty:
        spending_median = customer_sales["Total_Spending"].median()
        frequency_median = customer_sales["Transactions"].median()

        customer_sales["Segment"] = np.where(
            (customer_sales["Total_Spending"] >= spending_median)
            & (customer_sales["Transactions"] >= frequency_median),
            "High Value",
            np.where(
                customer_sales["Total_Spending"] >= spending_median,
                "Big Spenders",
                np.where(
                    customer_sales["Transactions"] >= frequency_median,
                    "Frequent Customers",
                    "Regular Customers",
                ),
            ),
        )

        st.subheader("Customer Segment Distribution")
        st.bar_chart(customer_sales["Segment"].value_counts())
        st.dataframe(customer_sales, use_container_width=True)
    else:
        st.info("No customer data available for the selected filters.")

elif page == "Customer Churn":
    st.header("Customer Churn Analysis")

    st.info(
        "These are rule-based inactivity indicators, not predictions "
        "from a trained machine-learning model."
    )

    customer_activity = (
        filtered.groupby("customer_id", observed=True)
        .agg(
            Last_Purchase=("date", "max"),
            Total_Spending=("total_value", "sum"),
            Transactions=("receipt_id", "nunique"),
        )
        .reset_index()
    )

    if not customer_activity.empty:
        reference_date = filtered["date"].max()
        customer_activity["Days_Inactive"] = (
            reference_date - customer_activity["Last_Purchase"]
        ).dt.days

        customer_activity["Status"] = np.where(
            customer_activity["Days_Inactive"] >= 90,
            "At Risk",
            "Active",
        )

        st.metric(
            "Customers at Risk",
            int((customer_activity["Status"] == "At Risk").sum()),
        )

        st.bar_chart(customer_activity["Status"].value_counts())
        st.dataframe(
            customer_activity.sort_values(
                "Days_Inactive", ascending=False
            ),
            use_container_width=True,
        )
    else:
        st.info("No customer activity available.")

elif page == "Inventory Optimization":
    st.header("Inventory Optimization")

    inv = inventory.merge(
        products[["sku_id", "sku_name", "category"]],
        on="sku_id",
        how="left",
    )

    inv["Stock_Status"] = np.select(
        [
            inv["stock_on_hand"] <= inv["reorder_point"],
            inv["stock_on_hand"] <= inv["safety_stock"],
        ],
        ["Reorder Needed", "Low Stock"],
        default="Stock Available",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Inventory Records", len(inv))
    c2.metric(
        "Reorder Needed",
        int((inv["Stock_Status"] == "Reorder Needed").sum()),
    )
    c3.metric(
        "Total Stock Units",
        f"{inv['stock_on_hand'].sum():,.0f}",
    )

    st.subheader("Inventory Status")
    st.bar_chart(inv["Stock_Status"].value_counts())
    st.subheader("Inventory Details")
    st.dataframe(inv, use_container_width=True)

elif page == "Product Analytics":
    st.header("Product Analytics")

    product_sales = (
        filtered.groupby(
            ["sku_id", "sku_name", "category"], observed=True
        )
        .agg(
            Revenue=("total_value", "sum"),
            Units_Sold=("quantity", "sum"),
            Cost=("cost_price", "first"),
        )
        .reset_index()
    )

    product_sales["Estimated Profit"] = (
        product_sales["Revenue"]
        - product_sales["Units_Sold"] * product_sales["Cost"]
    )

    st.subheader("Top Products by Revenue")
    top_products = product_sales.nlargest(10, "Revenue")
    st.bar_chart(top_products.set_index("sku_name")["Revenue"])

    st.subheader("Product Performance")
    st.dataframe(
        product_sales.sort_values("Revenue", ascending=False),
        use_container_width=True,
    )

elif page == "Demand Forecast":
    st.header("Demand Forecast")

    st.info(
        "This is a simple baseline forecast using recent average daily sales. "
        "It is not a trained forecasting model."
    )

    forecast_days = st.slider(
        "Forecast Days",
        min_value=7,
        max_value=30,
        value=7,
    )

    daily_units = (
        filtered.groupby(filtered["date"].dt.date)["quantity"].sum().sort_index()
    )

    if len(daily_units) >= 2:
        recent_average = daily_units.tail(7).mean()
        future_dates = pd.date_range(
            start=filtered["date"].max().normalize() + pd.Timedelta(days=1),
            periods=forecast_days,
        )

        forecast = pd.DataFrame(
            {
                "Date": future_dates,
                "Forecast Units": [recent_average] * forecast_days,
            }
        )

        st.metric("Average Daily Demand", f"{recent_average:,.2f} units")
        st.line_chart(forecast.set_index("Date")["Forecast Units"])
        st.dataframe(forecast, use_container_width=True)
    else:
        st.warning("At least two days of sales data are needed to forecast.")

elif page == "Settings":
    st.header("Dashboard Settings")

    st.write("**Project:** RetailPulse")
    st.write("**Platform:** Streamlit")
    st.write("**Data source:** Local CSV files")

    st.write("**Available data files**")
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith(".csv"):
            st.write(f"📄 {filename}")

    if st.button("Clear Cached Data"):
        st.cache_data.clear()
        st.success("Cached data cleared. Refresh the dashboard.")

    st.sidebar.divider()
    st.sidebar.caption("RetailPulse | Retail Intelligence")
