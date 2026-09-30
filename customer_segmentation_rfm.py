import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


print("=" * 70)
print("CUSTOMER SEGMENTATION - RFM ANALYSIS")
print("=" * 70)


# ============================================================
# PART 1 - INTRODUCTION TO RFM
# ============================================================

print("\nPART 1 - INTRODUCTION TO RFM")
print("=" * 70)

print("\nQuestion 1:")
print(
    "A customer who purchased once three years ago with a very large "
    "amount is different from a customer who purchases every month "
    "with smaller amounts. RFM helps identify these differences by "
    "looking at recency, frequency, and monetary value. This allows "
    "the business to create different strategies for valuable active "
    "customers and inactive customers."
)

print("\nQuestion 2:")
print(
    "A customer with high recency and low frequency is still active "
    "but does not purchase often, so the business could send a "
    "personalized offer or reminder to encourage another purchase. "
    "A customer with very high recency and high historical frequency "
    "may be at risk of being lost, so the business could send a "
    "win-back campaign or a special discount to bring the customer back."
)


# ============================================================
# PART 2 - PREPARE CUSTOMER ORDER DATA
# ============================================================

print("\n" + "=" * 70)
print("PART 2 - RFM CALCULATION")
print("=" * 70)


# Create realistic sample e-commerce order data
np.random.seed(42)

number_of_orders = 1000
number_of_customers = 200

customer_ids = np.random.randint(
    1,
    number_of_customers + 1,
    number_of_orders
)

start_date = pd.Timestamp("2025-01-01")
end_date = pd.Timestamp("2025-12-31")

date_range = pd.date_range(
    start=start_date,
    end=end_date,
    freq="D"
)

order_dates = np.random.choice(
    date_range,
    size=number_of_orders
)

revenues = np.round(
    np.random.uniform(20, 500, number_of_orders),
    2
)

orders = pd.DataFrame({
    "customer_id": customer_ids,
    "order_date": order_dates,
    "revenue": revenues
})

orders["order_date"] = pd.to_datetime(orders["order_date"])

print("\nFirst 10 orders:")
print(orders.head(10))

print("\nDataset shape:")
print(orders.shape)


# ============================================================
# RFM CALCULATION
# ============================================================

analysis_date = orders["order_date"].max()

rfm = orders.groupby("customer_id").agg(
    recency=("order_date", lambda x:
             (analysis_date - x.max()).days),
    frequency=("order_date", "count"),
    monetary=("revenue", "sum")
).reset_index()

print("\nRFM table - first 10 customers:")
print(rfm.head(10))

print("\nRFM shape:")
print(rfm.shape)


# ============================================================
# PART 2 - STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZING RFM VARIABLES")
print("=" * 70)

rfm_values = rfm[[
    "recency",
    "frequency",
    "monetary"
]]

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_values)

print("\nStandardized RFM data - first 5 rows:")
print(rfm_scaled[:5])
# ============================================================
# ELBOW METHOD
# ============================================================

print("\n" + "=" * 70)
print("ELBOW METHOD - FINDING THE BEST K")
print("=" * 70)

inertia_values = []

for k in range(2, 8):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(rfm_scaled)

    inertia_values.append(kmeans.inertia_)

    print(f"K = {k}, Inertia = {kmeans.inertia_:.2f}")


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 8),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for Customer Segmentation")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(2, 8))
plt.grid(True)

plt.tight_layout()
plt.show()
# ============================================================
# FINAL K-MEANS MODEL
# ============================================================

print("\n" + "=" * 70)
print("FINAL K-MEANS CUSTOMER SEGMENTATION")
print("=" * 70)

best_k = 4

final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

rfm["segment"] = final_kmeans.fit_predict(rfm_scaled)

print("\nRFM table with segments - first 10 customers:")
print(rfm.head(10))

print("\nNumber of customers in each segment:")
print(rfm["segment"].value_counts().sort_index())
# ============================================================
# PART 3 - INTERPRETING CUSTOMER SEGMENTS
# ============================================================

print("\n" + "=" * 70)
print("PART 3 - SEGMENT INTERPRETATION")
print("=" * 70)

segment_summary = rfm.groupby("segment")[
    ["recency", "frequency", "monetary"]
].mean().round(2)

print("\nAverage RFM values by segment:")
print(segment_summary)


# ============================================================
# BUSINESS SEGMENT NAMES
# ============================================================

segment_names = {}

for segment in segment_summary.index:

    recency = segment_summary.loc[segment, "recency"]
    frequency = segment_summary.loc[segment, "frequency"]
    monetary = segment_summary.loc[segment, "monetary"]

    if (
        recency <= segment_summary["recency"].median()
        and frequency >= segment_summary["frequency"].median()
        and monetary >= segment_summary["monetary"].median()
    ):
        name = "Champions"

    elif (
        recency > segment_summary["recency"].median()
        and frequency >= segment_summary["frequency"].median()
        and monetary >= segment_summary["monetary"].median()
    ):
        name = "At Risk"

    elif (
        recency <= segment_summary["recency"].median()
        and frequency < segment_summary["frequency"].median()
    ):
        name = "New Customers"

    else:
        name = "Regular Customers"

    segment_names[segment] = name


rfm["segment_name"] = rfm["segment"].map(segment_names)

print("\nSegment names:")
for segment, name in segment_names.items():
    print(f"Segment {segment} -> {name}")


# ============================================================
# SEGMENT DESCRIPTIONS AND RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS DESCRIPTION AND MARKETING ACTION")
print("=" * 70)

for segment, name in segment_names.items():

    segment_data = segment_summary.loc[segment]

    print(f"\nSegment {segment}: {name}")
    print(
        f"Average Recency: {segment_data['recency']:.2f} days"
    )
    print(
        f"Average Frequency: {segment_data['frequency']:.2f} orders"
    )
    print(
        f"Average Monetary: ${segment_data['monetary']:.2f}"
    )

    if name == "Champions":
        print(
            "Description: These customers purchase relatively recently, "
            "purchase frequently, and generate high revenue."
        )
        print(
            "Marketing Action: Reward them with loyalty benefits, "
            "exclusive offers, or early access to new products."
        )

    elif name == "At Risk":
        print(
            "Description: These customers have been inactive recently "
            "but previously purchased frequently and generated strong revenue."
        )
        print(
            "Marketing Action: Send personalized win-back offers "
            "or discounts to encourage them to return."
        )

    elif name == "New Customers":
        print(
            "Description: These customers purchased recently but have "
            "relatively low purchase frequency."
        )
        print(
            "Marketing Action: Use welcome offers and personalized "
            "recommendations to encourage a second purchase."
        )

    else:
        print(
            "Description: These customers show moderate purchasing "
            "activity and represent a regular customer base."
        )
        print(
            "Marketing Action: Use targeted promotions and product "
            "recommendations to increase purchase frequency and value."
        )
        # ============================================================
# PART 4 - FINAL CUSTOMER SEGMENTATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART 4 - FINAL CUSTOMER SEGMENTATION ANALYSIS")
print("=" * 70)


# ============================================================
# CUSTOMER COUNT AND PERCENTAGE
# ============================================================

segment_counts = rfm["segment_name"].value_counts()

segment_percentage = (
    segment_counts / len(rfm) * 100
).round(2)

segment_table = pd.DataFrame({
    "segment_name": segment_counts.index,
    "customer_count": segment_counts.values,
    "percentage_of_customers": segment_percentage.values
})

print("\nCustomer Segment Summary:")
print(segment_table)


# ============================================================
# REVENUE BY SEGMENT
# ============================================================

revenue_by_segment = rfm.groupby(
    "segment_name"
)["monetary"].sum().sort_values(ascending=False)

revenue_percentage = (
    revenue_by_segment / revenue_by_segment.sum() * 100
).round(2)

revenue_table = pd.DataFrame({
    "total_revenue": revenue_by_segment,
    "revenue_percentage": revenue_percentage
})

print("\nRevenue by Segment:")
print(revenue_table)

largest_revenue_segment = revenue_by_segment.idxmax()

print("\nSegment with the largest share of total revenue:")
print(largest_revenue_segment)

print(
    f"Revenue share: {revenue_percentage[largest_revenue_segment]:.2f}%"
)


# ============================================================
# SCATTER PLOT
# ============================================================

plt.figure(figsize=(10, 6))

for segment_name in rfm["segment_name"].unique():

    segment_data = rfm[
        rfm["segment_name"] == segment_name
    ]

    plt.scatter(
        segment_data["frequency"],
        segment_data["monetary"],
        label=segment_name,
        alpha=0.7
    )

plt.title("Customer Segmentation: Frequency vs Monetary")
plt.xlabel("Frequency - Number of Orders")
plt.ylabel("Monetary - Total Revenue")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXECUTIVE SUMMARY")
print("=" * 70)

executive_summary_1 = (
    "The RFM analysis divided customers into four business segments: "
    "At Risk, New Customers, Champions, and Regular Customers. "
    "Champions are the most valuable active customers because they "
    "have relatively low recency, high purchase frequency, and high "
    "monetary value. At Risk customers have relatively high purchase "
    "frequency and monetary value but have not purchased recently. "
    "New Customers have more recent activity but lower purchase "
    "frequency and spending, while Regular Customers show lower "
    "overall purchasing activity and monetary value."
)

executive_summary_2 = (
    f"The most important business finding is that the At Risk segment "
    f"generates the largest share of total revenue, accounting for "
    f"{revenue_percentage['At Risk']:.2f}% of total revenue, despite "
    f"representing only {segment_percentage['At Risk']:.2f}% of customers. "
    "This indicates that losing these customers could have a meaningful "
    "impact on business revenue. The main recommended action is to "
    "launch a personalized win-back campaign for At Risk customers, "
    "using targeted discounts, special offers, or loyalty incentives "
    "to encourage them to purchase again. Champions should also receive "
    "loyalty rewards and exclusive benefits to maintain their high value."
)

print("\nParagraph 1:")
print(executive_summary_1)

print("\nParagraph 2:")
print(executive_summary_2)


# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)

print("\nRFM metrics calculated successfully.")
print("StandardScaler applied successfully.")
print("Elbow Method completed for K = 2 to 7.")
print(f"Final K-Means model completed with K = {best_k}.")
print("Customer segments interpreted and named.")
print("Scatter Plot created.")
print("Customer and revenue summaries created.")
print("Executive summary completed.")