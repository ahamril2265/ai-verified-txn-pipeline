from pyspark.sql import functions as F, Window
from pyspark.sql.types import LongType

# ---- Load the credit-card transaction data ----
raw_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("/Workspace/Users/ahamril2265@gmail.com/ai-verified-txn-pipeline/Dataset/fraudTrain.csv")
)

# ============================================================
# Step 1a: Build the `customers` feed (one row per cc_num)
# ============================================================
# Profile columns carried from the raw transactions.
profile_cols = [
    "cc_num", "first", "last", "gender", "street", "city",
    "state", "zip", "lat", "long", "job", "dob",
]

customers = (
    raw_df
    .select(*profile_cols)
    .groupBy("cc_num")
    # Use first() to pick a representative profile row for each cc_num.
    .agg(
        *[F.first(c, ignorenulls=True).alias(c)
          for c in profile_cols if c != "cc_num"],
    )
)

# Assign a monotonically increasing surrogate key `customer_id`.
# row_number() gives a stable 1-based ordinal across the result set.

# Helper: generate a sequential customer_id starting at 1
w = Window.orderBy(F.col("cc_num"))
customers = (
    customers
    .withColumn("customer_id", F.row_number().over(w).cast(LongType()))
)

# Reorder columns so customer_id comes first.
customer_cols = ["customer_id"] + profile_cols
customers = customers.select(*customer_cols)

# ============================================================
# Step 1b: Build the `transactions` feed
# ============================================================
# Keep the transaction-level columns and resolve customer_id via cc_num.

transactions = (
    raw_df
    .select(
        F.col("trans_num"),
        F.col("trans_date_trans_time").alias("timestamp"),
        F.col("cc_num"),
        F.col("merchant"),
        F.col("category"),
        F.col("merch_lat"),
        F.col("merch_long"),
        F.col("amt"),
        F.col("is_fraud"),
    )
    # Join to customers to bring in the surrogate customer_id.
    .join(customers.select("cc_num", "customer_id"), on="cc_num", how="inner")
    .drop("cc_num")  # customer_id replaces cc_num in the transactions feed
    .select(
        "trans_num",
        "timestamp",
        "customer_id",
        "merchant",
        "category",
        "merch_lat",
        "merch_long",
        "amt",
        "is_fraud",
    )
)

# Optional: display to verify
print(f"customers  : {customers.count()} rows")
print(f"transactions: {transactions.count()} rows")
display(customers.limit(10))
display(transactions.limit(10))