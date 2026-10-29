import streamlit as st
import pandas as pd

st.set_page_config(page_title="Customer Intelligence Dashboard", layout="wide")

@st.cache_data
def load():
    return (pd.read_csv("nba_output.csv"), pd.read_csv("product_neighbors.csv"),
            pd.read_csv("segment_products.csv"), pd.read_csv("model_results.csv"))

nba, neigh, segprod, models = load()

st.title("SVD-Based Customer Intelligence & Next-Best-Action")
page = st.sidebar.radio("Page", ["Overview", "Customer analytics", "Product analytics", "Recommendations & actions"])

if page == "Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(nba):,}")
    c2.metric("Segments", nba["segment"].nunique())
    c3.metric("Total spend (training period)", f"£{nba['monetary'].sum():,.0f}")
    st.subheader("Model comparison (hidden 3-month test set)")
    st.dataframe(models.round(4))
    st.bar_chart(models.set_index("model")[["recall_at_10", "ndcg_at_10"]])
    st.subheader("Recommended actions")
    st.bar_chart(nba["action"].value_counts())

elif page == "Customer analytics":
    st.subheader("Customer segments (graph communities)")
    st.bar_chart(nba["segment"].value_counts())
    summary = nba.groupby("segment").agg(
        customers=("Customer ID", "count"), avg_spend=("monetary", "mean"),
        avg_orders=("frequency", "mean"), avg_recency_days=("recency_days", "mean")
    ).round(1).sort_values("customers", ascending=False)
    st.dataframe(summary)
    seg = st.selectbox("Inspect a segment", summary.index)
    st.write("**Typical products:**", ", ".join(segprod[segprod["segment"] == seg]["typical_product"].astype(str)))
    top = nba[nba["segment"] == seg].sort_values("monetary", ascending=False).head(10)
    st.write("**Highest-value customers in this segment**")
    st.dataframe(top[["Customer ID", "monetary", "frequency", "recency_days", "action"]])

elif page == "Product analytics":
    st.subheader("Related products (from SVD product embeddings)")
    prod = st.selectbox("Choose a product", sorted(neigh["product"].astype(str).unique()))
    st.dataframe(neigh[neigh["product"].astype(str) == prod][["related_product", "similarity"]])

else:
    st.subheader("Customer lookup")
    cid = st.selectbox("Customer ID", nba["Customer ID"])
    row = nba[nba["Customer ID"] == cid].iloc[0]
    st.write(f"**Segment:** {row['segment']}  |  **Next-best-action:** {row['action']}")
    st.write(f"Recency: {row['recency_days']} days | Orders: {row['frequency']} | Spend: £{row['monetary']:,.0f}")
    st.write("**Recommended products:**", row["rec_1"], "/", row["rec_2"], "/", row["rec_3"])
    st.subheader("Action list")
    pick = st.selectbox("Filter by action", ["All"] + sorted(nba["action"].unique()))
    view = nba if pick == "All" else nba[nba["action"] == pick]
    st.dataframe(view[["Customer ID", "segment", "action", "rec_1", "rec_2", "rec_3"]])
