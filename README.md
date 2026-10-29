# SVD-Based Graph Analytics for Customer Intelligence and Next-Best-Action Prediction

Final year project. SVD on a customer-product matrix, graph analytics (kNN graph and Louvain communities), a rule-based next-best-action engine, and a Streamlit dashboard.

Dataset: Online Retail II (UCI). Download online_retail_II.csv and place it in this folder. It is not included because of its size.

Files: the notebook (full implementation), app.py (dashboard), and the generated CSV outputs used by the dashboard.

To run: pip install -r requirements.txt, then run the notebook top to bottom, then streamlit run app.py

Limitation: validation is offline. A live A/B test would be needed to show the actions change customer behaviour.
