# CHURN / AI
Predict risk. Understand behavior. Retain intelligently.

Interactive customer-retention intelligence: churn prediction (logistic regression, random forest, gradient boosting with calibration), behavioural patterns, segmentation, individual explanations, retention strategy studio, ROI simulator and a tool-using agent.

## Run
Google Colab: paste `app.py` into one cell and run it.
Locally: `pip install -r requirements.txt && python app.py`

## Data
`data/sample_subscription_data.csv` is **fictional synthetic data**. Upload your own CSV in the app (pseudonymous IDs recommended). Never commit real customer data or API keys.

## Notes
Predictions show association, not causation. Simulator effects are hypothetical until validated with A/B tests. Optional LLM key: set `ANTHROPIC_API_KEY` as an environment variable or Colab secret.

## Stack
Python, Gradio, Plotly, scikit-learn, SciPy, pandas
