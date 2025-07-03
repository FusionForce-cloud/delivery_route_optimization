import pandas as pd
from sklearn.linear_model import LinearRegression

def predict_demand(df):
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    X = df[['hour']]
    y = df['package_load']
    
    model = LinearRegression()
    model.fit(X, y)
    df['predicted_load'] = model.predict(X).round(2)
    return df
