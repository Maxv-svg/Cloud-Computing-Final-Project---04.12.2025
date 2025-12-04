import boto3
import pandas as pd
import numpy as np
import joblib
import os
import io

s3 = boto3.client('s3')
BUCKET_NAME = 'favorita-project-data' 

def lambda_handler(event, context):
    print("Event received:", event)
    
    # 1. Get Uploaded File Details
    try:
        input_bucket = event['Records'][0]['s3']['bucket']['name']
        input_key = event['Records'][0]['s3']['object']['key']
    except Exception as e:
        return {"status": "Error", "message": f"Bad Event format: {e}"}

    print(f"Processing s3://{input_bucket}/{input_key}")

    # 2. Download Model (Cache in /tmp)
    model_path = '/tmp/sales_forecast_pipeline.pkl'
    if not os.path.exists(model_path):
        print("Downloading model from S3...")
        s3.download_file(BUCKET_NAME, 'models/sales_forecast_pipeline.pkl', model_path)
    
    model = joblib.load(model_path)

    # 3. Download & Read Input CSV
    obj = s3.get_object(Bucket=input_bucket, Key=input_key)
    input_df = pd.read_csv(obj['Body'])

    # 4. Feature Engineering
    required_cols = ['store_nbr', 'item_nbr', 'dcoilwtico', 'transactions', 'onpromotion']
    for col in required_cols:
        if col not in input_df.columns:
            input_df[col] = 0 
            
    X = input_df[required_cols].fillna(0)
    
    # 5. Predict
    print("Generating predictions...")
    preds_log = model.predict(X)
    preds = np.expm1(preds_log) 
    
    # 6. Save Results
    input_df['predicted_sales'] = preds
    
    csv_buffer = io.StringIO()
    input_df.to_csv(csv_buffer, index=False)
    
    # Save to 'predictions/' folder
    filename_only = os.path.basename(input_key).replace('.7z', '').replace('.zip', '').replace('.csv', '')
    output_key = f"predictions/{filename_only}_results.csv"
    
    s3.put_object(Bucket=BUCKET_NAME, Key=output_key, Body=csv_buffer.getvalue())
    print(f"Saved predictions to s3://{BUCKET_NAME}/{output_key}")
    
    return {"status": "Success", "output_file": output_key}