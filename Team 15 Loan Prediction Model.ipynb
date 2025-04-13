# predict_loans.py
import pandas as pd
import joblib

def main():
    try:
        # Load the trained model
        model_path = r'C:\Users\carlt\Desktop\Model\loan_model.pkl'
        model = joblib.load(model_path)
        
        # Load the new data
        new_data_path = r'C:\Users\carlt\Desktop\Model\LOANAPP_DB - Sheet1.csv'
        new_data = pd.read_csv(new_data_path)
        
        # Define the column mapping from new data to model features
        column_mapping = {
            'No of dependents': 'no_of_dependents',
            'Education': 'education',
            'Employment': 'self_employed',
            'Income': 'income_annum',
            'Loan Amount': 'loan_amount',
            'Loan Term': 'loan_term',
            'Residential Assets Value': 'residential_assets_value',
            'Commercial Assets Value': 'commercial_assets_value',
            'Luxury Assets Value': 'luxury_assets_value',
            'Bank Asset Value': 'bank_asset_value',
            'Credit Score': 'cibil_score'
        }

        # Prepare the new data
        processed_data = new_data.rename(columns=column_mapping)
        
        # Get the EXACT feature order from the trained model
        expected_columns = model.feature_names_in_
        
        # Check for missing columns and handle them
        missing_cols = set(expected_columns) - set(processed_data.columns)
        if missing_cols:
            print(f"Warning: Missing columns in input data: {missing_cols}")
            for col in missing_cols:
                processed_data[col] = 0  # Default value - adjust as needed

        # Ensure columns are in EXACTLY the same order as training data
        processed_data = processed_data[expected_columns]

        # Data preprocessing to match training format
        # Convert education
        processed_data['education'] = processed_data['education'].str.strip().map({
            'Graduate': 1,
            'Not Graduate': 0
        }).fillna(0)  # Default to Not Graduate if not specified
        
        # Convert employment status
        processed_data['self_employed'] = processed_data['self_employed'].str.strip().map({
            'Yes': 1,
            'No': 0
        }).fillna(0)  # Default to No if not specified

        # Make predictions
        predictions = model.predict(processed_data)
        prediction_probs = model.predict_proba(processed_data)[:, 1]  # Probability of approval (class 1)

        # Add predictions back to original data
        new_data['Predicted_Approval'] = ['Approved' if x == 1 else 'Rejected' for x in predictions]
        new_data['Approval_Probability'] = prediction_probs
        
        # Convert probability to percentage for better readability
        new_data['Approval_Probability_Pct'] = (prediction_probs * 100).round(2)

        # Save results
        output_path = r'C:\Users\carlt\Desktop\Model\LOANAPP_Predictions.csv'
        new_data.to_csv(output_path, index=False)

        print(f"\nSuccess! Predictions saved to {output_path}")
        print("\nPrediction Summary:")
        print(f"Total applications processed: {len(new_data)}")
        print(f"Approved: {sum(predictions)} ({sum(predictions)/len(new_data):.1%})")
        print(f"Rejected: {len(new_data) - sum(predictions)} ({(len(new_data) - sum(predictions))/len(new_data):.1%})")
        
        return True
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        return False

if __name__ == "__main__":
    main()