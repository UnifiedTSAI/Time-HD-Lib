import os
import re
import pandas as pd
import glob
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extract_time_metrics.log"),
        logging.StreamHandler()
    ]
)

# Regular expressions for data extraction
file_pattern = re.compile(r'(.+?)_(\d+)_(\d+)_(.+?)\.out')

# Time metrics patterns
epoch1_train_cost_pattern = re.compile(r'Epoch: 1 train cost time: ([\d.]+)')
batch_train_cost_pattern = re.compile(r'Avg batch train cost time: ([\d.]+)')
batch_test_cost_pattern = re.compile(r'Avg batch test cost time: ([\d.]+)')

def extract_time_metrics_from_out_file(file_path, model_type):
    """
    Extract time metrics from an .out file
    
    Args:
        file_path: Path to the .out file
        model_type: Model name (for consistency in results)
    """
    logging.info(f"Processing {file_path}...")
    
    # Extract information from filename
    filename = os.path.basename(file_path)
    match = file_pattern.match(filename)
    if not match:
        logging.warning(f"Could not analyze the filename pattern: {filename}")
        return None

    dataset, seq_len, pred_len, model = match.groups()
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Initialize variables as NaN
        epoch1_train_cost_time = np.nan
        batch_train_cost_time = np.nan
        batch_test_cost_time = np.nan
        
        # Extract the last occurrence of each time metric
        epoch1_matches = list(epoch1_train_cost_pattern.finditer(content))
        if epoch1_matches:
            # Take the last match
            last_match = epoch1_matches[-1]
            epoch1_train_cost_time = float(last_match.group(1))
        
        train_matches = list(batch_train_cost_pattern.finditer(content))
        if train_matches:
            # Take the last match
            last_match = train_matches[-1]
            batch_train_cost_time = float(last_match.group(1))
        
        test_matches = list(batch_test_cost_pattern.finditer(content))
        if test_matches:
            # Take the last match
            last_match = test_matches[-1]
            batch_test_cost_time = float(last_match.group(1))
        
        # Return results as a dictionary
        return {
            'dataset': dataset,
            'seq_len': int(seq_len),
            'pred_len': int(pred_len),
            'model': model_type,  # Use the model name from the directory
            'epoch1_train_cost_time': epoch1_train_cost_time,
            'batch_train_cost_time': batch_train_cost_time,
            'batch_test_cost_time': batch_test_cost_time
        }
    
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return None

def main():
    # Find all result directories
    result_dirs = [d for d in os.listdir() if os.path.isdir(d) and re.match(r'\d+_.*_results', d)]
    logging.info(f"Found {len(result_dirs)} result directories: {result_dirs}")
    
    # List to store all results
    all_results = []
    
    # Process each result directory
    for result_dir in result_dirs:
        # Extract the model name from the directory name
        model_name = result_dir.split('_')[1]
        logging.info(f"\nProcessing directory: {result_dir} (model: {model_name})")
        
        # Find all .out files in the directory
        out_files = glob.glob(os.path.join(result_dir, "*.out"))
        logging.info(f"  Found {len(out_files)} .out files")
        
        # Process each .out file
        for out_file in out_files:
            result = extract_time_metrics_from_out_file(out_file, model_name)
            if result:
                all_results.append(result)
                
                # Print the extracted metrics for verification
                logging.info(f"  Dataset: {result['dataset']}, "
                            f"Epoch 1 Train Cost: {result['epoch1_train_cost_time']}, "
                            f"Avg Batch Train Cost: {result['batch_train_cost_time']}, "
                            f"Avg Batch Test Cost: {result['batch_test_cost_time']}")
    
    logging.info(f"\nTime metrics extracted for {len(all_results)} files")
    
    # Check if we have results
    if not all_results:
        logging.warning("No results found to process.")
        return None
    
    # Create a DataFrame with all results
    results_df = pd.DataFrame(all_results)
    
    # Handle multiple results for the same dataset-method combination
    # (take the last result for each combination)
    results_df = results_df.drop_duplicates(subset=['dataset', 'model', 'seq_len', 'pred_len'], keep='last')
    
    # Save the detailed results
    detailed_output_file = 'all_time_metrics_detailed.csv'
    results_df.to_csv(detailed_output_file, index=False)
    logging.info(f"\nDetailed results saved to {detailed_output_file}")
    
    # Create a pivot table format
    # Create a unique identifier for each dataset
    results_df['dataset_id'] = results_df['dataset'] + '_' + results_df['seq_len'].astype(str) + '_' + results_df['pred_len'].astype(str)
    
    # Now create three separate pivot tables, one for each metric
    epoch1_train_df = results_df.pivot(index='dataset_id', columns='model', values='epoch1_train_cost_time')
    batch_train_df = results_df.pivot(index='dataset_id', columns='model', values='batch_train_cost_time')
    batch_test_df = results_df.pivot(index='dataset_id', columns='model', values='batch_test_cost_time')
    
    # Add prefixes to the column names to identify the metric
    epoch1_train_df.columns = [f"{col}_epoch1_train" for col in epoch1_train_df.columns]
    batch_train_df.columns = [f"{col}_batch_train" for col in batch_train_df.columns]
    batch_test_df.columns = [f"{col}_batch_test" for col in batch_test_df.columns]
    
    # Combine the three pivot tables
    final_df = pd.concat([epoch1_train_df, batch_train_df, batch_test_df], axis=1)
    
    # Extract dataset, seq_len, and pred_len back into separate columns for clarity
    dataset_info = results_df[['dataset_id', 'dataset', 'seq_len', 'pred_len']].drop_duplicates()
    dataset_info = dataset_info.set_index('dataset_id')
    
    # Add these columns at the beginning of the final dataframe
    final_df = pd.concat([dataset_info[['dataset', 'seq_len', 'pred_len']], final_df], axis=1)
    
    # Save the pivot table results
    pivot_output_file = 'all_time_metrics_pivot.csv'
    final_df.to_csv(pivot_output_file)
    logging.info(f"\nPivot results saved to {pivot_output_file}")
    
    return results_df

if __name__ == "__main__":
    df = main()
    if df is not None:
        logging.info("\nFirst 5 rows of the detailed result:")
        logging.info(df.head(5).to_string()) 