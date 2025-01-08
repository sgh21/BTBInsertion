import pandas as pd

def calculate_3sigma_mean(csv_file, column_name):
    """
    Reads a specified column from a CSV file and calculates its 3-sigma mean.
    
    Args:
        csv_file (str): Path to the CSV file.
        column_name (str): Name of the column to calculate the 3-sigma mean for.
    
    Returns:
        float: The 3-sigma mean of the specified column.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(csv_file)
        
        # Check if the column exists
        if column_name not in data.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")
        
        # Extract the specified column
        column_data = data[column_name]
        
        # Ensure the data is numeric and drop NaN values
        column_data = pd.to_numeric(column_data, errors='coerce').dropna()
        
        # Calculate the mean and standard deviation
        mean = column_data.mean()
        std_dev = column_data.std()
        
        # Calculate the 3-sigma range
        lower_bound = mean - 3 * std_dev
        upper_bound = mean + 3 * std_dev
        
        # Filter the data within the 3-sigma range
        filtered_data = column_data[(column_data >= lower_bound) & (column_data <= upper_bound)]
        
        # Calculate the mean of the filtered data
        three_sigma_mean = filtered_data.mean()
        
        return three_sigma_mean,upper_bound,std_dev
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{csv_file}' not found.")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

# Example usage (replace with actual file path and column name):
from config import PARAMS
import os
data_dir = PARAMS['experiment_dir']
csv_file = os.path.join(data_dir, 'exp_generalization_tranditional_vision_0104','vision_experiment_4040P.csv')
print(csv_file)


result, upper_bound, sigma = calculate_3sigma_mean(csv_file, 'abs_error_x_mm')
print(f"error_x_mm - Mean: {result:.4f}, Upper bound: {upper_bound:.4f}, Sigma: {sigma:.4f}")

result, upper_bound, sigma = calculate_3sigma_mean(csv_file, 'abs_error_y_mm')
print(f"error_y_mm - Mean: {result:.4f}, Upper bound: {upper_bound:.4f}, Sigma: {sigma:.4f}")

result, upper_bound, sigma = calculate_3sigma_mean(csv_file, 'abs_error_angle_deg')
print(f"error_angle_deg - Mean: {result:.4f}, Upper bound: {upper_bound:.4f}, Sigma: {sigma:.4f}")