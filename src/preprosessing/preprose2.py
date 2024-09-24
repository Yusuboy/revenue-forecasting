import pandas as pd
import time

file_path = '2024-09-23 Liikevaihto_data.csv'

def add_calendar_columns(df):
    
    df['Trans Date'] = pd.to_datetime(df['Trans Date'])
    df['Year'] = df['Trans Date'].dt.year  # Extract as integer year
    df['Month'] = df['Trans Date'].dt.month  # Extract as integer month (1-12)
    
    return df

def encode_location(trx_general_ledger):
    if trx_general_ledger.startswith('EU'):
        # See later if we want to split Finland to two locations: the mother company and subsidiary
        return 1
    elif trx_general_ledger.startswith('IN'):
        # See later if we want to evaluate Indian locations separately
        return 2
    else:
        # See later if we want to split leftovers as well (f.ex. Poland vs. others)
        return 3

def add_location_column(df):

    df['Location'] = df['Trx- General Ledger Unit'].apply(encode_location)
    return df

# Find the transactions that represent the Indian revenue transferred from the
# offshore project to the local project. 
# TODO: check that the negative transactions in 030 need to be dropped 
# (or are there matching positive transactions for individual members)
def drop_controller_transfers(df):
          
    df['Resource Amount'] = df['Resource Amount'].astype(str).str.replace(',', '').str.strip()
    df['Resource Amount'] = pd.to_numeric(df['Resource Amount'], errors='coerce')

    negative_glr_df = df[(df['An Type'] == 'GLR') & (df['Resource Amount'] < 0) & (df['Business Unit'] == 'EU030') & (df['Member ID'] == 'MBR-20000')]
    negative_glr_indices = negative_glr_df.index
    indices_to_drop = []    

    # Initialize an empty dataframe to store the cleaned (dropped) rows
    controller_transfers = pd.DataFrame()
    multiple_matches = 0

    for idx in negative_glr_indices:
        # get resource_amount for the row idx
        negative_row = df.loc[idx]
        resource_amount = abs(df.loc[idx, 'Resource Amount'])
        trans_date = df.loc[idx, 'Trans Date']

        matching_positive_rows = df[(df['An Type'] == 'GLR') & (df['Resource Amount'] == resource_amount) & (df['Business Unit'] == 'EU033') & (df['Member ID'] == 'MBR-20000') & (df['Trans Date'] == trans_date)]
        number_of_matching_rows = matching_positive_rows.shape[0]

        if number_of_matching_rows > 1:
            multiple_matches = multiple_matches + 1

        if number_of_matching_rows > 0:
            
            # drop the first matching positive row
            positive_idx = matching_positive_rows.index[0]

            i = 0

            # There is a single outlier in the data where there are not equal amount of negative and posive matches for the algorithm
            # It's only worth 0,01€ so there is no use to try to figure out why, just use try block to pick it.
            # After this addition, the dataframe after this function finally matches monthly amounts in the original excel sheet (after dropping controller transfers)
            try:
                while positive_idx in indices_to_drop:
                    i = i + 1
                    positive_idx = matching_positive_rows.index[i]
            
                # Drop both the negative row (idx) and the corresponding positive row (positive_idx)
                controller_transfers = pd.concat([controller_transfers, df.loc[[idx]], df.loc[[positive_idx]]], ignore_index=True)

                # collect controller transfer rows in an array to be dropped after the loop
                # droppin in the middle of the loop would invalidate indices in negative_glr_indices
                indices_to_drop.extend([idx, positive_idx])
    
            except Exception as e:
                print('Could not match a positive row for a negative row:')
                #print(negative_row)
                #print('All positive mtaches')
                #print(matching_positive_rows)

    df = df.drop(indices_to_drop)

    print('Multiple matches in rows: ' + str(multiple_matches))

    return df, controller_transfers
          

# Check duplicates. There are ~28k duplicate rows. Need to check what they are and how to handle them.
#def check_duplicates(df):
#
#    # Find duplicates and sort
#    duplicates = df[df.duplicated(keep=False)]  # Use duplicated(), not duplicates()
#    sorted_duplicates = duplicates.sort_values(by=['Acctg Date', 'Resource Amount', 'Project ID', 'Member ID'])
#
#    # Group by Year and Month and count duplicates
#    duplicates_per_month = sorted_duplicates.groupby(['Year', 'Month']).size().reset_index(name='Duplicate Count')
#    return sorted_duplicates, duplicates_per_month         

# Calculate total revenue per month, total + per locations
def calculate_sum_per_month(df):

    # Ensure that the main program df is not changed when we are manipulating headers for the report
    df = df.copy()

    # Create a mapping for the locations (replace 1, 2, 3 with actual values for your case)
    location_map = {
        1: 'Revenue FIN',  # Location 1 is FIN
        2: 'Revenue IND',  # Location 2 is IND
        3: 'Revenue NS'    # Location 3 is NS
    }

    # Apply the location mapping to the 'Location' column
    df['Location'] = df['Location'].map(location_map)

    # Group by 'Year' and 'Month' and calculate the total sum of 'Resource Amount'
    monthly_sum = df.groupby(['Year', 'Month'])['Resource Amount'].sum().reset_index()
 
    # Rename columns for clarity
    monthly_sum.columns = ['Year', 'Month', 'Total Resource Amount']

    # Group by Year, Month, and Location to calculate the sum per month and per location
    monthly_sum_per_location = df.groupby(['Year', 'Month', 'Location'])['Resource Amount'].sum().reset_index()

    # Rename columns for clarity
    monthly_sum_per_location.columns = ['Year', 'Month', 'Location', 'Total Resource Amount per Location']

    # Optionally return the result if further processing is needed
    return monthly_sum, monthly_sum_per_location

# Calculate average hourly rates, across locations and per location
def calculate_avg_hourly_rate_per_month(df):

    # Ensure that the main program df is not changed when we are manipulating headers for the report
    df = df.copy()

    # Create a mapping for the locations (replace 1, 2, 3 with actual values for your case)
    location_map = {
        1: 'Rate FIN',  # Location 1 is FIN
        2: 'Rate IND',  # Location 2 is IND
        3: 'Rate NS'    # Location 3 is NS
    }

    # Apply the location mapping to the 'Location' column
    df['Location'] = df['Location'].map(location_map)

    # Ensure 'Quantity' and 'Unit Price (Amount/Qty)' are numeric
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Unit Price (Amount/Qty)'] = pd.to_numeric(df['Unit Price (Amount/Qty)'], errors='coerce')

    # Remove rows where Quantity or Unit Price is NaN to avoid errors
    df_cleaned = df.dropna(subset=['Quantity', 'Unit Price (Amount/Qty)'])

    # Calculate weighted sum (Quantity * Unit Price) for overall averages
    df_cleaned['Weighted Price'] = df_cleaned['Quantity'] * df_cleaned['Unit Price (Amount/Qty)']

    # Group by Year and Month to calculate the sum of 'Quantity' and 'Weighted Price'
    monthly_avg = df_cleaned.groupby(['Year', 'Month']).agg(
        total_quantity=('Quantity', 'sum'),
        total_weighted_price=('Weighted Price', 'sum')
    ).reset_index()

    # Calculate the average hourly rate for each month
    monthly_avg['Rate'] = monthly_avg['total_weighted_price'] / monthly_avg['total_quantity']

    # Group by Year, Month, and Location for averages per location
    monthly_avg_per_location = df_cleaned.groupby(['Year', 'Month', 'Location']).agg(
        total_quantity=('Quantity', 'sum'),
        total_weighted_price=('Weighted Price', 'sum')
    ).reset_index()

    # Calculate the average hourly rate for each location per month
    monthly_avg_per_location['Average Hourly Rate per Location'] = (
        monthly_avg_per_location['total_weighted_price'] / monthly_avg_per_location['total_quantity']
    )

    # Return both DataFrames
    return monthly_avg, monthly_avg_per_location

# Calculate total hours per month, total and per location
def calculate_hours_per_month(df):

    # Ensure that the main program df is not changed when we are manipulating headers for the report
    df = df.copy()

    # Create a mapping for the locations (replace 1, 2, 3 with actual values for your case)
    location_map = {
        1: 'Hours FIN',  # Location 1 is FIN
        2: 'Hours IND',  # Location 2 is IND
        3: 'Hours NS'    # Location 3 is NS
    }

    # Apply the location mapping to the 'Location' column
    df['Location'] = df['Location'].map(location_map)
    # Ensure that Quantity is numeric (to handle potential non-numeric values)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

    # Group by Year and Month to calculate total hours (Quantity) across all locations
    total_hours_per_month = df.groupby(['Year', 'Month'])['Quantity'].sum().reset_index()

    # Rename the columns for clarity
    total_hours_per_month.columns = ['Year', 'Month', 'Total Hours']

    # Group by Year, Month, and Location to calculate total hours (Quantity) per location
    hours_per_location_monthly = df.groupby(['Year', 'Month', 'Location'])['Quantity'].sum().reset_index()

    # Rename the columns for clarity
    hours_per_location_monthly.columns = ['Year', 'Month', 'Location', 'Total Hours per Location']

    # Return both DataFrames
    return total_hours_per_month, hours_per_location_monthly

# 
def print_data(df): #, controller_transfers):

    # Print the preprosessed datafile, including derieved data columns and filtered data
    with open('Prepros_3.csv', 'w', newline='') as file:
        df.to_csv(file, index=False)

    # Write the dropped transfer rows to a CSV file
    #with open('controller_transfers.csv', 'w', newline='') as file:
    #    controller_transfers.to_csv(file, index=False)


def print_report(monthly_sums, monthly_sums_per_location, monthly_avg_rates, monthly_avg_rates_per_location, total_hours_per_month, hours_per_location_monthly):
    
    # Merge the non-location-based columns (monthly sums, avg rates, and total hours)
    report_df = pd.merge(monthly_sums, monthly_avg_rates, on=['Year', 'Month'], how='left')
    report_df = pd.merge(report_df, total_hours_per_month, on=['Year', 'Month'], how='left')

    # Pivot the location-based data to get separate columns for each location
    pivot_avg_rates_per_location = monthly_avg_rates_per_location.pivot_table(
        index=['Year', 'Month'], columns='Location', values='Average Hourly Rate per Location'
    ).reset_index()

    pivot_sums_per_location = monthly_sums_per_location.pivot_table(
        index=['Year', 'Month'], columns='Location', values='Total Resource Amount per Location'
    ).reset_index()

    pivot_hours_per_location = hours_per_location_monthly.pivot_table(
        index=['Year', 'Month'], columns='Location', values='Total Hours per Location'
    ).reset_index()

    # Merge the pivoted data with the main report
    report_df = pd.merge(report_df, pivot_avg_rates_per_location, on=['Year', 'Month'], how='left', suffixes=('', '_'))
    report_df = pd.merge(report_df, pivot_sums_per_location, on=['Year', 'Month'], how='left', suffixes=('', '_'))
    report_df = pd.merge(report_df, pivot_hours_per_location, on=['Year', 'Month'], how='left', suffixes=('', '_'))

    report_df.drop('total_quantity', axis=1, inplace=True)
    report_df.drop('total_weighted_price', axis=1, inplace=True)

    report_df.rename(columns={'Total Resource Amount': 'Revenue'}, inplace=True)
    report_df.rename(columns={'Total Hours': 'Hours'}, inplace=True)


    # Rename the columns for clarity
    report_df.columns = [col if not isinstance(col, tuple) else f'{col[1]}_{col[0]}' for col in report_df.columns]

    column_order = [
        'Year', 
        'Month',
        'Revenue', 
        'Revenue FIN', 
        'Revenue IND', 
        'Revenue NS', 
        'Hours',
        'Hours FIN',
        'Hours IND',
        'Hours NS',
        'Rate',
        'Rate FIN',
        'Rate IND',
        'Rate NS'
    ] + [col for col in report_df.columns if col not in [
        'Year', 
        'Month',
        'Revenue', 
        'Revenue FIN', 
        'Revenue IND', 
        'Revenue NS', 
        'Hours',
        'Hours FIN',
        'Hours IND',
        'Hours NS',
        'Rate',
        'Rate FIN',
        'Rate IND',
        'Rate NS'         
     ]]

    # Reorder the columns
    report_df = report_df[column_order]

    # Save the final report to a CSV file
    report_file_path = 'combined_monthly_report.csv'
    report_df.to_csv(report_file_path, index=False)


# THE MAIN PROGRAM STARTS HERE

start_time = time.time()

df = pd.read_csv(file_path) 
df = add_calendar_columns(df)
df = add_location_column(df)

print('Added columns: ' + str(time.time()-start_time) + ' sec')

df, controller_transfers = drop_controller_transfers(df) 

print('Dropped controller transfers: ' + str(time.time()-start_time) + ' sec')

monthly_sums, monthly_sums_per_location = calculate_sum_per_month(df)
monthly_avg_rates, monthly_avg_rates_per_location = calculate_avg_hourly_rate_per_month(df)
total_hours_per_month, hours_per_location_monthly = calculate_hours_per_month(df)

print('Created summary reports: ' + str(time.time()-start_time) + ' sec')

print_data(df) #, controller_transfers)
print_report(monthly_sums, monthly_sums_per_location, monthly_avg_rates, monthly_avg_rates_per_location, total_hours_per_month, hours_per_location_monthly)

print('Execution ending: ' + str(time.time()-start_time) + ' sec')
