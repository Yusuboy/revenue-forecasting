import pandas as pd
from datetime import datetime

bil_file_path = 'revenue.csv'
tlx_file_path = 'hours.csv'
calendar_file_path = 'calendar.csv'
output_datafile_path = 'data.csv'
prosessed_bil_file_path = 'revenue_preprosessed.csv'
prosessed_tlx_file_path = 'hours_preprosessed.csv'

# ------------------------- Preprosessing utility functions ----------------------
def add_id_column(df):
    ids = pd.Series(range(1, len(df) + 1))
    df['Row ID'] = 'R' + ids.astype(str).str.zfill(6)    
    return df

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


def drop_unnecessary_columns_bil(df):
    df.drop('Res Type', axis=1, inplace=True)
    df.drop('SubCat', axis=1, inplace=True)
    df.drop('Status', axis=1, inplace=True)
    df.drop('BI Distrib', axis=1, inplace=True)
    df.drop('UOM', axis=1, inplace=True)
    df.drop('Currency Code', axis=1, inplace=True)
    df.drop('Journal Date (Rev Rec)', axis=1, inplace=True)
    return df

def drop_unnecessary_columns_tlx(df):
    df.drop('An Type', axis=1, inplace=True)
    df.drop('SubCat', axis=1, inplace=True)
    df.drop('Status', axis=1, inplace=True)
    df.drop('BI Distrib', axis=1, inplace=True)
    df.drop('UOM', axis=1, inplace=True)
    df.drop('Resource Amount', axis=1, inplace=True)
    df.drop('Currency Code', axis=1, inplace=True)
    df.drop('Unit Price (Amount/Qty)', axis=1, inplace=True)
    df.drop('Journal Date (Rev Rec)', axis=1, inplace=True)
    return df

# Find the transactions that represent the Indian revenue transferred from the # offshore project to the local project monthly. 
# There is always a negative transaction in the offshore project (EU030) and corresponding positive transaction in the joint venture project (EU033). 
# These transfers need to be dropped out to get monthly revenues per location calculated correctly.
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
        # negative_row = df.loc[idx]
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
            # It's only worth 0,01€ so there is no use to try to figure out why, just use a try block to pick it.
            # After this addition, dataframe finally matches monthly sum amounts in the original excel sheet 
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

    #print('Multiple matches in rows: ' + str(multiple_matches))

    return df

# -------------------------------------- summarizations  -----------------------------------------
# Calculate total revenue per month, total + per locations
def bil_sums_per_month(df):

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

    # Drop rows where 'Resource Amount' is NaN (result of non-numeric values)
    #df = df.dropna(subset=['Resource Amount'])
    #df['Resource Amount'] = df['Resource Amount'].str.replace(',', '')  # Remove commas
    df['Resource Amount'] = pd.to_numeric(df['Resource Amount'], errors='raise')  # Convert to numeric

    # Group by Year, Month, and Location to calculate the sum per month and per location
    monthly_sums = df.groupby(['Year', 'Month', 'Location'])['Resource Amount'].sum().reset_index()

    # Pivot the table to have separate columns for each location's revenue
    pivoted_monthly_sums = monthly_sums.pivot_table(index=['Year', 'Month'], columns='Location', values='Resource Amount', fill_value=0).reset_index()

    pivoted_monthly_sums['Revenue'] =  pivoted_monthly_sums['Revenue FIN'] + pivoted_monthly_sums['Revenue IND'] + pivoted_monthly_sums['Revenue NS']
    # Optionally return the result if further processing is needed
    return pivoted_monthly_sums

# Calculate hours sums per month, per locations and project type (T&M, fixed fee, internal)
def tlx_sums_per_month(df):
    
    summary_columns = ['Year', 'Month', 'Hours 10* FI', 'Hours 20* FI', 'Hours 30*FI', 'Hours 10* IN', 'Hours 20* IN', 'Hours 30*IN', 'Hours 10* NS', 'Hours 20* NS', 'Hours 30*NS']
    summary_df = pd.DataFrame(columns=summary_columns)

    for (year, month), group in df.groupby(['Year', 'Month']):
        # Initialize the row with zeros
        summary_row = {
            'Year': year,
            'Month': month,
            'Hours 10* FI': 0,
            'Hours 20* FI': 0,
            'Hours 30*FI': 0,
            'Hours 10* IN': 0,
            'Hours 20* IN': 0,
            'Hours 30*IN': 0,
            'Hours 10* NS': 0,
            'Hours 20* NS': 0,
            'Hours 30*NS': 0
        }
        
        # Loop over the records in the group
        for _, row in group.iterrows():
            location = row['Location']
            proj_type = str(row['Proj Type'])
            quantity = row['Quantity']
            
            # Apply the conditions to assign the quantity to the correct summary column
            if location == 1:
                if proj_type.startswith('1'):
                    summary_row['Hours 10* FI'] += quantity
                elif proj_type.startswith('2'):
                    summary_row['Hours 20* FI'] += quantity
                elif proj_type.startswith('3'):
                    summary_row['Hours 30*FI'] += quantity
            elif location == 2:
                if proj_type.startswith('1'):
                    summary_row['Hours 10* IN'] += quantity
                elif proj_type.startswith('2'):
                    summary_row['Hours 20* IN'] += quantity
                elif proj_type.startswith('3'):
                    summary_row['Hours 30*IN'] += quantity
            elif location == 3:
                if proj_type.startswith('1'):
                    summary_row['Hours 10* NS'] += quantity
                elif proj_type.startswith('2'):
                    summary_row['Hours 20* NS'] += quantity
                elif proj_type.startswith('3'):
                    summary_row['Hours 30*NS'] += quantity
        
        # Append the summary row to the summary DataFrame
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)

    return summary_df    

# -------------------------------- Main functional parts -------------------------------------------
# Prerpocess the revenue data file. 
# This data includes the source data that is accumulated to monthly revenues per location (FIN/IND/others).
# Monthly revenues per location are target variables for the forecasting models.
def handle_bil_file():
    df = pd.read_csv(bil_file_path) 
    df = add_id_column(df)
    df = drop_unnecessary_columns_bil(df)
    df = add_calendar_columns(df)
    df = add_location_column(df)

    print('Dropping controller transfers. This may take serveral minutes...')
    df = drop_controller_transfers(df)
    print('Dropping controller transfers done')

    return df

# Prerpocess the hours data file. 
# This data is accumulated to monthly hours per project category (T&M/fixed fee/internal) and location (FIN/IND/ohters)
# This data can be used as exploratory variables in the model
def handle_tlx_file():
    df = pd.read_csv(tlx_file_path) 
    df = add_id_column(df)
    df = drop_unnecessary_columns_tlx(df)
    df = add_calendar_columns(df)
    df = add_location_column(df)
    return df

# Create the datafile to be used as a source data for the model. This combines summarized hours (TLX), revenue (BIL) and 
# calendar (working days calendar, rate raises) data. 
# Also printing out the preprosessed hours and revenue source files for debugging purposes. These can be commented out later if not needed.
def create_datafile(df_tlx_sums, df_bil_sums, df_tlx, df_bil):

    # Read working day calendar, split combined month field for Year and Month columns for merging
    df_calendar = pd.read_csv(calendar_file_path)

    # Merge revenue (bil), hour (tlx) data with working day calendar and rate raise information
    # To ensure successful merge, ensure that all datatypes of merge fields are integers
    df_calendar['Year'] = df_calendar['Year'].astype(int)
    df_calendar['Month'] = df_calendar['Month'].astype(int)
    df_tlx_sums['Year'] = df_tlx_sums['Year'].astype(int)
    df_tlx_sums['Month'] = df_tlx_sums['Month'].astype(int)
    df_tlx_sums['Year'] = df_tlx_sums['Year'].astype(int)
    df_tlx_sums['Month'] = df_tlx_sums['Month'].astype(int)    
    df_merged = pd.merge(df_calendar, df_tlx_sums, on=['Year', 'Month'], how='outer')
    df_merged = pd.merge(df_merged, df_bil_sums, on=['Year', 'Month'], how='outer')
    df_merged.to_csv(output_datafile_path)

    # Print preprosessed BIL and TLX files for debuggin purposes
    df_tlx.to_csv(prosessed_tlx_file_path)
    df_bil.to_csv(prosessed_bil_file_path)

def main():
    print('preprosessing the revenue.csv file...')
    df_bil = handle_bil_file()
    print('preprosessing the hours.csv (TLX) file...')    
    df_tlx = handle_tlx_file()
    print('calculating revenue sums...')    
    df_bil_sums = bil_sums_per_month(df_bil)
    print('calculating hours sums...')    
    df_tlx_sums = tlx_sums_per_month(df_tlx)
    print('creating the forecast source file...')    
    create_datafile(df_tlx_sums, df_bil_sums, df_tlx, df_bil)
    print('all done - finishing')   

main()