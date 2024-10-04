import pandas as pd
import time

input_file_path = '2024-09-20 TLX_data.csv'
output_file_path = 'Hours data - preprosessed.csv'
summary_file_path = 'Hours data summary.csv'

# To maintain the trace to the original raw data source excel, 
# this needs to be done first
def add_id_column(df):
    ids = pd.Series(range(1, len(df) + 1))
    df['Row ID'] = 'R' + ids.astype(str).str.zfill(6)    
    return df

def drop_unnecessary_columns(df):
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

def encode_location_2(trx_general_ledger):

    if trx_general_ledger == 'EU033':
        return 1 # the company
    elif trx_general_ledger == 'EU030':
        return 2 # mothership
    elif trx_general_ledger.startswith('EU') or trx_general_ledger.startswith('GB'):
        return 3 # Other EU locations except Poland (Latvia, Lithuania, Aktia Duetto) + GB
    elif trx_general_ledger == 'IN002': 
        return 4 # CGI ISMC Priv Ltd (Bangalore)
    elif trx_general_ledger == 'IN003':
        return 5 # CGI ISMC PRIV LTD (MUMBAI)
    elif trx_general_ledger == 'IN004':
        return 6 # CGI ISMC Priv Ltd (Chennai)
    elif trx_general_ledger.startswith('IN'):
        return 7 # Other Indian locations, only minor amount
    elif trx_general_ledger.startswith('PL'):
        return 8 # Poland, most of the nearshore
    else:
        print('Not able to map detailed location for: ' + trx_general_ledger)
        return 9

def add_location_columns(df):

    df['Location'] = df['Trx- General Ledger Unit'].apply(encode_location)
    df['Location2'] = df['Trx- General Ledger Unit'].apply(encode_location_2)
    return df


def create_summary_report(df):

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

def print_data(df, df_summary, output_file_path, summary_file_path):

    with open(output_file_path, 'w', newline='') as file:
        df.to_csv(file, index=False)

    with open(summary_file_path, 'w', newline='') as file:
        df_summary.to_csv(file, index=False)
   
def main(input_file_path, output_file_path, summary_file_path):
    start_time = time.time()

    df = pd.read_csv(input_file_path) 
    df = add_id_column(df)
    df = drop_unnecessary_columns(df)
    df = add_calendar_columns(df)
    df = add_location_columns(df)
    summary_df = create_summary_report(df)

    print_data(df, summary_df, output_file_path, summary_file_path)


    print('Execution ending: ' + str(time.time()-start_time) + ' sec')