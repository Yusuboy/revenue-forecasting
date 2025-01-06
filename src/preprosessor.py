import pandas as pd
from datetime import datetime
import os


class Preprocessor:

    # ------------------------- Preprocessing utility functions ----------------------
    @staticmethod
    def add_id_column(df):
        ids = pd.Series(range(1, len(df) + 1))
        df["Row ID"] = "R" + ids.astype(str).str.zfill(6)
        return df

    @staticmethod
    def add_calendar_columns(df):
        df["Trans Date"] = pd.to_datetime(
            df["Trans Date"], format="%d/%m/%Y %H:%M", errors="raise"
        )

        df["Year"] = df["Trans Date"].dt.year  # Extract as integer year
        df["Month"] = df["Trans Date"].dt.month  # Extract as integer month (1-12)
        return df

    @staticmethod
    def encode_location(trx_general_ledger):
        if trx_general_ledger.startswith("EU"):
            return 1
        elif trx_general_ledger.startswith("IN"):
            return 2
        else:
            return 3

    @staticmethod
    def add_location_column(df):
        df["Location"] = df["Trx- General Ledger Unit"].apply(
            Preprocessor.encode_location
        )
        return df

    @staticmethod
    def drop_unnecessary_columns_bil(df):
        df.drop(
            [
                "Res Type",
                "SubCat",
                "Status",
                "BI Distrib",
                "UOM",
                "Currency Code",
                "Journal Date (Rev Rec)",
            ],
            axis=1,
            inplace=True,
        )
        return df

    @staticmethod
    def drop_controller_transfers(df):
        df["Resource Amount"] = (
            df["Resource Amount"].astype(str).str.replace(",", "").str.strip()
        )
        df["Resource Amount"] = pd.to_numeric(df["Resource Amount"], errors="coerce")

        negative_glr_df = df[
            (df["An Type"] == "GLR")
            & (df["Resource Amount"] < 0)
            & (df["Business Unit"] == "EU030")
            & (df["Member ID"] == "MBR-20000")
        ]
        negative_glr_indices = negative_glr_df.index
        indices_to_drop = []
        controller_transfers = pd.DataFrame()
        multiple_matches = 0

        for idx in negative_glr_indices:
            resource_amount = abs(df.loc[idx, "Resource Amount"])
            trans_date = df.loc[idx, "Trans Date"]
            matching_positive_rows = df[
                (df["An Type"] == "GLR")
                & (df["Resource Amount"] == resource_amount)
                & (df["Business Unit"] == "EU033")
                & (df["Member ID"] == "MBR-20000")
                & (df["Trans Date"] == trans_date)
            ]
            number_of_matching_rows = matching_positive_rows.shape[0]

            if number_of_matching_rows > 1:
                multiple_matches += 1

            if number_of_matching_rows > 0:
                positive_idx = matching_positive_rows.index[0]
                i = 0
                try:
                    while positive_idx in indices_to_drop:
                        i += 1
                        positive_idx = matching_positive_rows.index[i]

                    controller_transfers = pd.concat(
                        [controller_transfers, df.loc[[idx]], df.loc[[positive_idx]]],
                        ignore_index=True,
                    )
                    indices_to_drop.extend([idx, positive_idx])
                except Exception as e:
                    print("Could not match a positive row for a negative row:")
        df = df.drop(indices_to_drop)
        return df

    # -------------------------------------- Summarizations  -----------------------------------------
    @staticmethod
    def bil_sums_per_month(df):
        df = df.copy()
        location_map = {1: "Revenue FIN", 2: "Revenue IND", 3: "Revenue NS"}
        df["Location"] = df["Location"].map(location_map)
        df["Resource Amount"] = pd.to_numeric(df["Resource Amount"], errors="raise")
        monthly_sums = (
            df.groupby(["Year", "Month", "Location"])["Resource Amount"]
            .sum()
            .reset_index()
        )
        pivoted_monthly_sums = monthly_sums.pivot_table(
            index=["Year", "Month"],
            columns="Location",
            values="Resource Amount",
            fill_value=0,
        ).reset_index()
        pivoted_monthly_sums["Revenue"] = (
            pivoted_monthly_sums["Revenue FIN"]
            + pivoted_monthly_sums["Revenue IND"]
            + pivoted_monthly_sums["Revenue NS"]
        )
        return pivoted_monthly_sums

    # -------------------------------- Main Functional Parts -------------------------------------------
    @staticmethod
    def handle_bil_file(bil_file_path):
        df = pd.read_csv(bil_file_path)
        df = Preprocessor.add_id_column(df)
        df = Preprocessor.drop_unnecessary_columns_bil(df)
        df = Preprocessor.add_calendar_columns(df)
        df = Preprocessor.add_location_column(df)
        print("Dropping controller transfers. This may take several minutes...")
        df = Preprocessor.drop_controller_transfers(df)
        print("Dropping controller transfers done")
        return df

    # Write datafile data.csv that will be used by models.
    @staticmethod
    def create_datafile(df_bil_sums, calendar_file_path, output_datafile_path):
        df_calendar = pd.read_csv(calendar_file_path)
        df_calendar["Year"] = df_calendar["Year"].astype(int)
        df_calendar["Month"] = df_calendar["Month"].astype(int)
        df_bil_sums["Year"] = df_bil_sums["Year"].astype(int)
        df_bil_sums["Month"] = df_bil_sums["Month"].astype(int)

        # hours data not used, removed handling and merging from preprosessing
        df_merged = pd.merge(
            df_calendar, df_bil_sums, on=["Year", "Month"], how="outer"
        )

        # drop the Sep 2021 that has incorrect data
        index_to_drop = df_merged[
            (df_merged["Year"] == 2021) & (df_merged["Month"] < 10)
        ].index
        df_merged.drop(index_to_drop, inplace=True)

        df_merged.to_csv(output_datafile_path)

    def prerpocess(self, revenue_file_path, calendar_file_path, output_file_path):
        df_bil = Preprocessor.handle_bil_file(revenue_file_path)
        df_bil_sums = Preprocessor.bil_sums_per_month(df_bil)
        Preprocessor.create_datafile(df_bil_sums, calendar_file_path, output_file_path)


if __name__ == "__main__":

    print("Current Working Directory:", os.getcwd())

    output_file_path = "data.csv"
    revenue_file_path = "revenue.csv"
    calendar_file_path = "calendar.csv"

    p = Preprocessor()
    p.prerpocess(revenue_file_path, calendar_file_path, output_file_path)
