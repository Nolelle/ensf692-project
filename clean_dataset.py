"""
ENSF 692 – Clean and Describe Calgary Housing & Demographics Data
----------------------------------------------------------------
Loads, cleans, merges, and enriches three City of Calgary open data tables
(filtered to 2016–2017 data):

Input Files:
  1. Civic Census by Community and Dwelling Structure
     → CSVFiles/Civic_Census_by_Community_and_Dwelling_Structure_20250613.csv
  2. Assessments by Community
     → CSVFiles/Assessments by Community Jun 2025.csv
  3. Communities by Ward
     → CSVFiles/Communities by Ward June 2025.csv

Data Processing:
• Standardizes column names and community names (uppercase, stripped)
• Filters census and assessment data to 2016-2017 only
• Merges datasets on community name and year
• Removes duplicates and system/unclassified entries
• Converts numeric columns (handles comma-separated values)
• Derives AREA_TYPE classification (Inner-City vs Suburban) from community category

Calculated Metrics:
  • ASSESSMENT_PER_PERSON  = MEDIAN_ASSESSMENT / RES_CNT
  • VACANCY_RATE           = DWELLINGS_VACANT / DWELLINGS_TOTAL

Output:
• Exports cleaned dataset to 'cleaned_calgary_housing_demographics.csv'
• Prints first 5 rows, descriptive statistics, and missing value counts
• Creates multi-index DataFrame (COMMUNITY_NAME, YEAR) sorted alphabetically

The final dataset supports analysis of Calgary housing and demographic trends
across communities and area types for the 2016-2017 period.
"""

import pandas as pd

# ------------------------------------------------------------------ constants
YEARS = (2016, 2017)
COL_NAME = "COMMUNITY_NAME"


# ------------------------------------------------------------------ helpers
def clean_names(df, col):
    """
    Standardizes the values in a specified column of a DataFrame by converting them to uppercase and stripping whitespace.
    This helps ensure reliable merging and comparison of string columns (e.g., community names).

    Parameters:
        df (pd.DataFrame): The DataFrame to modify.
        col (str): The column name to clean.
    Returns:
        None. The DataFrame is modified in place.
    """
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().str.strip()


def map_inner_city(cat):
    """
    Classifies a community as 'Inner-City' or 'Suburban' based on its category label.

    Methodology:
    - If the input 'cat' is not a string (e.g., missing or NaN), returns 'Suburban'.
    - Converts the input to uppercase for case-insensitive comparison.
    - If the uppercase value is one of {'CITY CENTRE', 'ESTABLISHED', 'INNER CITY'}, returns 'Inner-City'.
    - Otherwise, returns 'Suburban'.

    Parameters:
        cat (str): The community category label.
    Returns:
        str: 'Inner-City' or 'Suburban' classification.

    Summary Table:
        Input Category (case-insensitive) | Output
        ----------------------------------|-----------
        'CITY CENTRE'                     | Inner-City
        'ESTABLISHED'                     | Inner-City
        'INNER CITY'                      | Inner-City
        Anything else or not a string     | Suburban
    """
    if not isinstance(cat, str):
        return "Suburban"
    upper = cat.upper()
    if upper == "CITY CENTRE":
        return "Inner-City"
    elif upper == "ESTABLISHED":
        return "Inner-City"
    elif upper == "INNER CITY":
        return "Inner-City"
    else:
        return "Suburban"


# ------------------------------------------------------------------ main ETL


def load_and_prepare_data(census_path, assessment_path, ward_path):
    """
    Loads, cleans, merges, and enriches three Calgary community datasets (census, assessment, and ward info).
    Standardizes columns, filters years, merges on community and year, and adds calculated columns.

    Parameters:
        census_path (str): Path to the census CSV file.
        assessment_path (str): Path to the assessment CSV file.
        ward_path (str): Path to the ward/community info CSV file.
    Returns:
        pd.DataFrame: The cleaned, merged, and enriched DataFrame, indexed by community and year.
    """
    # Load the three CSV files into DataFrames
    census = pd.read_csv(census_path)
    assess = pd.read_csv(assessment_path)
    ward = pd.read_csv(ward_path)

    # Standardize ward/community DataFrame column names for merging
    ward = ward.rename(columns={"NAME": COL_NAME, "sector": "SECTOR"})

    # Create AREA_TYPE column if it doesn't exist, classifying communities
    if "AREA_TYPE" not in ward.columns:
        # Try to find a column with 'category' in its name
        cat_col = next((c for c in ward.columns if "category" in c.lower()), None)
        # Use map_inner_city to classify, or default to 'Suburban' if not found
        ward["AREA_TYPE"] = (
            ward[cat_col].apply(map_inner_city) if cat_col else "Suburban"
        )

    # Standardize census DataFrame column names for consistency
    census = census.rename(
        columns={
            "CENSUS_YEAR": "YEAR",
            "COMMUNITY": COL_NAME,
            "RESIDENT_CNT": "RES_CNT",
            "DWELLING_CNT": "DWELLINGS_TOTAL",
            "VACANT_DWELLING_CNT": "DWELLINGS_VACANT",
        }
    )

    # Find and standardize the median assessment column name in assessment DataFrame
    median_col = next(
        (c for c in assess.columns if "median" in c.lower() and "assess" in c.lower()),
        "Median assessed value",
    )
    assess = assess.rename(
        columns={
            "date": "YEAR",
            "Community name": COL_NAME,
            median_col: "MEDIAN_ASSESSMENT",
        }
    )

    # Filter census and assessment data to only include the years of interest
    census = census[census["YEAR"].isin(YEARS)]
    assess = assess[assess["YEAR"].isin(YEARS)]

    # Clean up community names in all DataFrames for reliable merging
    for df in (census, assess, ward):
        clean_names(df, COL_NAME)

    # Remove duplicate assessment rows (by community and year)
    assess.drop_duplicates(subset=[COL_NAME, "YEAR"], keep="first", inplace=True)

    # Remove system/unclassified rows from all DataFrames
    EXCLUDE_NAME = "SYSTEM/UNCLASSIFIED/RESIDUAL WARD"
    for df in (census, assess, ward):
        df.drop(df[df[COL_NAME] == EXCLUDE_NAME].index, inplace=True)

    # Remove redundant columns from census and assessment DataFrames
    for col in ("SECTOR", "AREA_TYPE"):
        census.drop(columns=col, errors="ignore", inplace=True)
        assess.drop(columns=col, errors="ignore", inplace=True)

    # Merge census and ward DataFrames on community name
    merged = pd.merge(census, ward, on=COL_NAME, how="left", validate="m:1")
    # Merge in assessment DataFrame on community name and year
    merged = pd.merge(merged, assess, on=[COL_NAME, "YEAR"], how="left", validate="m:1")

    # Convert important columns to numeric, removing commas and filling missing with 0
    for col in ("MEDIAN_ASSESSMENT", "RES_CNT", "DWELLINGS_VACANT", "DWELLINGS_TOTAL"):
        merged[col] = (
            merged[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
            .fillna(0)
        )

    # Add calculated columns: assessment per person and vacancy rate
    merged["ASSESSMENT_PER_PERSON"] = merged["MEDIAN_ASSESSMENT"] / merged[
        "RES_CNT"
    ].replace(0, pd.NA)
    merged["VACANCY_RATE"] = merged["DWELLINGS_VACANT"] / merged[
        "DWELLINGS_TOTAL"
    ].replace(0, pd.NA)

    # Set a multi-level index (community name and year) and sort the DataFrame
    merged.set_index([COL_NAME, "YEAR"], inplace=True)
    merged.sort_index(inplace=True)

    # Return the cleaned, merged, and enriched DataFrame
    return merged


# ------------------------------------------------------------------ CLI entry


def main():
    """
    Main entry point for the script. Loads, cleans, and prints the Calgary housing & demographics dataset.
    Prints the first 5 rows and descriptive statistics for the cleaned dataset.
    """
    # Run the data loading and cleaning process, then print results
    df = load_and_prepare_data(
        census_path="CSVFiles/Civic_Census_by_Community_and_Dwelling_Structure_20250613.csv",
        assessment_path="CSVFiles/Assessments by Community Jun 2025.csv",
        ward_path="CSVFiles/Communities by Ward June 2025.csv",
    )

    # Export the cleaned dataset to CSV
    df.to_csv("cleaned_calgary_housing_demographics.csv")
    print("Dataset exported to 'cleaned_calgary_housing_demographics.csv'")

    print("=" * 60)
    print("First 5 Rows of the Cleaned Dataset")
    print("=" * 60)
    print(df.head())
    print("\n" + "=" * 60)
    print("Descriptive Statistics for the Cleaned Dataset (2016–2017)")
    print("=" * 60)
    print(df.describe(include="all"))
    print("\n" + "=" * 60)
    print("Missing values per column:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
