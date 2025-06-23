"""
ENSF 692 Spring 2025 - Capstone Project
Group 3 – Calgary Housing & Demographics Data Cleaning Module
Authors: Peter Osaade, Edmund Yu
Created: June 2025

Calgary Housing & Demographics Data Cleaning and Processing Module
================================================================

Merges three City of Calgary open data tables (filtered to 2016–2017):
  1. Civic Census by Community and Dwelling Structure  (set9‑futw)
  2. Assessments by Community                         (p84b‑7zbi)
  3. Communities by Ward                              (jd78‑wxjp)

Classification Logic:
- Inner-City: Communities with 'INNER CITY', 'CENTRE CITY', or pre-1980s development
- Suburban: Communities with 'BUILDING OUT', post-1980s development, or new/developing areas

Derived Metrics:
  • ASSESSMENT_PER_PERSON  = MEDIAN_ASSESSMENT / RES_CNT
  • VACANCY_RATE           = DWELLINGS_VACANT / DWELLINGS_TOTAL

Output:
- Exports cleaned dataset to 'cleaned_calgary_housing_demographics.csv'
- Creates multi-index DataFrame (COMMUNITY_NAME, YEAR) sorted alphabetically
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


def map_inner_city(comm_structure_value):
    """
    Classify a community as 'Inner-City' or 'Suburban' based on COMM_STRUCTURE
    value from Calgary's classification system.

    Calgary's COMM_STRUCTURE indicates development timeline:
    - 'INNER CITY': Explicitly marked inner-city communities
    - 'CENTRE CITY': Downtown core areas
    - Pre-1980s decades: Established/mature communities (inner-city)
    - Post-1980s decades: Newer suburban developments
    - 'BUILDING OUT': Currently developing (suburban)

    Parameters:
        comm_structure_value (str): The community COMM_STRUCTURE value.
    Returns:
        str: 'Inner-City' or 'Suburban' classification.
    """
    if not isinstance(comm_structure_value, str):
        return "Suburban"  # Default if no structure info

    structure = comm_structure_value.upper().strip()

    # Explicit inner-city classifications
    inner_city_structures = {
        "INNER CITY",
        "CENTRE CITY",
        "INNER-CITY",
        "CENTER CITY",
        "DOWNTOWN",
    }

    # Check for exact matches
    if structure in inner_city_structures:
        return "Inner-City"

    # Decade-based classification
    # Communities developed before 1980 are typically inner-city
    inner_city_decades = {
        "PRE 1910",
        "PRE-1910",
        "BEFORE 1910",
        "1910S",
        "1920S",
        "1930S",
        "1940S",
        "1950S",
        "1960S",
        "1970S",
        "1960S/1970S",
        "1960/1970",
        "1960-1970",
    }

    # Check decade-based classification
    for decade in inner_city_decades:
        if decade in structure:
            return "Inner-City"

    # Suburban indicators
    suburban_structures = {
        "BUILDING OUT",
        "BUILDOUT",
        "BUILD OUT",
        "DEVELOPING",
        "FUTURE",
        "1980S",
        "1990S",
        "2000S",
        "2010S",
        "2020S",
        "NEW",
        "GREENFIELD",
    }

    # Check for suburban matches
    for suburban in suburban_structures:
        if suburban in structure:
            return "Suburban"

    # Default to Suburban for unknown structures
    return "Suburban"


# ------------------------------------------------------------------ load and prepare data


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

    # Create AREA_TYPE column based on COMM_STRUCTURE column (FIXED)
    if "AREA_TYPE" not in ward.columns:
        if "COMM_STRUCTURE" in ward.columns:
            # Use the COMM_STRUCTURE column for classification
            ward["AREA_TYPE"] = ward["COMM_STRUCTURE"].apply(map_inner_city)
            print("\nArea Type Classification Summary:")
            print(ward["AREA_TYPE"].value_counts())
            print("\nSample classifications:")
            sample = (
                ward[["COMM_STRUCTURE", "AREA_TYPE", COL_NAME]]
                .drop_duplicates()
                .head(10)
            )
            print(sample)
        else:
            # Fallback if no COMM_STRUCTURE column found
            print(
                "Warning: No COMM_STRUCTURE column found for area type classification"
            )
            ward["AREA_TYPE"] = "Suburban"

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
    print("\nDataset exported to 'cleaned_calgary_housing_demographics.csv'")

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

    # Show area type distribution
    print("\n" + "=" * 60)
    print("Area Type Distribution in Final Dataset:")
    area_counts = df.groupby("AREA_TYPE").size()
    print(area_counts)
    print("=" * 60)


if __name__ == "__main__":
    main()
