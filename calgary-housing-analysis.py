"""
ENSF 692 Spring 2025 Project
Group 3 – Calgary Housing & Demographics Analysis System
Authors: Peter Osaade, Edmund Yu,
Created: June 2025

Calgary Housing and Demographics Analysis System
==============================================

This program analyzes Calgary housing and demographic data from 2016-2017,
allowing users to explore community statistics, create visualizations,
and export comprehensive reports.

Research Questions Addressed:
1. Which City Sectors are Growing Fastest?
2. Which sectors have the highest property values?
3. Inner-City vs Suburban Growth Patterns

Dataset Classification:
- Communities are classified as Inner-City or Suburban based on COMM_STRUCTURE
- Inner-City: Communities with 'INNER CITY', 'CENTRE CITY', or pre-1980s development
- Suburban: Communities with 'BUILDING OUT', post-1980s development, or new/developing areas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clean_dataset import load_and_prepare_data


class CalgaryHousingAnalyzer:
    """
    Analyzes Calgary housing and demographic data with user interaction capabilities.

    Handles missing values appropriately:
    - Ward/Sector/Area Type: Displayed as "N/A" in outputs
    - Assessment/Population rates: Kept as NaN to avoid misleading calculations
    - Dwelling counts: Filled with 0 where appropriate
    - All aggregations filter NaN values before calculation

    Instance Variables:
        df (pd.DataFrame): Merged and cleaned dataset containing census, assessment, and ward information
        user_community (str): Community selected by the user
        user_year (int): Year selected by the user
    """

    def __init__(self):
        """Initialize the analyzer by loading and preparing the dataset."""
        print("Loading Calgary Housing and Demographics Data...")
        self.df = load_and_prepare_data(
            census_path="CSVFiles/Civic_Census_by_Community_and_Dwelling_Structure_20250613.csv",
            assessment_path="CSVFiles/Assessments by Community Jun 2025.csv",
            ward_path="CSVFiles/Communities by Ward June 2025.csv",
        )
        print(f"Dataset loaded successfully! {len(self.df)} records available.")

        # Display missing values summary
        print("\nMissing Values Summary:")
        missing_counts = self.df.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            for col, count in missing_counts.items():
                pct = (count / len(self.df)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print("  No missing values found!")
        print()

        self.user_community = None
        self.user_year = None

    def handle_missing_values(self):
        """
        Handle missing values in the dataset according to data type and context.

        Parameters:
            None
        Returns:
            None
        """
        # For numeric columns that should not have zeros (like population), keep NaN
        # For columns where 0 is meaningful (like vacant dwellings), fill with 0
        fill_zero_cols = [
            "DWELLINGS_VACANT",
            "RENOVATION_DWELLING_CNT",
            "INACTIVE_CNT",
            "OTHER_PURPOSE_CNT",
        ]

        for col in fill_zero_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        # For WARD and WARD_NUM, they should be the same - use coalesce
        if "WARD" in self.df.columns and "WARD_NUM" in self.df.columns:
            self.df["WARD"] = self.df["WARD"].fillna(self.df["WARD_NUM"])
            self.df["WARD_NUM"] = self.df["WARD_NUM"].fillna(self.df["WARD"])

        # For percentage/rate columns, keep NaN to indicate "cannot calculate"
        # These are already handled in clean_dataset.py with .replace(0, pd.NA)

    def get_user_input(self):
        """
        Prompt user for community name and year with validation.

        Parameters:
            None
        Returns:
            tuple: (community_name, year) both validated against the dataset
        """
        # Get list of unique communities for validation
        communities = sorted(self.df.index.get_level_values("COMMUNITY_NAME").unique())

        # Get community input with validation
        while True:
            try:
                print("Available communities (showing first 10):")
                for comm in communities[:10]:
                    print(f"  - {comm}")
                print(f"  ... and {len(communities) - 10} more\n")

                community = (
                    input("Enter community name (or 'list' to see all): ")
                    .strip()
                    .upper()
                )

                if community == "LIST":
                    print("\nAll communities:")
                    for comm in communities:
                        print(f"  - {comm}")
                    print()
                    continue

                if community not in communities:
                    print(f"\nError: '{community}' not found.")
                    print(
                        "Tip: Try typing just the first few letters, or use 'list' to see all options"
                    )
                    continue

                break

            except ValueError as e:
                print(f"\nError: {e}")
                print("Please try again.\n")

        # Get year input with validation
        while True:
            try:
                year = int(input("Enter year (2016 or 2017): "))

                if year not in [2016, 2017]:
                    raise ValueError("Year must be 2016 or 2017")

                # Check if this community has data for this year
                if (community, year) not in self.df.index:
                    raise ValueError(f"No data available for {community} in {year}")

                break

            except ValueError as e:
                print(f"\nError: {e}")
                print("Please try again.\n")

        self.user_community = community
        self.user_year = year
        return community, year

    def display_community_info(self, community, year):
        """
        Display detailed information for the selected community and year.

        Parameters:
            community (str): The community name to display
            year (int): The year to display
        Returns:
            None
        """
        print("\n" + "=" * 60)
        print(f"Community Profile: {community} ({year})")
        print("=" * 60)

        # Get all data for this community/year combination
        community_data = self.df.loc[(community, year)]

        # Display key metrics
        print("\nBasic Information:")

        # Handle WARD display (can be float or NaN)
        ward_value = community_data["WARD"].iloc[0]
        if pd.notna(ward_value):
            print(f"  Ward: {int(ward_value)}")
        else:
            print("  Ward: N/A")

        # Handle SECTOR display
        sector_value = community_data["SECTOR"].iloc[0]
        if pd.notna(sector_value):
            print(f"  Sector: {sector_value}")
        else:
            print("  Sector: N/A")

        # Handle AREA_TYPE display
        area_value = community_data["AREA_TYPE"].iloc[0]
        if pd.notna(area_value):
            print(f"  Area Type: {area_value}")
        else:
            print("  Area Type: N/A")

        print("\nPopulation & Housing:")
        total_pop = community_data["RES_CNT"].sum()
        total_dwellings = community_data["DWELLINGS_TOTAL"].sum()
        vacant = community_data["DWELLINGS_VACANT"].sum()

        print(f"  Total Population: {total_pop:,.0f}")
        print(f"  Total Dwellings: {total_dwellings:,.0f}")
        print(f"  Vacant Dwellings: {vacant:,.0f}")

        if total_dwellings > 0:
            vacancy_pct = vacant / total_dwellings * 100
            print(f"  Vacancy Rate: {vacancy_pct:.1f}%")
        else:
            print("  Vacancy Rate: N/A")

        print("\nAssessment Values:")
        median_assess = community_data["MEDIAN_ASSESSMENT"].iloc[0]
        assess_per_person = community_data["ASSESSMENT_PER_PERSON"].iloc[0]

        if pd.notna(median_assess):
            print(f"  Median Assessment: ${median_assess:,.0f}")
        else:
            print("  Median Assessment: N/A")

        if pd.notna(assess_per_person):
            print(f"  Assessment per Person: ${assess_per_person:,.0f}")
        else:
            print("  Assessment per Person: N/A")

        print("\nDwelling Types:")
        dwelling_summary = community_data.groupby("DWELLING_TYPE_DESCRIPTION")[
            "DWELLINGS_TOTAL"
        ].sum()
        for dtype, count in dwelling_summary.items():
            if count > 0:
                print(f"  {dtype}: {count:,.0f}")

    def perform_analysis(self):
        """
        Perform comprehensive analysis operations including aggregation,
        masking, groupby, and pivot table creation, plus answer three
        key research questions using the data.

        Parameters:
            None
        Returns:
            None
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE DATA ANALYSIS")
        print("=" * 60)

        # 1. Aggregation computation for subset
        print("\n1. AGGREGATION: Average Assessment by Area Type (2017 only):")
        subset_2017 = self.df[self.df.index.get_level_values("YEAR") == 2017]
        subset_2017_clean = subset_2017[
            (subset_2017["MEDIAN_ASSESSMENT"].notna()) & (subset_2017["RES_CNT"] > 0)
        ]
        area_avg = subset_2017_clean.groupby("AREA_TYPE")["MEDIAN_ASSESSMENT"].agg(
            ["mean", "count"]
        )
        area_avg.columns = ["Average Assessment", "Community Count"]
        print(area_avg.round(0))
        print(
            f"Note: Excluded {len(subset_2017) - len(subset_2017_clean)} records with missing data"
        )

        # 2. Masking operation
        print("\n2. MASKING: High-Value Communities (Median Assessment > $600,000):")
        mask = (
            (self.df["MEDIAN_ASSESSMENT"] > 600000)
            & (self.df["MEDIAN_ASSESSMENT"].notna())
            & (self.df["RES_CNT"] > 0)
        )
        high_value = self.df[mask]
        print(f"  Total high-value records: {len(high_value)}")
        print(
            f"  Unique communities: {high_value.index.get_level_values('COMMUNITY_NAME').nunique()}"
        )

        top_5 = high_value.nlargest(5, "MEDIAN_ASSESSMENT")[
            ["MEDIAN_ASSESSMENT", "SECTOR", "AREA_TYPE"]
        ]
        print("\n  Top 5 Highest Assessments:")
        for idx, row in top_5.iterrows():
            print(
                f"    {idx[0]} ({idx[1]}): ${row['MEDIAN_ASSESSMENT']:,.0f} - {row['SECTOR']}"
            )

        # 3. Groupby operation
        print("\n3. GROUPBY: Population and Assessment Summary by Sector:")
        sector_data = self.df[(self.df["SECTOR"].notna()) & (self.df["RES_CNT"] > 0)]
        sector_stats = sector_data.groupby("SECTOR").agg(
            {"RES_CNT": "sum", "MEDIAN_ASSESSMENT": "mean", "VACANCY_RATE": "mean"}
        )
        sector_stats.columns = [
            "Total Population",
            "Avg Assessment",
            "Avg Vacancy Rate",
        ]
        sector_stats["Avg Assessment"] = sector_stats["Avg Assessment"].round(0)
        sector_stats["Avg Vacancy Rate"] = (
            pd.to_numeric(sector_stats["Avg Vacancy Rate"], errors="coerce") * 100
        ).round(1)
        print(sector_stats.sort_values("Total Population", ascending=False))

        # 4. Pivot table
        print("\n4. PIVOT TABLE: Growth by Sector and Year:")
        pivot_data = self.df[self.df["SECTOR"].notna()].reset_index()
        pivot = pd.pivot_table(
            pivot_data,
            values=["RES_CNT", "MEDIAN_ASSESSMENT", "VACANCY_RATE"],
            index="SECTOR",
            columns="YEAR",
            aggfunc={
                "RES_CNT": "sum",
                "MEDIAN_ASSESSMENT": "mean",
                "VACANCY_RATE": "mean",
            },
        )
        print(pivot.round(2))

        print("\n" + "=" * 80)
        print("RESEARCH QUESTIONS: ANALYSIS FINDINGS")
        print("=" * 80)

        # RESEARCH QUESTION 1: Population Growth by Sector
        print("\nRESEARCH QUESTION 1: Which sectors show the most population growth?")
        print("-" * 70)

        # Calculate actual growth rates using the pivot table data
        sector_yearly = self.df.groupby(
            ["SECTOR", self.df.index.get_level_values("YEAR")]
        ).agg({"RES_CNT": "sum"})

        growth_data = []
        print("Sector Population Growth Analysis (2016 → 2017):")
        for sector in sorted(self.df["SECTOR"].dropna().unique()):
            if (sector, 2016) in sector_yearly.index and (
                sector,
                2017,
            ) in sector_yearly.index:
                pop_2016 = sector_yearly.loc[(sector, 2016), "RES_CNT"]
                pop_2017 = sector_yearly.loc[(sector, 2017), "RES_CNT"]

                if pop_2016 > 0:
                    growth_rate = (pop_2017 - pop_2016) / pop_2016 * 100
                    change = pop_2017 - pop_2016

                    growth_data.append(
                        {
                            "Sector": sector,
                            "Pop_2016": pop_2016,
                            "Pop_2017": pop_2017,
                            "Change": change,
                            "Growth_Rate": growth_rate,
                        }
                    )

                    print(
                        f"  {sector:12}: {pop_2016:>7,} → {pop_2017:>7,} (+{change:>5,}) = {growth_rate:>+5.1f}%"
                    )

        # Find and highlight the findings
        growth_df = pd.DataFrame(growth_data)
        if not growth_df.empty:
            fastest_growing = growth_df.loc[growth_df["Growth_Rate"].idxmax()]
            largest_absolute = growth_df.loc[growth_df["Change"].idxmax()]

            print(
                f"\n  KEY FINDING 1A: {fastest_growing['Sector']} has highest growth rate at {fastest_growing['Growth_Rate']:+.1f}%"
            )
            print(
                f"  KEY FINDING 1B: {largest_absolute['Sector']} has largest population increase of {largest_absolute['Change']:,} people"
            )

        # RESEARCH QUESTION 2: Housing Market Assessment Patterns
        print("\nRESEARCH QUESTION 2: Which sectors have the highest property values?")
        print("-" * 70)

        # Use the sector_stats we already calculated
        highest_assessment = sector_stats.nlargest(3, "Avg Assessment")
        print("Top 3 Sectors by Average Property Assessment:")

        for sector, row in highest_assessment.iterrows():
            print(
                f"  {sector:12}: ${row['Avg Assessment']:>8,.0f} average (Population: {row['Total Population']:>6,})"
            )

        print(
            f"\n  KEY FINDING 2: {highest_assessment.index[0]} sector has highest average assessments"
        )

        # Show relationship between population size and property values
        print("\nPopulation vs Property Value Analysis:")
        largest_pop = sector_stats.nlargest(1, "Total Population")
        highest_val = sector_stats.nlargest(1, "Avg Assessment")

        print(
            f"  Largest Population: {largest_pop.index[0]} ({largest_pop['Total Population'].iloc[0]:,} people)"
        )
        print(
            f"  Highest Values: {highest_val.index[0]} (${highest_val['Avg Assessment'].iloc[0]:,.0f} average)"
        )

        if largest_pop.index[0] != highest_val.index[0]:
            print(
                "  KEY FINDING 2B: Population size and property values are inversely related"
            )

        # RESEARCH QUESTION 3: Inner-City vs Suburban Development Patterns
        print("\nRESEARCH QUESTION 3: How do Inner-City and Suburban areas compare?")
        print("-" * 70)

        # Calculate area type comparisons for both years
        area_data = self.df[self.df["AREA_TYPE"].notna()]
        area_yearly = area_data.groupby(
            ["AREA_TYPE", area_data.index.get_level_values("YEAR")]
        ).agg(
            {
                "RES_CNT": "sum",
                "MEDIAN_ASSESSMENT": "mean",
                "VACANCY_RATE": "mean",
            }
        )

        print("Area Type Comparison (2016 vs 2017):")
        area_comparison_data = []

        for area_type in ["Inner-City", "Suburban"]:
            if (area_type, 2016) in area_yearly.index and (
                area_type,
                2017,
            ) in area_yearly.index:
                pop_2016 = area_yearly.loc[(area_type, 2016), "RES_CNT"]
                pop_2017 = area_yearly.loc[(area_type, 2017), "RES_CNT"]

                assess_2016 = area_yearly.loc[(area_type, 2016), "MEDIAN_ASSESSMENT"]
                assess_2017 = area_yearly.loc[(area_type, 2017), "MEDIAN_ASSESSMENT"]

                vacancy_2017 = area_yearly.loc[(area_type, 2017), "VACANCY_RATE"]

                pop_growth = (
                    (pop_2017 - pop_2016) / pop_2016 * 100 if pop_2016 > 0 else 0
                )
                assess_growth = (
                    (assess_2017 - assess_2016) / assess_2016 * 100
                    if pd.notna(assess_2016) and assess_2016 > 0
                    else 0
                )

                area_comparison_data.append(
                    {
                        "Area_Type": area_type,
                        "Pop_Growth": pop_growth,
                        "Assessment_Growth": assess_growth,
                        "Vacancy_Rate_2017": vacancy_2017 * 100
                        if pd.notna(vacancy_2017)
                        else 0,
                        "Pop_2017": pop_2017,
                    }
                )

                print(f"\n  {area_type}:")
                print(
                    f"    Population Growth: {pop_growth:>+5.1f}% ({pop_2016:,} → {pop_2017:,})"
                )
                print(
                    f"    Assessment Growth: {assess_growth:>+5.1f}% (${assess_2016:,.0f} → ${assess_2017:,.0f})"
                )
                print(f"    2017 Vacancy Rate: {vacancy_2017 * 100:>5.1f}%")

        # Generate key findings
        if len(area_comparison_data) == 2:
            inner_city = next(
                item
                for item in area_comparison_data
                if item["Area_Type"] == "Inner-City"
            )
            suburban = next(
                item for item in area_comparison_data if item["Area_Type"] == "Suburban"
            )

            print(
                f"\n  KEY FINDING 3A: {'Suburban' if suburban['Pop_Growth'] > inner_city['Pop_Growth'] else 'Inner-City'} areas growing faster"
            )
            print(
                f"    - Suburban: {suburban['Pop_Growth']:+.1f}% vs Inner-City: {inner_city['Pop_Growth']:+.1f}%"
            )

            if suburban["Pop_2017"] > inner_city["Pop_2017"]:
                print(
                    f"  KEY FINDING 3B: Suburban areas now have larger population ({suburban['Pop_2017']:,} vs {inner_city['Pop_2017']:,})"
                )

            vacancy_diff = (
                suburban["Vacancy_Rate_2017"] - inner_city["Vacancy_Rate_2017"]
            )
            print(
                f"  KEY FINDING 3C: {'Suburban' if vacancy_diff > 0 else 'Inner-City'} areas have higher vacancy ({abs(vacancy_diff):.1f}% difference)"
            )

        # SUMMARY OF ALL FINDINGS
        print("\n" + "=" * 80)
        print("SUMMARY: KEY INSIGHTS")
        print("=" * 80)

        if not growth_df.empty:
            fastest_sector = growth_df.loc[growth_df["Growth_Rate"].idxmax(), "Sector"]
            fastest_rate = growth_df.loc[
                growth_df["Growth_Rate"].idxmax(), "Growth_Rate"
            ]
            print(
                f"1. FASTEST GROWING SECTOR: {fastest_sector} (+{fastest_rate:.1f}% population growth)"
            )

        if not sector_stats.empty:
            richest_sector = sector_stats.nlargest(1, "Avg Assessment").index[0]
            richest_value = sector_stats.nlargest(1, "Avg Assessment")[
                "Avg Assessment"
            ].iloc[0]
            print(
                f"2. HIGHEST PROPERTY VALUES: {richest_sector} sector (${richest_value:,.0f} average)"
            )

        if len(area_comparison_data) == 2:
            urban_pattern = (
                "suburban sprawl"
                if suburban["Pop_Growth"] > inner_city["Pop_Growth"]
                else "urban densification"
            )
            print(f"3. DEVELOPMENT PATTERN: Calgary shows {urban_pattern} trend")

        print(
            f"4. DATA QUALITY: Analysis based on {len(sector_data)} valid records from {len(self.df)} total"
        )
        print("=" * 80)

    def create_research_visualizations(self):
        """
        Create clean visualizations that directly support the console analysis findings.
        """
        print("Creating research visualizations...")

        # Set up matplotlib style and figure
        plt.style.use("default")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Calgary Housing Analysis: Key Research Findings (2016-2017)",
            fontsize=18,
            fontweight="bold",
        )

        # 1. RESEARCH QUESTION 1: Sector Population Growth
        print("  Creating Chart 1: Sector Population Growth...")

        # Calculate growth data
        sector_yearly = self.df.groupby(
            ["SECTOR", self.df.index.get_level_values("YEAR")]
        ).agg({"RES_CNT": "sum"})

        growth_data = []
        for sector in self.df["SECTOR"].dropna().unique():
            if (sector, 2016) in sector_yearly.index and (
                sector,
                2017,
            ) in sector_yearly.index:
                pop_2016 = sector_yearly.loc[(sector, 2016), "RES_CNT"]
                pop_2017 = sector_yearly.loc[(sector, 2017), "RES_CNT"]

                if pop_2016 > 0:
                    growth_rate = (pop_2017 - pop_2016) / pop_2016 * 100
                    growth_data.append(
                        {
                            "Sector": sector,
                            "Growth_Rate": growth_rate,
                            "Pop_2017": pop_2017,
                        }
                    )

        if growth_data:
            growth_df = pd.DataFrame(growth_data).sort_values(
                "Growth_Rate", ascending=True
            )

            # Create horizontal bar chart
            colors = [
                "#d73027" if x < 0 else "#1a9850" for x in growth_df["Growth_Rate"]
            ]
            bars = ax1.barh(
                range(len(growth_df)),
                growth_df["Growth_Rate"],
                color=colors,
                alpha=0.7,
                edgecolor="black",
            )

            ax1.set_yticks(range(len(growth_df)))
            ax1.set_yticklabels(growth_df["Sector"], fontsize=10)
            ax1.set_xlabel("Population Growth Rate (%)", fontsize=12, fontweight="bold")
            ax1.set_title(
                "Q1: Population Growth by Sector (2016→2017)",
                fontsize=14,
                fontweight="bold",
            )
            ax1.grid(axis="x", alpha=0.3)
            ax1.axvline(x=0, color="black", linewidth=1)

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, growth_df["Growth_Rate"])):
                label_x = value + 0.1 if value >= 0 else value - 0.1
                ax1.text(
                    label_x,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}%",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontweight="bold",
                    fontsize=9,
                )

        # 2. RESEARCH QUESTION 2: Property Values by Sector
        print("  Creating Chart 2: Property Values Analysis...")

        # Get sector data with valid assessments
        sector_data = self.df[
            (self.df["SECTOR"].notna())
            & (self.df["MEDIAN_ASSESSMENT"].notna())
            & (self.df["RES_CNT"] > 0)
        ]
        sector_stats = (
            sector_data.groupby("SECTOR")
            .agg({"RES_CNT": "sum", "MEDIAN_ASSESSMENT": "mean"})
            .sort_values("MEDIAN_ASSESSMENT", ascending=False)
        )

        if len(sector_stats) > 0:
            # Create bar chart for property values
            colors = plt.cm.viridis(np.linspace(0, 1, len(sector_stats)))
            bars = ax2.bar(
                range(len(sector_stats)),
                sector_stats["MEDIAN_ASSESSMENT"] / 1000,
                color=colors,
                alpha=0.8,
                edgecolor="black",
            )

            ax2.set_xticks(range(len(sector_stats)))
            ax2.set_xticklabels(
                sector_stats.index, rotation=45, ha="right", fontsize=10
            )
            ax2.set_ylabel(
                "Average Assessment ($1000s)", fontsize=12, fontweight="bold"
            )
            ax2.set_title(
                "Q2: Average Property Values by Sector", fontsize=14, fontweight="bold"
            )
            ax2.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, sector_stats["MEDIAN_ASSESSMENT"]):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"${value / 1000:.0f}K",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9,
                )

        # 3. RESEARCH QUESTION 3: Inner-City vs Suburban Growth
        print("  Creating Chart 3: Area Type Comparison...")

        # Calculate area type data
        area_data = self.df[self.df["AREA_TYPE"].notna()]
        area_yearly = area_data.groupby(
            ["AREA_TYPE", area_data.index.get_level_values("YEAR")]
        ).agg({"RES_CNT": "sum"})

        area_colors = {"Inner-City": "#e74c3c", "Suburban": "#3498db"}
        markers = {"Inner-City": "o", "Suburban": "s"}
        years = [2016, 2017]

        growth_data = {}
        max_pop = 0

        for area_type in area_yearly.index.get_level_values("AREA_TYPE").unique():
            if (area_type, 2016) in area_yearly.index and (
                area_type,
                2017,
            ) in area_yearly.index:
                pop_2016 = area_yearly.loc[(area_type, 2016), "RES_CNT"]
                pop_2017 = area_yearly.loc[(area_type, 2017), "RES_CNT"]
                pop_data = [pop_2016, pop_2017]

                # Calculate growth metrics
                growth_rate = (pop_2017 - pop_2016) / pop_2016 * 100
                change_abs = pop_2017 - pop_2016

                growth_data[area_type] = {
                    "pop_data": pop_data,
                    "growth_rate": growth_rate,
                    "change_abs": change_abs,
                    "pop_2016": pop_2016,
                    "pop_2017": pop_2017,
                }

                max_pop = max(max_pop, max(pop_data))

                # Plot line with enhanced styling
                ax3.plot(
                    years,
                    pop_data,
                    marker=markers.get(area_type, "^"),
                    linewidth=4,
                    markersize=14,
                    label=f"{area_type}",
                    color=area_colors.get(area_type, "#2E86AB"),
                    markeredgecolor="white",
                    markeredgewidth=2,
                    alpha=0.8,
                )

        # Add data labels with better positioning
        label_offset = max_pop * 0.02  # Dynamic offset based on data scale

        for area_type, data in growth_data.items():
            color = area_colors.get(area_type, "#2E86AB")

            for i, (year, pop) in enumerate(zip(years, data["pop_data"])):
                # Position labels above points with consistent offset
                ax3.text(
                    year,
                    pop + label_offset,
                    f"{pop:,}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=11,
                    color=color,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor=color,
                        linewidth=1,
                    ),
                )

        # Add growth rate annotations in a clean box
        annotation_y = max_pop * 0.85  # Position in upper area of chart

        # Create a summary box for growth information
        growth_text_lines = []
        for area_type, data in growth_data.items():
            change_direction = "increase" if data["growth_rate"] > 0 else "decrease"
            growth_text_lines.append(
                f"{area_type}: {data['growth_rate']:+.1f}% ({data['change_abs']:+,} people)"
            )

        growth_summary = "Population Change (2016→2017):\n" + "\n".join(
            growth_text_lines
        )

        ax3.text(
            0.98,
            0.98,
            growth_summary,
            transform=ax3.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="lightyellow",
                alpha=0.9,
                edgecolor="orange",
                linewidth=2,
            ),
            fontweight="bold",
        )

        # Enhanced formatting
        ax3.set_xlabel("Year", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Total Population", fontsize=12, fontweight="bold")
        ax3.set_title(
            "Q3: Inner-City vs Suburban Population Growth Comparison",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax3.legend(
            fontsize=12, frameon=True, fancybox=True, shadow=True, loc="upper left"
        )
        ax3.grid(alpha=0.3, linestyle="--")
        ax3.set_xticks(years)
        ax3.set_xlim(2015.8, 2017.2)  # Add some padding on x-axis
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

        # Set y-axis limits with some padding
        y_min = min([min(data["pop_data"]) for data in growth_data.values()]) * 0.95
        y_max = max([max(data["pop_data"]) for data in growth_data.values()]) * 1.15
        ax3.set_ylim(y_min, y_max)

        # 4. Summary Statistics
        print("  Creating Chart 4: Dataset Summary...")

        # Create summary metrics
        total_records = len(self.df)
        valid_records = len(
            self.df[(self.df["SECTOR"].notna()) & (self.df["RES_CNT"] > 0)]
        )
        communities = self.df.index.get_level_values("COMMUNITY_NAME").nunique()

        # Sector distribution pie chart
        if self.df["SECTOR"].notna().any():
            sector_counts = self.df["SECTOR"].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(sector_counts)))

            wedges, texts, autotexts = ax4.pie(
                sector_counts.values,
                labels=sector_counts.index,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )

            ax4.set_title(
                "Dataset Distribution by Sector", fontsize=14, fontweight="bold"
            )

            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")
                autotext.set_fontsize(10)

        # Add summary statistics as text
        summary_text = (
            f"Dataset Summary:\n"
            f"• Total Records: {total_records:,}\n"
            f"• Valid Records: {valid_records:,}\n"
            f"• Communities: {communities}\n"
            f"• Years: 2016-2017"
        )

        ax4.text(
            1.3,
            0.5,
            summary_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
            fontweight="bold",
        )

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        # Save the figure
        filename = "calgary_housing_research_analysis.png"
        plt.savefig(
            filename, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        print(f"  Visualization saved as: {filename}")

        plt.show()

    def export_to_excel(self):
        """
        Export the complete dataset to an Excel file with proper formatting.

        Parameters:
            None
        Returns:
            None
        """
        # Create Excel writer object
        filename = "calgary_housing_complete_analysis.xlsx"

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            # Export main dataset
            self.df.to_excel(writer, sheet_name="Complete Dataset")

            # Add summary statistics sheet
            summary_stats = self.df.describe()
            summary_stats.to_excel(writer, sheet_name="Summary Statistics")

            # Add pivot table sheet
            pivot_data = self.df[self.df["SECTOR"].notna()].reset_index()
            pivot = pd.pivot_table(
                pivot_data,
                values=["MEDIAN_ASSESSMENT", "RES_CNT", "VACANCY_RATE"],
                index="SECTOR",
                columns="YEAR",
                aggfunc="mean",
            )
            pivot.to_excel(writer, sheet_name="Sector Analysis Pivot")

            # Add high-value communities sheet
            high_value_mask = (self.df["MEDIAN_ASSESSMENT"] > 600000) & self.df[
                "MEDIAN_ASSESSMENT"
            ].notna()
            high_value = self.df[high_value_mask]
            high_value.to_excel(writer, sheet_name="High Value Communities")

            # Add missing values summary sheet
            missing_summary = pd.DataFrame(
                {
                    "Column": self.df.columns,
                    "Missing Count": [
                        self.df[col].isna().sum() for col in self.df.columns
                    ],
                    "Missing %": [
                        (self.df[col].isna().sum() / len(self.df) * 100)
                        for col in self.df.columns
                    ],
                }
            )
            missing_summary = missing_summary[missing_summary["Missing Count"] > 0]
            missing_summary.to_excel(
                writer, sheet_name="Missing Values Summary", index=False
            )

        print(f"\nComplete dataset exported to '{filename}'")
        print(f"  - Main dataset: {len(self.df)} records")
        print(
            "  - 5 sheets included: Complete Dataset, Summary Statistics, Sector Analysis, High Value Communities, Missing Values"
        )


def main():
    """
    Main program entry point. Orchestrates the complete analysis workflow.

    Executes the following analysis steps:
    1. Data loading and missing value handling
    2. User input collection for community selection
    3. Comprehensive statistical analysis (aggregation, masking, groupby, pivot tables)
    4. Three research question analyses (sector growth, development pressure, urban patterns)
    5. Research visualization generation
    6. Excel export with multiple sheets

    Missing Value Handling:
    - WARD data missing for ~8% of records (displays as N/A)
    - VACANCY_RATE missing when total dwellings is 0 (excluded from analysis)
    - Assessment values missing when population is 0 (excluded from averages)
    - All aggregations and visualizations handle missing values appropriately
    """
    print("=" * 60)
    print("Calgary Housing and Demographics Analysis System")
    print("ENSF 692 Spring 2025 - Final Project")
    print("=" * 60)

    # Initialize analyzer
    analyzer = CalgaryHousingAnalyzer()

    # Handle missing values
    analyzer.handle_missing_values()

    # Display overall statistics
    print("\nDataset Overview:")
    print(analyzer.df.describe())

    # Get user input
    print("\n" + "-" * 60)
    print("User Input Section")
    print("-" * 60)
    community, year = analyzer.get_user_input()

    # Display community information
    analyzer.display_community_info(community, year)

    # Perform comprehensive analysis
    analyzer.perform_analysis()

    # Create visualizations
    print("\n" + "-" * 60)
    print("Creating Visualizations...")
    print("-" * 60)
    analyzer.create_research_visualizations()

    # Export to Excel
    print("\n" + "-" * 60)
    print("Exporting Results...")
    print("-" * 60)
    analyzer.export_to_excel()

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nFiles generated:")
    print("  1. calgary_housing_research_analysis.png - Research visualization plots")
    print("  2. calgary_housing_complete_analysis.xlsx - Complete dataset and analysis")
    print("\nThank you for using the Calgary Housing Analysis System!")


if __name__ == "__main__":
    main()
