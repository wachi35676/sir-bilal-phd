import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_hierarchical_balance(file_path):
    """
    Analyze dataset balance using a hierarchical approach.
    
    Parameters:
    file_path (str): Path to the Excel file containing the dataset
    
    Returns:
    dict: Results of hierarchical analysis
    """
    # Load the data
    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)
    
    # Print basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Total number of reviews: {len(df)}")
    
    # Define the hierarchical structure
    hierarchy = {
        # Level 1: Main categories
        "mainCategories": {
            "Usability": ["Learnability ", "Memorability ", "User Error Protection", 
                         "Operability", "Accessability", "Satisfaction", "Efficiency", "Effectiveness"],
            "Security": ["Confidentiality ", "Integrity ", "Availability ", "Authenticity ", 
                        "Accountability", "Non repudation", "Traceability", "Authorization", "Resiliance "]
        },
        
        # Level 2: Sub-categories
        "subCategories": {
            "Interaction": ["Learnability ", "Memorability ", "Operability"],
            "User Experience": ["Satisfaction", "Accessability"],
            "Performance": ["Efficiency", "Effectiveness"],
            "Error Handling": ["User Error Protection"],
            "Data Protection": ["Confidentiality ", "Integrity ", "Authenticity "],
            "Access Control": ["Authorization"],
            "System Health": ["Availability ", "Resiliance "],
            "Accountability": ["Accountability", "Non repudation", "Traceability"]
        }
    }
    
    # Function to count reviews with at least one principle in the category
    def count_reviews_with_category(dataframe, principles):
        """Count reviews that have at least one principle from the given list."""
        mask = dataframe[principles].any(axis=1)
        return mask.sum()
    
    # Calculate imbalance ratio
    def calculate_imbalance_ratio(counts_dict):
        """Calculate the ratio between the most and least common categories."""
        values = [count for count in counts_dict.values() if count > 0]
        if not values:
            return 0
        return max(values) / min(values)
    
    # Initialize results dictionary
    results = {
        "level1": {},
        "level2": {},
        "level3": {"Usability": {}, "Security": {}},
        "review_distribution": {},
        "imbalance_ratios": {}
    }
    
    # Level 1 analysis: Main categories
    print("\n--- LEVEL 1: MAIN CATEGORIES ---")
    for category, principles in hierarchy["mainCategories"].items():
        count = count_reviews_with_category(df, principles)
        percentage = (count / len(df)) * 100
        results["level1"][category] = {"count": count, "percentage": percentage}
        print(f"{category}: {count} reviews ({percentage:.2f}%)")
    
    # Calculate imbalance ratio for Level 1
    l1_counts = {k: v["count"] for k, v in results["level1"].items()}
    results["imbalance_ratios"]["level1"] = calculate_imbalance_ratio(l1_counts)
    print(f"Imbalance ratio at Level 1: {results['imbalance_ratios']['level1']:.2f}")
    
    # Level 2 analysis: Sub-categories
    print("\n--- LEVEL 2: SUB-CATEGORIES ---")
    for subcategory, principles in hierarchy["subCategories"].items():
        count = count_reviews_with_category(df, principles)
        percentage = (count / len(df)) * 100
        results["level2"][subcategory] = {"count": count, "percentage": percentage}
        print(f"{subcategory}: {count} reviews ({percentage:.2f}%)")
    
    # Calculate imbalance ratio for Level 2
    l2_counts = {k: v["count"] for k, v in results["level2"].items()}
    results["imbalance_ratios"]["level2"] = calculate_imbalance_ratio(l2_counts)
    print(f"Imbalance ratio at Level 2: {results['imbalance_ratios']['level2']:.2f}")
    
    # Level 3 analysis: Original principles
    print("\n--- LEVEL 3: ORIGINAL PRINCIPLES ---")
    for category, principles in hierarchy["mainCategories"].items():
        print(f"\n{category} principles:")
        for principle in principles:
            # Check if column exists
            if principle not in df.columns:
                print(f"  Warning: Column '{principle}' not found in the dataset")
                results["level3"][category][principle] = {"count": 0, "percentage": 0}
                continue
                
            # Handle mixed types by converting to numeric if possible
            try:
                # Try to convert column to numeric, errors='coerce' will convert non-numeric values to NaN
                numeric_values = pd.to_numeric(df[principle], errors='coerce')
                # Replace NaN with 0 and sum
                count = numeric_values.fillna(0).sum()
            except:
                # If conversion completely fails, try counting True/1 values
                try:
                    # For boolean columns or columns with 0/1 values
                    count = df[principle].astype(bool).sum()
                except:
                    # If all else fails, count non-empty strings as True
                    count = df[principle].astype(str).str.strip().apply(lambda x: x != '').sum()
                
            percentage = (count / len(df)) * 100
            results["level3"][category][principle] = {"count": count, "percentage": percentage}
            print(f"  {principle.strip()}: {count} reviews ({percentage:.2f}%)")
        
        # Calculate imbalance ratio for this category's principles
        principle_counts = {k: v["count"] for k, v in results["level3"][category].items()}
        # Filter out zero counts to avoid division by zero
        non_zero_counts = {k: v for k, v in principle_counts.items() if v > 0}
        if non_zero_counts:
            results["imbalance_ratios"][f"level3_{category}"] = calculate_imbalance_ratio(non_zero_counts)
            print(f"  Imbalance ratio for {category} principles: {results['imbalance_ratios'][f'level3_{category}']:.2f}")
        else:
            results["imbalance_ratios"][f"level3_{category}"] = 0
            print(f"  No non-zero counts for {category} principles")
    
    # Review distribution analysis
    usability_principles = hierarchy["mainCategories"]["Usability"]
    security_principles = hierarchy["mainCategories"]["Security"]
    
    has_usability = df[usability_principles].any(axis=1)
    has_security = df[security_principles].any(axis=1)
    
    both_count = (has_usability & has_security).sum()
    either_count = (has_usability | has_security).sum()
    neither_count = (~(has_usability | has_security)).sum()
    
    results["review_distribution"]["both"] = {"count": both_count, "percentage": (both_count/len(df))*100}
    results["review_distribution"]["either"] = {"count": either_count, "percentage": (either_count/len(df))*100}
    results["review_distribution"]["neither"] = {"count": neither_count, "percentage": (neither_count/len(df))*100}
    
    print("\n--- REVIEW DISTRIBUTION ---")
    print(f"Reviews with both categories: {both_count} ({results['review_distribution']['both']['percentage']:.2f}%)")
    print(f"Reviews with either category: {either_count} ({results['review_distribution']['either']['percentage']:.2f}%)")
    print(f"Reviews with neither category: {neither_count} ({results['review_distribution']['neither']['percentage']:.2f}%)")
    
    # Visualize the results
    plot_hierarchical_results(results, hierarchy)
    
    return results

def plot_hierarchical_results(results, hierarchy):
    """Create visualizations of the hierarchical analysis results."""
    # Set up the figure
    plt.figure(figsize=(15, 12))
    
    # 1. Level 1 Categories (Pie chart)
    plt.subplot(2, 2, 1)
    level1_data = {k: v["count"] for k, v in results["level1"].items()}
    plt.pie(level1_data.values(), labels=level1_data.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('Level 1: Main Categories')
    
    # 2. Level 2 Sub-categories (Bar chart)
    plt.subplot(2, 2, 2)
    subcategories = list(results["level2"].keys())
    values = [v["count"] for v in results["level2"].values()]
    plt.barh(subcategories, values)
    plt.xlabel('Number of Reviews')
    plt.title('Level 2: Sub-categories')
    
    # 3. Usability principles (Bar chart)
    plt.subplot(2, 2, 3)
    usability_principles = [p.strip() for p in hierarchy["mainCategories"]["Usability"]]
    usability_values = [results["level3"]["Usability"][p]["count"] for p in hierarchy["mainCategories"]["Usability"]]
    plt.barh(usability_principles, usability_values)
    plt.xlabel('Number of Reviews')
    plt.title('Usability Principles')
    
    # 4. Security principles (Bar chart)
    plt.subplot(2, 2, 4)
    security_principles = [p.strip() for p in hierarchy["mainCategories"]["Security"]]
    security_values = [results["level3"]["Security"][p]["count"] for p in hierarchy["mainCategories"]["Security"]]
    plt.barh(security_principles, security_values)
    plt.xlabel('Number of Reviews')
    plt.title('Security Principles')
    
    plt.tight_layout()
    plt.savefig('hierarchical_balance_analysis.png')
    print("\nVisualization saved as 'hierarchical_balance_analysis.png'")

def create_balanced_dataset(file_path, output_path=None, method="hierarchical", level=2):
    """
    Create a balanced dataset using the hierarchical approach.
    
    Parameters:
    file_path (str): Path to the Excel file containing the dataset
    output_path (str): Path to save the balanced dataset (if None, will not save)
    method (str): Method to use for balancing ('hierarchical' or 'random')
    level (int): Hierarchical level to balance at (1 or 2)
    
    Returns:
    pd.DataFrame: Balanced dataset
    """
    # Load the data
    df = pd.read_excel(file_path)
    
    # Define the hierarchical structure
    hierarchy = {
        # Level 1: Main categories
        "mainCategories": {
            "Usability": ["Learnability ", "Memorability ", "User Error Protection", 
                         "Operability", "Accessability", "Satisfaction", "Efficiency", "Effectiveness"],
            "Security": ["Confidentiality ", "Integrity ", "Availability ", "Authenticity ", 
                        "Accountability", "Non repudation", "Traceability", "Authorization", "Resiliance "]
        },
        
        # Level 2: Sub-categories
        "subCategories": {
            "Interaction": ["Learnability ", "Memorability ", "Operability"],
            "User Experience": ["Satisfaction", "Accessability"],
            "Performance": ["Efficiency", "Effectiveness"],
            "Error Handling": ["User Error Protection"],
            "Data Protection": ["Confidentiality ", "Integrity ", "Authenticity "],
            "Access Control": ["Authorization"],
            "System Health": ["Availability ", "Resiliance "],
            "Accountability": ["Accountability", "Non repudation", "Traceability"]
        }
    }
    
    # Balance at selected level
    if method == "hierarchical":
        if level == 1:
            # Balance at main category level
            categories = hierarchy["mainCategories"]
            balanced_df = balance_by_categories(df, categories)
        elif level == 2:
            # Balance at sub-category level
            categories = hierarchy["subCategories"]
            balanced_df = balance_by_categories(df, categories)
        else:
            raise ValueError("Level must be 1 or 2 for hierarchical balancing")
    else:
        # Random balancing (not recommended but included for comparison)
        # This simply undersamples all principles to match the least common one
        min_count = min([df[col].sum() for col in df.columns if df[col].dtype == bool or df[col].isin([0, 1]).all()])
        balanced_df = pd.DataFrame()
        for col in df.columns:
            if df[col].dtype == bool or df[col].isin([0, 1]).all():
                # Get reviews where this principle is present
                principle_df = df[df[col] == 1]
                # Sample to min_count or take all if less than min_count
                if len(principle_df) > min_count:
                    principle_df = principle_df.sample(min_count, random_state=42)
                balanced_df = pd.concat([balanced_df, principle_df])
        
        # Remove duplicates
        balanced_df = balanced_df.drop_duplicates()
    
    # Save if output path is provided
    if output_path:
        balanced_df.to_excel(output_path, index=False)
        print(f"Balanced dataset saved to {output_path}")
    
    # Return the balanced dataframe
    return balanced_df

def balance_by_categories(df, categories):
    """
    Balance dataset by sampling equally from each category.
    
    Parameters:
    df (pd.DataFrame): Original dataset
    categories (dict): Dictionary mapping category names to lists of principles
    
    Returns:
    pd.DataFrame: Balanced dataset
    """
    # Find which reviews belong to each category
    category_masks = {}
    for category, principles in categories.items():
        category_masks[category] = df[principles].any(axis=1)
    
    # Find the minimum count across categories
    min_count = min([mask.sum() for mask in category_masks.values() if mask.sum() > 0])
    print(f"Balancing to {min_count} reviews per category")
    
    # Sample equally from each category
    balanced_df = pd.DataFrame()
    for category, mask in category_masks.items():
        category_df = df[mask]
        if len(category_df) > min_count:
            category_df = category_df.sample(min_count, random_state=42)
        balanced_df = pd.concat([balanced_df, category_df])
    
    # Remove duplicates (in case some reviews belong to multiple categories)
    balanced_df = balanced_df.drop_duplicates()
    
    return balanced_df

def main():
    """Main function to run the analysis."""
    # File path to your Excel file
    file_path = "30K Final Reviews for Thesis - Relabelled.xlsx"
    
    # Analyze the dataset
    print("Analyzing original dataset...")
    analyze_hierarchical_balance(file_path)
    
    # Create balanced datasets
    print("\nCreating balanced datasets...")
    
    # Level 1 balanced dataset
    print("\nBalancing at Level 1 (Main Categories):")
    balanced_l1 = create_balanced_dataset(file_path, "balanced_level1.xlsx", method="hierarchical", level=1)
    print(f"Balanced dataset shape: {balanced_l1.shape}")
    
    # Level 2 balanced dataset
    print("\nBalancing at Level 2 (Sub-Categories):")
    balanced_l2 = create_balanced_dataset(file_path, "balanced_level2.xlsx", method="hierarchical", level=2)
    print(f"Balanced dataset shape: {balanced_l2.shape}")
    
    # Analyze balanced datasets
    print("\nAnalyzing Level 1 balanced dataset...")
    analyze_hierarchical_balance("balanced_level1.xlsx")
    
    print("\nAnalyzing Level 2 balanced dataset...")
    analyze_hierarchical_balance("balanced_level2.xlsx")

if __name__ == "__main__":
    main()