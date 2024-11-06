import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Download the dataset from Kaggle
dataset_path = kagglehub.dataset_download("nehaprabhavalkar/indian-food-101")
print("Path to dataset files:", dataset_path)

# Step 2: Load the dataset into a DataFrame
df = pd.read_csv(f"{dataset_path}/indian_food.csv")

# Step 3: Define a function to plot the top 10 most common ingredients
def plot_common_ingredients(df):
    # Split ingredients by commas, strip spaces, count occurrences, and get the top 10
    ingredient_counts = df['ingredients'].str.split(',').explode().str.strip().value_counts().head(10)
    
    # Plot as a bar chart
    plt.figure(figsize=(10, 6))
    ingredient_counts.plot(kind='bar', color='skyblue')
    plt.title("Top 10 Most Common Ingredients in Indian Dishes")
    plt.xlabel("Ingredients")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Step 4: Define a function to plot the distribution of dishes by region as a pie chart
def plot_dishes_by_region(df):
    region_counts = df['region'].value_counts()
    
    plt.figure(figsize=(8, 8))
    plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Distribution of Dishes by Region in India")
    plt.show()

# Step 5: Define a function to create a scatter plot of cooking time vs. complexity
def plot_cooking_time_vs_complexity(df):
    # Create a 'complexity' measure by adding prep and cook times
    df['complexity'] = df['prep_time'] + df['cook_time']  
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='cook_time', y='complexity', color='orange')
    plt.title("Cooking Time vs. Complexity of Indian Dishes")
    plt.xlabel("Cooking Time (minutes)")
    plt.ylabel("Complexity")
    plt.show()

# Step 6: Define a function to show correlations among preparation, cooking, and total time
def plot_time_correlations(df):
    # Calculate total time for each dish
    df['total_time'] = df['prep_time'] + df['cook_time']
    
    # Select time-related columns and calculate correlations
    time_data = df[['prep_time', 'cook_time', 'total_time']]
    correlations = time_data.corr()
    
    # Plot the correlations as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Between Preparation, Cooking, and Total Times")
    plt.show()

# Step 7: Define a function to display basic statistics and correlations
def display_statistics(df):
    # Generate summary statistics
    summary_stats = df.describe()
    # Generate correlation matrix
    correlation_matrix = df.corr(numeric_only=True)
    
    # Display the results
    print("Summary Statistics:\n", summary_stats)
    print("\nCorrelation Matrix:\n", correlation_matrix)
    
    return summary_stats, correlation_matrix

# Step 8: Generate and display statistics and correlation matrix
summary_stats, correlation_matrix = display_statistics(df)

# Step 9: Call each function to generate the plots
plot_common_ingredients(df)
plot_dishes_by_region(df)
plot_cooking_time_vs_complexity(df)
plot_time_correlations(df)