import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV data
df = pd.read_csv("SHD_ResultsTotal.csv")  # Update with actual filename

# Set up the figure
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Create the boxplot
sns.boxplot(
    data=df, 
    x="p (Variables)", 
    y="SHD", 
    hue="Noise Type", 
    dodge=True,  # Ensures boxplots are side by side
    showfliers=True,  # Show outliers
    boxprops={'alpha': 0.5}  # Make boxplots transparent to see overlap
)

# Customize the plot
plt.title("Overlapping Boxplots of SHD by Noise Type")
plt.xlabel("Number of Variables (p)")
plt.ylabel("Structural Hamming Distance (SHD)")
plt.legend(title="Noise Type")

# Show the plot
plt.show()

g = sns.FacetGrid(df, col="Variance Type", height=5, aspect=1.2)
g.map_dataframe(
    sns.boxplot, x="p (Variables)", y="SHD", hue="Noise Type", dodge=True, showfliers=True
)
g.add_legend()
plt.show()