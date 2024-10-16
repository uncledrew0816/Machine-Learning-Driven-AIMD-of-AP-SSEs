import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Load different Times fonts
font_path_regular = './times.ttf'  # Regular font
font_path_bold = './timesbd.ttf'   # Bold font
font_path_italic = './timesi.ttf'  # Italic font
font_path_bolditalic = './timebi.ttf'  # Bold Italic font

# Define font properties
regular_font = fm.FontProperties(fname=font_path_regular)
bold_font = fm.FontProperties(fname=font_path_bold)
italic_font = fm.FontProperties(fname=font_path_italic)
bolditalic_font = fm.FontProperties(fname=font_path_bolditalic)

# Load data
file_path = './ML_Data.xlsx'  # Data file path
data_df = pd.read_excel(file_path, sheet_name='Data')

# Drop the Defect column
data_cleaned_df = data_df.drop(columns=['Defect'])

# Calculate Pearson correlation coefficient
correlation_matrix = data_cleaned_df.corr()

# Save correlation matrix as CSV file
correlation_matrix.to_csv('./Pearson_Correlation_Matrix.csv')

# Save the heatmap data (i.e., correlation matrix)
correlation_matrix.to_csv('./Heatmap_Data.csv', header=True)

# Calculate correlations with diffusion coefficient and ionic conductivity separately
diffusion_coefficient_corr = correlation_matrix['Diffusion Coefficient (cm2/s)']
ionic_conductivity_corr = correlation_matrix['Ionic Conductivity (mS/cm)']

# Save these correlation coefficients as CSV files
diffusion_coefficient_corr.to_csv('./Diffusion_Coefficient_Correlation.csv', header=True)
ionic_conductivity_corr.to_csv('./Ionic_Conductivity_Correlation.csv', header=True)

# Disable LaTeX usage, use MathText for subscripts and italics
plt.rcParams['text.usetex'] = False

# Custom tick labels list, using MathText syntax for italics and subscripts
tick_labels = [
    r'$N_{Li}$', r'$N_{O}$', r'$N_{Cl}$', r'$N_{Br}$', r'$D_f$', r'$E_f$', r'$arepsilon$',
    r'$a$', r'$b$', r'$c$', r'$V$', r'$D_j$', r'$E_a$', r'$f$', r'$S_f$', r'$A$', r'$p$', r'$T$',
    r'$rac{ne^{2}}{k_B}$', r'$D$', r'$\sigma$'
]

# Plot heatmap and apply Times New Roman font
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, 
                      annot_kws={"fontproperties": regular_font})  # Set annotation to Times New Roman font

# Set main title with increased font size
plt.title('Pearson Correlation Coefficient Heatmap', fontproperties=bold_font, size=24)

# Remove X and Y axis labels
plt.xlabel('')
plt.ylabel('')

# Apply custom tick labels with increased font size
heatmap.set_xticklabels(tick_labels, fontproperties=regular_font, size=14, rotation=0)
heatmap.set_yticklabels(tick_labels, fontproperties=regular_font, size=14)

# Set colorbar font to Times New Roman
cbar = heatmap.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(labelsize=12)
cbar.ax.set_yticklabels(cbar.ax.get_yticks(), fontproperties=regular_font)

# Save heatmap as PNG file
plt.savefig('./Pearson_Correlation_Heatmap.png', dpi=300, bbox_inches='tight')

# Show heatmap
plt.show()