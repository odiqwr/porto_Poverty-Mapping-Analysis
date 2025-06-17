# Load Dataset
df = pd.read_csv('Data Processing.csv')
df.head()

# Checking Structure Data
df.info()
numeric=['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num=df.select_dtypes(include=numeric)
df_cat=df.select_dtypes(include='object')

# Analysis Descriptive Statistics
describeNum = df.describe(include =['float64', 'int64', 'float', 'int'])      # for numeric analysis
describeNum.T.style.background_gradient(cmap='viridis',low=0.2,high=0.1)      # for show analysis results
describeNumCat = df.describe(include=["O"])                                   # for object analysis
describeNumCat.T.style.background_gradient(cmap='viridis',low=0.2,high=0.1)

# Checking Missing Values
null=pd.DataFrame(df.isnull().sum(),columns=["Null Values"])                  # to calculate missing value columns
null["% Missing Values"]=(df.isna().sum()/len(df)*100)                        # to calculate percentage of missing values
null = null[null["% Missing Values"] > 0]                                     # to filter column with a percentage of missing values above zero "0"
null.style.background_gradient(cmap='viridis',low =0.2,high=0.1)              # to show table of column missing values

# Handling Missing Values
df['Unnamed: 6'] = df.apply(lambda row: (row.city) + " " + (row.state_code) + " " +(row.zip_code)  , axis = 1)      # to handle missing values ​​in column Unnamed: 6
df['closed_at'] = df['closed_at'].fillna(value="31/12/2013")                                                        # to handle missing values ​​in column closed_at, with fixed date assumption
df['age_first_milestone_year'] = df['age_first_milestone_year'].fillna(value="0")                                   # to handle missing values ​​in column age_first_milestone_year, with assumption no milestone
df['age_last_milestone_year'] = df['age_last_milestone_year'].fillna(value="0")                                     # to handle missing values ​​in column age_last_milestone_year, with assumption no milestone
df.drop(["state_code.1"], axis=1, inplace=True)                                                                     # to handle missing values ​​in column state_code.1, because values in column state_code.1 = state_code

# Graphic Approach
# heatmap correlation
df_numeric = df.drop(columns=['Province'], errors='ignore').select_dtypes(include='number')      # to exclude the 'province' column
corr_matrix = df_numeric.corr()                                                                  # to calculate correlation
mean_corr = corr_matrix.abs().mean().sort_values(ascending=False)                                # to calculate the absolute average correlation of all features
top_features = mean_corr.head(10).index                                                          # to take the top 10 features with the largest correlation
top_corr = corr_matrix.loc[top_features, top_features]                                           # to create a subset of the correlation matrix of the top 10 features

# displaying correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(top_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Heatmap Korelasi 10 Besar Fitur (Tanpa Province)")                                  

# scatter plot for correlation between two features
fig, ax = plt.subplots()
_ = plt.scatter(x=df['Persentase Pelajar_Bekerja'], y=df['Pekerja Informal'], edgecolors="#000000", linewidths=0.5)
_ = ax.set(xlabel="Persentase Pelajar_Bekerja", ylabel="Pekerja Informal")]

# box plots for outlier detection
featuresNum = ['Persentase Pelajar_Bekerja','Persentase Pelajar_Belajar','Persentase Tidak/Belum Sekolah','Pekerja Formal (%)','Pekerja Informal (%)','Jumlah Sekolah Atas dan Kejuruan','Jumlah Kampus (Unit)','Jumlah Industri Kecil n Mikro (Unit)','Jumlah Industri Besar n Sedang (Unit)','Upah Minimum Provinsi']
plt.figure(figsize=(15, 7))
for i in range(0, len(featuresNum)):
    plt.subplot(1, len(featuresNum), i+1)
    sns.boxplot(y=df[featuresNum[i]], color='green', orient='v')
    plt.tight_layout()
