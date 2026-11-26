df = pd.read_csv('/kaggle/input/daily-commodity-prices-india/csv/2024.csv')
df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'])

print(f"Shape: {df.shape}")
print(f"Date range: {df['Arrival_Date'].min()} to {df['Arrival_Date'].max()}")
df.head()
