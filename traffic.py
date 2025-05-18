import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data (replace with your CSV)
data = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-01', periods=100, freq='H'),
    'vehicle_count': range(100),
    'speed': [60 - x % 20 for x in range(100)]
})

# Plot
sns.lineplot(x='timestamp', y='speed', data=data)
plt.title('Traffic Speed Over Time')
plt.show()