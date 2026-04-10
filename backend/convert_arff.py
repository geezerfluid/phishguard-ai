from scipy.io import arff
import pandas as pd

# Load ARFF file
data, meta = arff.loadarff(r"C:\Users\Dell\Downloads\h3cgnj8hft-1\Phishing_Legitimate_full.arff")

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert bytes to string
df = df.apply(lambda col: col.map(lambda x: x.decode() if isinstance(x, bytes) else x))

# Save as CSV
df.to_csv("phishing_dataset.csv", index=False)

print("✅ Converted to CSV successfully!")