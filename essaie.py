#import python as py
import matplotlib.pyplot as plt
import pandas as pd

# Load the Excel file
df = pd.read_excel("Cinematique_data.xlsx")  # adjust path

# Rename columns for clarity (adjust if necessary)
df = df.rename(columns={
    "triangle[Mass Center]->X,Displacement(abs), RF(A002) [mm]": "X_disp_mm",
    "triangle[Mass Center]->Y,Displacement(abs), RF(A002) [mm]": "Y_disp_mm",
    "triangle[Mass Center]->Z,Displacement(abs), RF(A002) [mm]": "Z_disp_mm",
    "CAMBER": "Camber_deg",
    "TOE": "Toe_deg"
})

# Convert displacements from mm to meters (optional for SI consistency)
df["Z_disp_m"] = df["Z_disp_mm"] / 1000

# Plot Camber vs Z displacement
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(df["Z_disp_m"], df["Camber_deg"], color='blue')
plt.xlabel("Vertical Travel (m)")
plt.ylabel("Camber Angle (°)")
plt.title("Camber vs Vertical Travel")
plt.grid(True)

# Plot Toe vs Z displacement
plt.subplot(1, 2, 2)
plt.plot(df["Z_disp_m"], df["Toe_deg"], color='green')
plt.xlabel("Vertical Travel (m)")
plt.ylabel("Toe Angle (°)")
plt.title("Toe vs Vertical Travel")
plt.grid(True)

plt.tight_layout()
plt.show()
