



import numpy as np
import matplotlib.pyplot as plt


# Parameters
T_env = 25  # Ambient temperature (°C)
T_0 = 85    # Initial temperature (°C)
k = 0.0576  # Cooling constant (min^-1)
time = np.linspace(0, 30, 100)  # Time in minutes

# Temperature as a function of time
T = T_env + (T_0 - T_env) * np.exp(-k * time)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(time, T, label="Coffee Temperature", color="darkblue", linewidth=2)
plt.axhline(y=T_env, color="red", linestyle="--", label="Ambient Temperature")
plt.title("Temperature vs. Time (Newton's Law of Cooling)", fontsize=14)
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()



# Logarithmic transformation
log_diff = np.log(T - T_env)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(time, log_diff, label="Logarithmic Decay", color="green", linewidth=2)
plt.title("Logarithmic Transformation of Newton's Law of Cooling", fontsize=14)
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("ln(T - T_env)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()




# Calculate rate of cooling (numerical differentiation)
rate_of_cooling = -k * (T - T_env)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(T - T_env, rate_of_cooling, label="Rate of Cooling", color="purple", linewidth=2)
plt.title("Rate of Cooling vs. Temperature Difference", fontsize=14)
plt.xlabel("Temperature Difference (T - T_env)", fontsize=12)
plt.ylabel("Rate of Cooling (dT/dt)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()





# Example observed data
observed_time = np.array([0, 5, 10, 15, 20, 25, 30])
observed_temp = np.array([85, 70, 58.7, 50, 43.5, 39, 35])

# Predicted temperatures
predicted_temp = T_env + (T_0 - T_env) * np.exp(-k * observed_time)

# Residuals
residuals = observed_temp - predicted_temp

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(observed_time, residuals, color="orange", label="Residuals", s=50)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Residual Analysis", fontsize=14)
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Residuals (Observed - Predicted)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()






import plotly.graph_objects as go

# Create figure
fig = go.Figure()

# Add temperature decay line
fig.add_trace(go.Scatter(x=time, y=T, mode='lines', name='Coffee Temperature', line=dict(color='blue', width=2)))

# Add ambient temperature line
fig.add_trace(go.Scatter(x=time, y=[T_env]*len(time), mode='lines', name='Ambient Temperature', line=dict(color='red', dash='dash')))

# Customize layout
fig.update_layout(
    title="Temperature vs. Time (Newton's Law of Cooling)",
    xaxis_title="Time (minutes)",
    yaxis_title="Temperature (°C)",
    legend_title="Legend",
    template="plotly_white"
)

# Show plot
fig.show()













plt.figure(figsize=(8, 6))
plt.fill_between(time, T, T_env, color="lightblue", alpha=0.5, label="Cooling Gap")
plt.plot(time, T, label="Coffee Temperature", color="darkblue", linewidth=2)
plt.axhline(y=T_env, color="red", linestyle="--", label="Ambient Temperature")
plt.title("Temperature Decay with Cooling Gap Highlight", fontsize=14)
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()




import seaborn as sns

# Create a 2D grid for heatmap
time_grid = np.arange(0, 31, 1)  # Time in minutes
temperature_grid = T_env + (T_0 - T_env) * np.exp(-k * time_grid)

# Reshape for heatmap
data = np.array([temperature_grid])

# Plot
plt.figure(figsize=(12, 2))
sns.heatmap(data, annot=False, cmap="coolwarm", cbar=True, xticklabels=time_grid, yticklabels=["Temperature"])
plt.title("Temperature Change Over Time (Heatmap)", fontsize=14)
plt.xlabel("Time (minutes)", fontsize=12)
plt.show()





# Create slope field grid
time_vals = np.linspace(0, 30, 15)  # Time in minutes
temp_vals = np.linspace(25, 85, 15)  # Temperature in °C
time_mesh, temp_mesh = np.meshgrid(time_vals, temp_vals)

# Slope field (rate of cooling)
dT_dt = -k * (temp_mesh - T_env)

# Normalize slope vectors for display
dt = 1  # Time step
dT = dT_dt * dt
norm = np.sqrt(dt**2 + dT**2)
dt_normalized = dt / norm
dT_normalized = dT / norm

# Plot slope field
plt.figure(figsize=(8, 6))
plt.quiver(time_mesh, temp_mesh, dt_normalized, dT_normalized, color="darkblue")
plt.title("Slope Field: Rate of Cooling Over Time", fontsize=14)
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.grid(alpha=0.3)
plt.show()





# Create slope field grid
time_vals = np.linspace(0, 30, 15)  # Time in minutes
temp_vals = np.linspace(25, 85, 15)  # Temperature in °C
time_mesh, temp_mesh = np.meshgrid(time_vals, temp_vals)

# Slope field (rate of cooling)
dT_dt = -k * (temp_mesh - T_env)

# Normalize slope vectors for display
dt = 1  # Time step
dT = dT_dt * dt
norm = np.sqrt(dt**2 + dT**2)
dt_normalized = dt / norm
dT_normalized = dT / norm

# Plot slope field
plt.figure(figsize=(8, 6))
plt.quiver(time_mesh, temp_mesh, dt_normalized, dT_normalized, color="darkblue")
plt.title("Slope Field: Rate of Cooling Over Time", fontsize=14)
plt.xlabel("Time (minutes)", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.grid(alpha=0.3)
plt.show()








from mpl_toolkits.mplot3d import Axes3D

# Create data for surface
time_3d = np.linspace(0, 30, 100)  # Time in minutes
k_vals = np.linspace(0.03, 0.1, 50)  # Different cooling constants
time_3d, k_mesh = np.meshgrid(time_3d, k_vals)
temperature_3d = T_env + (T_0 - T_env) * np.exp(-k_mesh * time_3d)

# Plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(time_3d, k_mesh, temperature_3d, cmap="coolwarm", edgecolor='none')
ax.set_title("3D Surface: Temperature Over Time for Varying k", fontsize=14)
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Cooling Constant (k)")
ax.set_zlabel("Temperature (°C)")
plt.show()








import matplotlib.animation as animation

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], color="darkblue", linewidth=2)
ax.axhline(y=T_env, color="red", linestyle="--", label="Ambient Temperature")
ax.set_xlim(0, 30)
ax.set_ylim(20, 90)
ax.set_title("Dynamic Cooling Process", fontsize=14)
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid(alpha=0.3)

# Update function for animation
def update(frame):
    t = time[:frame]
    temp = T_env + (T_0 - T_env) * np.exp(-k * t)
    line.set_data(t, temp)
    return line,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(time), blit=True, interval=100)
plt.show()








gap = T - T_env  # Cooling gap

fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(time, T, label="Coffee Temperature", color="blue", linewidth=2)
ax1.axhline(y=T_env, color="red", linestyle="--", label="Ambient Temperature")
ax1.set_xlabel("Time (minutes)")
ax1.set_ylabel("Temperature (°C)")

ax2 = ax1.twinx()
ax2.bar(time, gap, alpha=0.3, color="orange", label="Cooling Gap")
ax2.set_ylabel("Cooling Gap (°C)")

fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
plt.title("Temperature Decay with Cooling Gap (Bar Chart)", fontsize=14)
plt.grid(alpha=0.3)
plt.show()
