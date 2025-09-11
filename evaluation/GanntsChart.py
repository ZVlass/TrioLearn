
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Re-define task list with Start Week and Duration
tasks = [
    {"Task": "Background Research & Related Works", "Start Week": 1, "Duration (weeks)": 3},
    {"Task": "Reading & Literature Review", "Start Week": 1, "Duration (weeks)": 3},
    {"Task": "Project Design", "Start Week": 3, "Duration (weeks)": 2},
    {"Task": "Project Proposal", "Start Week": 4, "Duration (weeks)": 1},
    {"Task": "Data Collection & Integration", "Start Week": 4, "Duration (weeks)": 2},
    {"Task": "Text Preprocessing & Cleaning", "Start Week": 4, "Duration (weeks)": 2},
    {"Task": "Embedding & Topic Modelling", "Start Week": 5, "Duration (weeks)": 2},
    {"Task": "System Architecture Setup", "Start Week": 6, "Duration (weeks)": 1},
    {"Task": "Plan evaluation", "Start Week": 7, "Duration (weeks)": 1},
    {"Task": "Prototype Recommender Development", "Start Week": 8, "Duration (weeks)": 2},
    {"Task": "Recommendation Algorithm Implementation", "Start Week": 9, "Duration (weeks)": 1},
    {"Task": "User Interface (Frontend/API)", "Start Week": 10, "Duration (weeks)": 2},
    {"Task": "Evaluation (Offline & Intrinsic)", "Start Week": 11, "Duration (weeks)": 2},
    {"Task": "Testing & A/B Testing", "Start Week": 12, "Duration (weeks)": 2},
    {"Task": "Final Report & Testing", "Start Week": 13, "Duration (weeks)": 2},
    {"Task": "Final Touches & Buffer", "Start Week": 14, "Duration (weeks)": 1},
    {"Task": "Deployment", "Start Week": 15, "Duration (weeks)": 1},
]

# Create DataFrame
df = pd.DataFrame(tasks)

# Sort background and literature review tasks to top
top_tasks = ["Background Research & Related Works", "Reading & Literature Review"]
df_sorted = pd.concat([
    df[df["Task"].isin(top_tasks)],
    df[~df["Task"].isin(top_tasks)].sort_values("Start Week")
], ignore_index=True)

# Plot Gantt chart with Week labels
fig, ax = plt.subplots(figsize=(12, 8))

for i, row in df_sorted.iterrows():
    ax.barh(row["Task"], row["Duration (weeks)"], left=row["Start Week"] - 1, height=0.4)

# Week ticks
ax.set_xticks(range(16))
ax.set_xticklabels([f"Week {i+1}" for i in range(16)])
plt.xlabel("Project Weeks")
plt.title("TrioLearn Project Gantt Chart (Using Week Labels)")
plt.tight_layout()
plt.grid(True, axis='x')
plt.gca().invert_yaxis()
plt.show()
