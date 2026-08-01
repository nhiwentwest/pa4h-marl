import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Find the downloaded csv
csv_files = glob.glob('downloaded_results/gang_network_full_300eps_*/train.csv')
if not csv_files:
    print("CSV not found!")
    exit(1)

csv_path = csv_files[0]
df = pd.read_csv(csv_path)

os.makedirs('downloaded_results/plots', exist_ok=True)

# 1. Reward
plt.figure(figsize=(10, 5))
plt.plot(df['ep'], df['team_reward'], label='Team Reward')
plt.title('Team Reward over Episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.savefig('downloaded_results/plots/reward.png')
plt.close()

# 2. Rack Violations
plt.figure(figsize=(10, 5))
plt.plot(df['ep'], df['rack_viol'], label='Rack Violations', color='red')
plt.title('Rack Violations over Episodes')
plt.xlabel('Episode')
plt.ylabel('Violations')
plt.grid(True)
plt.savefig('downloaded_results/plots/rack_viol.png')
plt.close()

# 3. Jobs (Admitted vs Completed vs Preempted)
plt.figure(figsize=(10, 5))
plt.plot(df['ep'], df['admitted'], label='Admitted')
plt.plot(df['ep'], df['completed'], label='Completed')
plt.plot(df['ep'], df['preempted'], label='Preempted', color='red')
plt.title('Job Metrics over Episodes')
plt.xlabel('Episode')
plt.ylabel('Jobs')
plt.legend()
plt.grid(True)
plt.savefig('downloaded_results/plots/jobs.png')
plt.close()

# 4. Agent Accuracies
plt.figure(figsize=(10, 5))
plt.plot(df['ep'], df['a2_choice_accuracy'], label='A2 Accuracy')
plt.plot(df['ep'], df['a3_choice_accuracy'], label='A3 Accuracy')
plt.plot(df['ep'], df['a4_placement_rate'], label='A4 Placement Rate')
plt.title('Agent Performance over Episodes')
plt.xlabel('Episode')
plt.ylabel('Rate/Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('downloaded_results/plots/agent_accuracies.png')
plt.close()

print("Saved plots to downloaded_results/plots/")
