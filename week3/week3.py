import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns


url = "https://data.cityofnewyork.us/api/views/6fi9-q3ta/rows.csv?accessType=DOWNLOAD"

df = pd.read_csv(url)

df['hour_beginning'] = pd.to_datetime(df['hour_beginning'])

df['DayOfWeek'] = df['hour_beginning'].dt.day_name()

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

df_filtered  = df[df['DayOfWeek'].isin(days)]

weekday_counts = df_filtered.groupby('DayOfWeek')['Pedestrians'].sum()

weekday_counts = df_filtered.groupby('DayOfWeek')['Pedestrians'].sum()

plt.figure(figsize=(10, 6))
plt.plot(weekday_counts.index, weekday_counts.values, marker='o')
plt.title("Total Pedestrian Counts by Weekday")
plt.xlabel("Day of the Week")
plt.ylabel("Total Pedestrian Counts")
plt.grid(True)
plt.tight_layout()
plt.show()

df_2019 = df[(df['location'] == 'Brooklyn Bridge') & (df['hour_beginning'].dt.year == 2019)]


weather_encoded = pd.get_dummies(df_2019['weather_summary'])


df_weather = pd.concat([df_2019[['Pedestrians']], weather_encoded], axis=1)


correlation_matrix = df_weather.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation between Weather Conditions and Pedestrian Count (Brooklyn Bridge, 2019)")
plt.tight_layout()
plt.show()


def get_time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df['TimeOfDay'] = df['hour_beginning'].dt.hour.apply(get_time_of_day)

time_counts = df.groupby('TimeOfDay')['Pedestrians'].sum()

time_counts = time_counts.reindex(['Morning', 'Afternoon', 'Evening', 'Night'])

plt.figure(figsize=(8, 5))
plt.bar(time_counts.index, time_counts.values, color='skyblue')
plt.title("Pedestrian Activity by Time of Day (All Locations)")
plt.xlabel("Time of Day")
plt.ylabel("Total Pedestrians")
plt.tight_layout()
plt.show()