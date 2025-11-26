import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_mock_data():
    """Generates synthetic building consumption data for testing."""
    start_date = pd.to_datetime('2024-12-01')
    end_date = pd.to_datetime('2024-12-31')
    

    timestamps = pd.date_range(start=start_date, end=end_date, freq='H', inclusive='left')
    
    all_data = []
    

    building_profiles = {
        'Main_Admin': {'base': 10, 'trend_factor': 0.1, 'noise_level': 5},
        'Library': {'base': 15, 'trend_factor': -0.05, 'noise_level': 8},
        'Dorm_South': {'base': 7, 'trend_factor': 0.2, 'noise_level': 3}
    }
    
    for name, profile in building_profiles.items():

        consumption = (
            profile['base'] + 
            np.sin(np.linspace(0, 2 * np.pi * 30, len(timestamps))) * profile['trend_factor'] * 30 + 
            np.random.normal(0, profile['noise_level'], len(timestamps))
        )
        consumption = np.maximum(0, consumption) 
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'kwh': consumption
        })
        df['building'] = name
        all_data.append(df)
        
    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined.set_index('timestamp', inplace=True)
    
    print("--- Data Ingestion (Mock) Complete ---")
    return df_combined



class MeterReading:
    """Models a single energy meter reading."""
    def __init__(self, timestamp, kwh):
        self.timestamp = pd.to_datetime(timestamp)
        self.kwh = kwh
    
class Building:
    """Models a campus building and manages its energy analysis."""
    def __init__(self, name, df):
        self.name = name
        self.df = df
        self.summary = {} 

    def calculate_total_consumption(self):
        """Calculates total consumption from the stored DataFrame."""
        return self.df['kwh'].sum()

    def generate_report(self):
        """Generates a text report snippet for the building."""
        report = f"--- Report for {self.name} ---\n"
        report += f"Total Consumption: {self.calculate_total_consumption():,.2f} kWh\n"
        if self.summary:
            report += f"Mean Daily Usage: {self.summary.get('mean', 0):,.2f} kWh\n"
            report += f"Max Daily Usage: {self.summary.get('max', 0):,.2f} kWh\n"
        return report

class BuildingManager:
    """Manages all Building objects and coordinates analysis."""
    def __init__(self, combined_df):
        self.combined_df = combined_df
        self.buildings = {}  
        self._initialize_buildings()

    def _initialize_buildings(self):
        """Creates Building objects from the combined DataFrame."""
        for name, df in self.combined_df.groupby('building'):
            self.buildings[name] = Building(name, df)
        print(f"Initialized {len(self.buildings)} Building objects.")




    def calculate_daily_totals(self):
        """Calculate daily total consumption using .resample('D')."""

        daily_totals = self.combined_df.groupby('building')['kwh'].resample('D').sum().reset_index()
        daily_totals.rename(columns={'kwh': 'daily_kwh'}, inplace=True)
        return daily_totals

    def calculate_weekly_aggregates(self):
        """Calculate weekly aggregates (sum/mean) using .resample('W')."""

        weekly_aggregates = self.combined_df.groupby('building')['kwh'].resample('W').agg(['sum', 'mean']).reset_index()
        weekly_aggregates.rename(columns={'sum': 'weekly_sum_kwh', 'mean': 'weekly_mean_kwh'}, inplace=True)
        return weekly_aggregates

    def building_wise_summary(self, daily_df):
        """Calculates and stores mean, min, max, total consumption per building."""
        if daily_df is None: return pd.DataFrame()
        

        summary = daily_df.groupby('building')['daily_kwh'].agg(['mean', 'min', 'max', 'sum'])
        summary.rename(columns={'sum': 'total_consumption'}, inplace=True)
        

        for name, row in summary.iterrows():
            if name in self.buildings:
                self.buildings[name].summary = row.to_dict()
                
        return summary.reset_index()




    def generate_dashboard(self, daily_df, weekly_df):
        """Generates a multi-chart dashboard visualization."""
        if daily_df.empty or weekly_df.empty:
            print("Cannot generate dashboard: Missing aggregated data.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 18))
        plt.suptitle('Campus Energy-Use Dashboard', fontsize=18, y=1.02)


        daily_pivot = daily_df.pivot(index='timestamp', columns='building', values='daily_kwh')
        daily_pivot.plot(ax=axes[0], kind='line', legend=True)
        axes[0].set_title('1. Daily Energy Consumption Trend (All Buildings)', fontsize=14)
        axes[0].set_ylabel('Consumption (kWh)')
        axes[0].grid(True, linestyle='--', alpha=0.6)


        avg_weekly = weekly_df.groupby('building')['weekly_mean_kwh'].mean().sort_values(ascending=False)
        avg_weekly.plot(ax=axes[1], kind='bar', color=['#4CAF50', '#2196F3', '#FF9800'])
        axes[1].set_title('2. Average Weekly Consumption Comparison', fontsize=14)
        axes[1].set_ylabel('Avg. Weekly Consumption (kWh)')
        axes[1].set_xlabel('Building')
        axes[1].tick_params(axis='x', rotation=0) 
        axes[1].grid(axis='y', linestyle='--', alpha=0.6)



        peak_daily_usage = daily_df.loc[daily_df['daily_kwh'].idxmax()]
        peak_time = peak_daily_usage['timestamp']
        peak_building = peak_daily_usage['building']
        peak_kwh = peak_daily_usage['daily_kwh']


        axes[2].scatter(daily_df['timestamp'], daily_df['daily_kwh'], 
                        c=daily_df['building'].astype('category').cat.codes, 
                        cmap='Pastel1', alpha=0.7, s=50)
        

        axes[2].scatter(peak_time, peak_kwh, color='red', s=150, 
                        label=f'Overall Peak Load ({peak_building})', marker='*', zorder=5)

        axes[2].set_title('3. Daily Consumption Distribution & Peak Load', fontsize=14)
        axes[2].set_ylabel('Daily Consumption (kWh)')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        

        output_path = OUTPUT_DIR / 'dashboard.png'
        plt.savefig(output_path)
        print(f"\nDashboard visualization saved to: {output_path}")


        return peak_time, peak_building, peak_kwh




    def generate_summary_report(self, daily_df, summary_table, peak_info):
        """Creates a concise written report (summary.txt)."""
        
        total_campus_consumption = daily_df['daily_kwh'].sum()
        

        highest_consumer_row = summary_table.sort_values(by='total_consumption', ascending=False).iloc[0]
        highest_consumer = highest_consumer_row['building']
        highest_consumer_total = highest_consumer_row['total_consumption']


        peak_time, peak_building, peak_kwh = peak_info


        avg_daily = daily_df['daily_kwh'].mean()

        report_content = f"""
        *** Executive Summary: Campus Energy Consumption Analysis ***
        
        Date Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        1. Total Campus Consumption: 
           {total_campus_consumption:,.2f} kWh (over the measured period)
           
        2. Highest Consuming Building: 
           {highest_consumer} (Total Consumption: {highest_consumer_total:,.2f} kWh)
           
        3. Average Daily Trend: 
           The campus consumes an average of {avg_daily:,.2f} kWh per day.
           
        4. Peak Load Event: 
           The highest recorded daily consumption was {peak_kwh:,.2f} kWh on {peak_time.strftime('%Y-%m-%d')},
           recorded in the {peak_building} building.
           
        5. Recommendation: 
           Focus energy-saving efforts and audits on the {highest_consumer} building.
        """
        

        print("\n" + report_content)
        

        output_path = OUTPUT_DIR / 'summary.txt'
        with open(output_path, 'w') as f:
            f.write(report_content)
        print(f"Executive summary saved to: {output_path}")

    def export_data(self, daily_df, summary_table):
        """Exports processed and summarized data to CSV files."""

        daily_df.to_csv(OUTPUT_DIR / 'cleaned_energy_data.csv', index=False)
        print(f"Cleaned daily data exported to: {OUTPUT_DIR / 'cleaned_energy_data.csv'}")


        summary_table.to_csv(OUTPUT_DIR / 'building_summary.csv', index=False)
        print(f"Summary stats exported to: {OUTPUT_DIR / 'building_summary.csv'}")


def main():
    """Main execution function to run the full dashboard pipeline."""
    

    combined_data_df = generate_mock_data()
    
    if combined_data_df.empty:
        print("Pipeline aborted due to failed data generation.")
        return

    manager = BuildingManager(combined_data_df)
    

    daily_totals_df = manager.calculate_daily_totals()
    weekly_aggregates_df = manager.calculate_weekly_aggregates()
    building_summary_table = manager.building_wise_summary(daily_totals_df)
    
    print("\nAggregation complete.")
    

    print("Example Building Reports (Task 3 Output):")
    for name, building in manager.buildings.items():
        print(building.generate_report())


    peak_info = manager.generate_dashboard(daily_totals_df, weekly_aggregates_df)


    manager.export_data(daily_totals_df, building_summary_table)
    manager.generate_summary_report(daily_totals_df, building_summary_table, peak_info)
    
    print("\nEnergy Consumption Dashboard Pipeline execution complete.")

if __name__ == "__main__":
    main()