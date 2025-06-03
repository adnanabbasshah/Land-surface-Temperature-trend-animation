# Land-surface-Temperature-trend-animation



https://github.com/user-attachments/assets/c1e5c9aa-8d62-4ccc-9a3a-ff84e8ffbed6




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pymannkendall as mk
import plotly.graph_objects as go
from tqdm import tqdm
import imageio
import os

def analyze_trends(df, temperature_column):
    results = []
    for month in range(1, 13):
        monthly_data = df[df['month'] == month].sort_values('year')
        if len(monthly_data) > 3:
            test_result = mk.original_test(monthly_data[temperature_column].values)
            results.append({
                'Month': month,
                'Trend': test_result.trend,
                'P-Value': test_result.p,
                'Slope': getattr(test_result, 'slope', None)
            })
        else:
            results.append({
                'Month': month,
                'Trend': 'Insufficient data',
                'P-Value': None,
                'Slope': None
            })
    return pd.DataFrame(results).set_index('Month')

def create_trend_plots(final_df):
    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    
    day_results = analyze_trends(final_df, 'Day_LST')
    night_results = analyze_trends(final_df, 'Night_LST')

    fig, axes = plt.subplots(3, 4, figsize=(18, 15), sharex=True, sharey=True)
    plt.suptitle('Jhang District - LST Trends (2000-2024)', fontsize=20)

    for month in range(1, 13):
        ax = axes[(month-1)//4, (month-1)%4]
        month_data = final_df[final_df['month'] == month]

        sns.regplot(data=month_data, x='year', y='Day_LST',
                    scatter_kws={'alpha':0.3, 'color':'#F57C00'},
                    line_kws={'color':'red'}, ax=ax, label='Day')
        sns.regplot(data=month_data, x='year', y='Night_LST',
                    scatter_kws={'alpha':0.3, 'color':'#1976D2'},
                    line_kws={'color':'blue'}, ax=ax, label='Night')

        add_trend_annotation(ax, day_results, month, 'Day', 'red')
        add_trend_annotation(ax, night_results, month, 'Night', 'blue')

        ax.set_title(month_names[month])
        ax.set_ylabel('Temperature (°C)')
        if (month-1) // 4 == 2:
            ax.set_xlabel('Year')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 1))
    plt.savefig('jhang_lst_trends.jpg', dpi=300)
    plt.show()
![download (2)](https://github.com/user-attachments/assets/1cd90f45-01d2-4a14-a040-586895e0d179)

def add_trend_annotation(ax, results, month, label, color):
    if month in results.index and results.loc[month, 'Trend'] != 'Insufficient data':
        trend = results.loc[month, 'Trend']
        if trend != 'no trend':
            ax.text(0.05, 0.9 - (0.1 if label == 'Night' else 0), 
                   f'{label}: {trend}', 
                   transform=ax.transAxes, 
                   color=color)

def create_seasonal_animation(df, lst_column, output_name):
    monthly_data = df.groupby(['year', 'month'])[lst_column].mean().reset_index()
    monthly_data['Season'] = monthly_data['month'].apply(get_season)
    monthly_data['Color'] = monthly_data['Season'].map(get_season_colors())
    monthly_data['theta'] = (monthly_data['month'] - 1) * (2 * np.pi / 12)
    
    years = sorted(monthly_data['year'].unique())
    frames = [create_animation_frame(monthly_data, year) for year in years]
    
    fig = go.Figure(
        data=frames[0].data,
        layout=create_animation_layout(monthly_data, lst_column),
        frames=frames
    )
    
    fig.write_html(f"{output_name}.html")
    return fig

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    if month in [3, 4, 5]: return 'Spring'
    if month in [6, 7, 8, 9]: return 'Summer'
    return 'Autumn'

def get_season_colors():
    return {
        'Winter': '#1f77b4',
        'Spring': '#2ca02c',
        'Summer': '#d62728',
        'Autumn': '#ff7f0e'
    }

def create_animation_layout(data, column_name):
    return go.Layout(
        title=f'🕐 {column_name} Animation (2000–2024) by Season',
        polar=dict(
            radialaxis=dict(range=[0, data[column_name].max() + 5]),
            angularaxis=dict(
                tickvals=[30 * i for i in range(1, 13)],
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            )
        ),
        updatemenus=[create_play_button()],
        showlegend=False
    )

def create_play_button():
    return dict(
        type='buttons',
        showactive=False,
        buttons=[dict(label='▶ Play',
                     method='animate',
                     args=[None, {"frame": {"duration": 1000, "redraw": True},
                                 "fromcurrent": True}]]
    )

def create_animation_frame(data, year):
    year_data = data[data['year'] == year]
    return go.Frame(
        data=[go.Barpolar(
            r=year_data['Night_LST'],
            theta=year_data['theta'],
            marker_color=year_data['Color'],
            marker_line_color='black',
            marker_line_width=1.2,
            opacity=0.85
        )],
        name=str(year)
    )

def generate_matplotlib_animation(df, lst_column, output_name):
    monthly_data = prepare_animation_data(df, lst_column)
    os.makedirs("frames", exist_ok=True)
    
    for i, year in enumerate(tqdm(sorted(monthly_data['year'].unique()))):
        create_animation_frame_image(monthly_data, year, i, lst_column)
    
    compile_animation(output_name)

def prepare_animation_data(df, lst_column):
    monthly_data = df.groupby(['year', 'month'])[lst_column].mean().reset_index()
    monthly_data['Season'] = monthly_data['month'].apply(get_season)
    monthly_data['Color'] = monthly_data['Season'].map(get_season_colors())
    monthly_data['theta'] = (monthly_data['month'] - 1) * (2 * np.pi / 12)
    return monthly_data

def create_animation_frame_image(data, year, index, column_name):
    plt.rcParams['font.family'] = 'DejaVu Serif'
    year_data = data[data['year'] == year]
    
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.bar(year_data['theta'], year_data[column_name], 
           width=0.1, color=year_data['Color'], edgecolor='black')
    
    configure_polar_plot(ax, data, column_name, year)
    plt.tight_layout(pad=3)
    plt.savefig(f"frames/frame_{index:03d}.png", dpi=950, bbox_inches='tight')
    plt.close()

def configure_polar_plot(ax, data, column_name, year):
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=12)
    
    yticks = ax.get_yticks()
    ax.set_yticklabels([str(int(y)) for y in yticks], fontsize=12)
    ax.set_ylim(0, max(data[column_name]) * 1.1)
    
    ax.set_title(f"{column_name} by Season\nYear: {year} - Jhang District", 
                 fontsize=18, pad=30)
    
    add_season_legend(ax)

def add_season_legend(ax):
    season_colors = get_season_colors()
    legend_labels = [plt.Line2D([0], [0], color=color, lw=6) 
                    for color in season_colors.values()]
    ax.legend(legend_labels, season_colors.keys(), 
              loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=4, frameon=False, fontsize=12)

def compile_animation(output_name):
    images = [imageio.v2.imread(f"frames/frame_{i:03d}.png") 
              for i in range(len(os.listdir("frames")))]
    imageio.mimsave(f"{output_name}.mp4", images, fps=1)

    

https://github.com/user-attachments/assets/e9e6ff38-5c80-4fd7-8db9-7ccb807b0e3f

