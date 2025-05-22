import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from streamlit_extras.add_vertical_space import add_vertical_space
import plotly.graph_objects as go

#Getting Data
URL_DATA = 'https://storage.data.gov.my/transportation/ridership_headline.parquet'
df = pd.read_parquet(URL_DATA)
if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])


#Start of the Page
st.set_page_config(page_title="🚅 Ridership Dashboard", layout="wide")

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 2, 0.2, 1, 0.1)
)

row0_1.title("Rapid Train Ridership Visualization 🚅")

with row0_2:
    add_vertical_space()

row0_2.subheader(
    "A Streamlit web app by [Nizar Mazlan](https://www.linkedin.com/in/mohd-nizar-mustaqeem-mazlan-a5192516b/)"
)

row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))

with row1_1:
    about = st.expander("Open Data Portal? 👉")
    with about:
        st.markdown(
            """ 
            - The data used for this project has been sourced from Malaysia's official open data portal, [data.gov.my](https://data.gov.my/). Launched on the 18th of August 2014, data.gov.my serves as the Malaysian government's pioneering Open Data initiative. It stands as a testament to Malaysia's commitment to transparency, accessibility, and innovation in the digital age.At the core of data.gov.my's vision is to provide a wealth of information and datasets that are accessible to the public free of charge. Any user of the portal is encouraged to explore and harness this valuable resource for a wide range of purposes, including research, analysis, and application development.
            
            - The data shared on data.gov.my is governed by a set of terms and conditions outlined in the data policy. While the data is accessible for free, it is essential that users adhere to these terms and conditions to ensure responsible and ethical usage. Compliance with these guidelines ensures that the data is used to promote the common good and national progress. In alignment with data.gov.my's mission, our objective is to contribute to the realization of Malaysia's aspiration to become the leading data-driven nation in the world. By leveraging the power of data, we aim to drive innovation, foster evidence-based decision-making, and unlock opportunities that will benefit not only Malaysia but also the global community.Together, we aspire to harness the transformative potential of data to build a more connected, informed, and prosperous future for all.
            """
        )

    need_help = st.expander("What is this project about? 👉")
    with need_help:
        st.markdown(
            """ 
            This project is centered around data visualizations derived from the field of transportation, with a specific focus on ridership data for both rail and bus systems in Malaysia. The visualizations we've explored encompass a range of insights and patterns within this dataset, helping us better understand and analyze the dynamics of public transportation in the country.

            These visualizations include heatmaps that reveal the density of ridership, total ridership trends over time, monthly averages, ridership patterns by the day of the week, and line charts depicting ridership changes over time. Additionally, we utilize heatmaps for a unique perspective on ridership patterns over time, comparing weekend and weekday ridership through area charts, and examining the distribution of ridership for in-depth analysis.

            Through these visualizations, we aim to shed light on the intricacies of Malaysia's transportation systems, with the ultimate goal of contributing to the nation's mission to become a prominent data-driven country.
            """
        )
    
    column_description = st.expander("Data Description 🧾")
    variable_definitions = {
    'date (Date)': 'Date in YYYY-MM-DD format',
    'bus_rkl (Integer)': 'Ridership: Rapid Bus (KL)\nNumber of trips, NOT number of unique individuals',
    'bus_rkn (Integer)': 'Ridership: Rapid Bus (Kuantan)\nNumber of trips, NOT number of unique individuals',
    'bus_rpn (Integer)': 'Ridership: Rapid Bus (Penang)\nNumber of trips, NOT number of unique individuals',
    'rail_lrt_ampang (Integer)': 'Ridership: LRT Ampang Line\nNumber of trips, NOT number of unique individuals',
    'rail_lrt_kj (Integer)': 'Ridership: LRT Kelana Jaya Line\nNumber of trips, NOT number of unique individuals',
    'rail_monorail (Integer)': 'Ridership: Monorail Line\nNumber of trips, NOT number of unique individuals',
    'rail_mrt_kajang (Integer)': 'Ridership: MRT Kajang Line\nNumber of trips, NOT number of unique individuals',
    'rail_mrt_pjy (Integer)': 'Ridership: MRT Putrajaya Line\nNumber of trips, NOT number of unique individuals',
    'rail_ets (Integer)': 'Ridership: KTMB ETS\nNumber of trips, NOT number of unique individuals',
    'rail_komuter (Integer)': 'Ridership: KTM Komuter Utara\nNumber of trips, NOT number of unique individuals',
    'rail_tebrau (Integer)': 'Ridership: KTM Shuttle Tebrau\nNumber of trips, NOT number of unique individuals',
    'rail_intercity (Integer)': 'Ridership: KTM Intercity\nNumber of trips, NOT number of unique individuals'
    }

    with column_description:
        # Convert the dictionary to a DataFrame for easier display in Streamlit
        df_variable_description = pd.DataFrame(variable_definitions.items(), columns=['Variable', 'Description'])

        # Create a table to display the variable definitions
        st.write("Variable Descriptions")
        st.table(df_variable_description)

        # To download the latest data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        csv = convert_df(df)
        st.write("You can get the latest data from [data.gov.my](https://data.gov.my/) or you download it here using the button")
        st.download_button(
            "Press to Download",
            csv,
            "ridership.csv",
            "text/csv",
            key='download-csv'
        )

st.write("")
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

with row3_1:
    # First Visualization
    st.title('Heatmap of Ridership')
    # Calculate the correlation matrix
    correlation_matrix = df[['rail_lrt_ampang', 'rail_mrt_kajang', 'rail_lrt_kj', 'rail_monorail','rail_ets','rail_intercity','rail_komuter','rail_tebrau','bus_rkl','bus_rkn','bus_rpn','rail_mrt_pjy']].corr()
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.01, ax=ax)
    #ax.set_title('Heatmap of Rail and Bus Lines Ridership')
    # Display the heatmap in the Streamlit app
    # Set the dark mode style for Matplotlib
    plt.style.use('dark_background')
    st.pyplot(fig)
    st.write("These heatmaps aim to uncover the correlation between ridership on various rail and bus lines. By visualizing the data in this manner, we can discern any patterns or relationships that may exist among different transportation modes, which can inform decision-making and resource allocation")

with row3_2:
    st.title('Total Ridership Over Time')
    totaldf = df.copy()
    # Calculate the total ridership for all rail and bus lines
    totaldf['total_ridership'] = totaldf[['bus_rkl', 'bus_rkn', 'bus_rpn', 'rail_lrt_ampang', 'rail_mrt_kajang', 'rail_lrt_kj', 'rail_monorail', 'rail_mrt_pjy', 'rail_ets', 'rail_intercity', 'rail_komuter', 'rail_tebrau']].sum(axis=1)
    # Create a line chart for total ridership over time
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.plot(totaldf['date'], totaldf['total_ridership'], label='Total Ridership', color='red', linewidth=2)
    # Customize the plot
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Ridership')
    ax.grid(True)
    ax.legend()
    # Display the plot in the Streamlit app
    st.pyplot(fig)
    st.write('This visualization offers a comprehensive view of ridership trends over an extended period. By tracking total ridership over time, we can identify long-term patterns, spot growth trends, and assess the overall health of the public transportation system.')

row4_spacer1, row4_1, row4_spacer2 = st.columns(
    (0.1, 2, 0.2)
)

with row4_1:

    st.title('Analysis by Rail and Bus Lines')

    # Create a multiselect widget to select rail lines
    st.subheader('Please Select Rail/Bus Lines:')
    selected_rail_lines = st.selectbox('Rail Lines', df.columns[1:])  # Assuming the first two columns are not rail lines
    
    # Get the selected rail lines as a string
    selected_rail_lines_str = selected_rail_lines

    # Filter the data based on selected rail lines
    if selected_rail_lines:
        filtered_data = df[['date',selected_rail_lines]]  # Include the 'date' column along with selected rail lines
    else:
        filtered_data = df[['date']]  # If no rail lines are selected, show only the 'date' column

    # Create data visualizations or further analysis with the filtered data
    st.subheader('Charts:')

add_vertical_space()
row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

#Visualization 1
with row5_1:
    # Main content area title
    st.title('Monthly Average Ridership')

    # Copy the DataFrame
    monthlyDF = pd.DataFrame(filtered_data.copy())

    # Ensure the 'date' column is in datetime format
    monthlyDF['date'] = pd.to_datetime(monthlyDF['date'])

    # Extract the month of the year (1=January, 2=February, etc.)
    monthlyDF['month'] = monthlyDF['date'].dt.month

    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    monthlyDF['month'] = monthlyDF['month'].apply(lambda x: month_names[x - 1])

    # Group the data by 'month' and calculate the mean for each column
    average_ridership_by_month = monthlyDF.groupby('month').mean()

    months = average_ridership_by_month.index
    values = average_ridership_by_month[selected_rail_lines]
    
    # Create a bar chart using Plotly Express
    fig = px.bar(
        average_ridership_by_month,
        x= months,
        y= values,
        title='Monthly Average Ridership by Month of the Year',
        color_discrete_sequence=["#9EE6CF"]
    )

    fig.update_xaxes(title='Month of the Year')
    fig.update_yaxes(title='Average Monthly Ridership')

    # Display the plot in Streamlit using st.plotly_chart
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    st.write('By calculating and visualizing the monthly averages of ridership, we can gain valuable insights into the seasonality and trends within the dataset. This can help us identify which months experience higher or lower ridership, which, in turn, could be due to various factors such as holidays, weather, or special events.')

#Visualization 2
with row5_2:
    # Main content area title
    st.title('Daily Average Ridership')

    # Copy
    DaysofWeekDF = filtered_data.copy()

    # Ensure the 'date' column is in datetime format
    DaysofWeekDF['date'] = pd.to_datetime(DaysofWeekDF['date'])

    # Extract the day of the week (0=Monday, 1=Tuesday, etc.)
    DaysofWeekDF['day_of_week'] = DaysofWeekDF['date'].dt.dayofweek

    # Map the day of the week number to day name
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    DaysofWeekDF['day_name'] = DaysofWeekDF['day_of_week'].apply(lambda x: day_names[x])

    # Group the data by day name and calculate the average daily ridership for each rail line
    average_ridership_by_day = DaysofWeekDF.groupby('day_name').mean().reset_index()

    weeks = average_ridership_by_day['day_name']
    values = average_ridership_by_day[selected_rail_lines]

    # Create a bar chart using Plotly Express
    # Create a bar chart using Plotly Express
    fig = px.bar(
        average_ridership_by_day,
        x= weeks,
        y= values,
        title='Daily Average Ridership by Day of the Week',
        color_discrete_sequence=["#9EE6CF"]
    )    
    fig.update_xaxes(title='Day of the Week')
    fig.update_yaxes(title='Average Daily Ridership')

    # Display the plot in Streamlit using st.plotly_chart
    st.plotly_chart(fig, use_container_width=True)

    st.write('This visualization is geared towards understanding the ridership patterns across different days of the week. By discerning the trends on a day-to-day basis, we can pinpoint which days are busier or quieter, potentially revealing the impact of workdays, weekends, or other factors on ridership.')

add_vertical_space()
row6_space1, row6_1, row6_space2, row6_2, row6_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

with row6_1:
    # Main content area title
    st.title('Ridership over Time ({})'.format(selected_rail_lines))

    # Copy
    ridership_over_time = filtered_data.copy()
    ridership_over_time = ridership_over_time.dropna()

    fig = px.bar(
        ridership_over_time,
        x= ridership_over_time['date'],
        y= ridership_over_time[selected_rail_lines],
        title='Ridership over Time',
        color_discrete_sequence=["#9EE6CF"]
    )    
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Ridership')

    # Display the plot in Streamlit using st.plotly_chart
    st.plotly_chart(fig, use_container_width=True)

    st.write('A line chart depicting ridership over time serves as a dynamic tool for examining how ridership fluctuates on a daily, weekly, or monthly basis. It allows us to visualize the short-term variations and spot any recurring patterns or anomalies.')

with row6_2:
    st.title("Ridership Heatmap over Time")
    # Sample data for "bus_rkl" ridership over time (dates)
    heatmapdata = filtered_data.copy()
    heatmapdata = heatmapdata.dropna()
    dates = heatmapdata['date']
    heatmapchart = heatmapdata[selected_rail_lines]

    # Convert dates to numerical values (e.g., day index)
    date_values = np.arange(len(dates))

    # Create a heatmap using plotly
    fig = go.Figure(data=go.Heatmap(z=[heatmapchart], x=dates, y=["Ridership"],colorscale='YlGnBu'))
    fig.update_layout(title="Ridership Heatmap for {} Over Time".format(selected_rail_lines))

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    st.write('This heatmap provides an additional perspective on ridership over time, helping us delve deeper into the dataset. By utilizing color intensity to represent the changing levels of ridership across various timeframes, we can uncover nuances that might not be immediately evident through other visualizations.')

add_vertical_space()
row7_space1, row7_1, row7_space2, row7_2, row7_space3 = st.columns(
    (0.1, 1, 0.1, 1, 0.1)
)

with row7_1:
    st.title('Weekend vs Weekdays Area Chart')

    # Copy
    weekday_end = filtered_data.copy()
    weekday_end = weekday_end.dropna()

    # Calculate the day of the week for each date (0 = Monday, 6 = Sunday)
    weekday_end['DayOfWeek'] = pd.to_datetime(weekday_end['date']).dt.dayofweek

    # Define a function to classify days as weekdays (0-4) or weekends (5-6)
    def classify_day(day):
        return 'Weekday' if day < 5 else 'Weekend'

    weekday_end['DayType'] = weekday_end['DayOfWeek'].apply(classify_day)

    # Create a stacked area chart
    fig = px.area(
        weekday_end,
        x='date',
        y=selected_rail_lines,
        color='DayType',
        title='Stacked Area Chart for Daily Ridership of {}'.format(selected_rail_lines),
        color_discrete_map={'Weekday': '#9EE6CF', 'Weekend': '#eb6060'}
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    st.write('This area chart is designed to highlight the differences between weekend and weekday ridership. It enables us to compare and contrast the ridership patterns on these two distinct types of days, potentially revealing varying commuter behaviors and preferences.')

with row7_2:
    # Main content area title
    st.title('Ridership Distribution')

    # Copy
    ridership_dist = filtered_data.copy()
    # Create a box plot for the selected mode
    fig = px.box(
        ridership_dist,
        x=selected_rail_lines,
        title=f'Ridership Distribution for {selected_rail_lines}',
        labels={selected_rail_lines: 'Number of Riders'},
        notched=True,  # Show the notched box (confidence intervals)
        hover_name="date",  # Display date when hovering over data points
        color_discrete_sequence=["#9EE6CF"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("The aim here is to conduct a comprehensive statistical analysis of ridership data. By visualizing the distribution of ridership, we can assess key statistical measures such as mean, median, and variance. This information can provide crucial insights into the ridership data's characteristics and potential outliers.")

row8_space1, row8_1, row8_space2 = st.columns(
    (0.05, 1, 0.1)
)

with row8_1:
    # Add a horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("If you have any questions or feedback, or if you simply want to connect, please don't hesitate to get in touch. You can find me on [Twitter](https://twitter.com/NizarMazlan) and [LinkedIn](https://www.linkedin.com/in/mohd-nizar-mustaqeem-mazlan-a5192516b/). For a deeper dive into my work, visit and checkout my [My Data Science Portfolio](https://www.datascienceportfol.io/nizarmazlan).")
