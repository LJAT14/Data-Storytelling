import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Business Intelligence Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .story-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 1rem;
    }
    .chapter-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .insight-box {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%);
        padding: 1.5rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-highlight {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1rem;
        border-radius: 6px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .narrative-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #34495e;
        margin: 1rem 0;
        text-align: justify;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #3498db;
    }
    .conclusion-box {
        background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
        padding: 2rem;
        border-radius: 8px;
        margin: 2rem 0;
        border-left: 5px solid #e74c3c;
        color: #2c3e50;
    }
    .recommendation-item {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 4px solid #27ae60;
        border: 1px solid #dee2e6;
    }
    .story-navigation {
        background: #ecf0f1;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border: 1px solid #bdc3c7;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .phase-indicator {
        background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate comprehensive business dataset
@st.cache_data
def generate_business_story_data():
    np.random.seed(42)
    
    # Date range: 3 years of business data
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Business context: E-commerce company expanding globally
    products = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Health & Beauty']
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East & Africa']
    channels = ['Online', 'Mobile App', 'Retail Store', 'Marketplace']
    customer_segments = ['Premium', 'Standard', 'Budget']
    
    data = []
    
    # Simulate business growth story
    for i, date in enumerate(dates):
        # Business phases
        if date.year == 2021:
            base_revenue = 50000  # Starting phase
            growth_factor = 1 + (i / len(dates)) * 0.5  # Gradual growth
        elif date.year == 2022:
            base_revenue = 75000  # Expansion phase
            growth_factor = 1 + (i / len(dates)) * 0.8  # Accelerated growth
        else:
            base_revenue = 100000  # Maturity phase
            growth_factor = 1 + (i / len(dates)) * 0.3  # Slower but steady growth
        
        # Seasonal effects
        month_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        if date.month in [11, 12]:  # Holiday season
            month_factor *= 1.5
        elif date.month in [6, 7]:  # Summer sales
            month_factor *= 1.2
        
        # Weekly patterns
        weekday_factor = 1.2 if date.weekday() < 5 else 0.8  # Weekday vs weekend
        
        # Generate daily transactions
        n_transactions = max(1, int(np.random.poisson(100 * growth_factor * month_factor)))
        
        for _ in range(n_transactions):
            product = np.random.choice(products)
            region = np.random.choice(regions, p=[0.35, 0.25, 0.25, 0.1, 0.05])  # Weighted by market size
            channel = np.random.choice(channels, p=[0.4, 0.35, 0.15, 0.1])  # Digital-first business
            segment = np.random.choice(customer_segments, p=[0.2, 0.6, 0.2])
            
            # Revenue calculation based on multiple factors
            base_price = {
                'Electronics': 300, 'Clothing': 80, 'Home & Garden': 150,
                'Sports': 120, 'Books': 25, 'Health & Beauty': 60
            }[product]
            
            # Channel markup
            channel_multiplier = {
                'Online': 1.0, 'Mobile App': 0.95, 'Retail Store': 1.2, 'Marketplace': 0.9
            }[channel]
            
            # Segment pricing
            segment_multiplier = {
                'Premium': 1.5, 'Standard': 1.0, 'Budget': 0.7
            }[segment]
            
            # Regional pricing
            region_multiplier = {
                'North America': 1.2, 'Europe': 1.1, 'Asia Pacific': 0.9,
                'Latin America': 0.8, 'Middle East & Africa': 0.7
            }[region]
            
            revenue = (base_price * channel_multiplier * segment_multiplier * 
                      region_multiplier * np.random.uniform(0.8, 1.3))
            
            # Calculate costs and margins
            cost_ratio = np.random.uniform(0.4, 0.7)  # 40-70% cost ratio
            profit = revenue * (1 - cost_ratio)
            
            # Marketing spend (varies by channel and segment)
            marketing_spend = revenue * np.random.uniform(0.05, 0.15)
            
            # Customer metrics
            customer_acquisition_cost = marketing_spend * np.random.uniform(0.8, 1.2)
            customer_lifetime_value = revenue * np.random.uniform(3, 8)  # 3-8x revenue multiplier
            
            data.append({
                'Date': date,
                'Product_Category': product,
                'Region': region,
                'Channel': channel,
                'Customer_Segment': segment,
                'Revenue': revenue,
                'Cost': revenue * cost_ratio,
                'Profit': profit,
                'Marketing_Spend': marketing_spend,
                'Customer_Acquisition_Cost': customer_acquisition_cost,
                'Customer_Lifetime_Value': customer_lifetime_value,
                'Units_Sold': max(1, int(revenue / (base_price * np.random.uniform(0.8, 1.2)))),
                'Customer_Rating': np.random.uniform(3.5, 5.0),
                'Return_Rate': np.random.uniform(0.02, 0.08)
            })
    
    df = pd.DataFrame(data)
    
    # Add derived metrics
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Profit_Margin'] = (df['Profit'] / df['Revenue'] * 100)
    df['ROI'] = (df['Profit'] / df['Marketing_Spend'] * 100)
    df['CLV_CAC_Ratio'] = df['Customer_Lifetime_Value'] / df['Customer_Acquisition_Cost']
    
    return df

# Load the data
df = generate_business_story_data()

# Header
st.markdown('<h1 class="story-header">Business Intelligence Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Data-Driven Business Intelligence and Strategic Analytics</p>', unsafe_allow_html=True)

# Story navigation
st.markdown("""
<div class="story-navigation">
<h4>Business Analysis Framework</h4>
<p><strong>Executive Summary:</strong> High-level business performance overview<br>
<strong>Foundation Phase:</strong> Initial market entry and establishment (2021)<br>
<strong>Growth Phase:</strong> Scaling operations and market expansion (2022)<br>
<strong>Market Analysis:</strong> Geographic expansion and regional performance<br>
<strong>Optimization Phase:</strong> Data-driven operational improvements (2023)<br>
<strong>Strategic Planning:</strong> Future roadmap and recommendations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.header("Business Intelligence Navigation")
selected_chapter = st.sidebar.selectbox(
    "Select Analysis Section",
    ["Executive Summary", "Foundation Phase (2021)", "Growth Phase (2022)", 
     "Market Analysis", "Optimization Phase (2023)", "Strategic Planning"]
)

# Interactive filters
st.sidebar.header("Data Filters")
year_filter = st.sidebar.multiselect(
    "Select Years",
    options=sorted(df['Year'].unique()),
    default=sorted(df['Year'].unique())
)

region_filter = st.sidebar.multiselect(
    "Select Regions",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# Apply filters
df_filtered = df[
    (df['Year'].isin(year_filter)) &
    (df['Region'].isin(region_filter))
]

# Analysis content based on selected section
if selected_chapter == "Executive Summary":
    st.markdown('<h2 class="chapter-header">Executive Summary</h2>', unsafe_allow_html=True)
    
    # Key metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df_filtered['Revenue'].sum()
        st.markdown(f"""
        <div class="metric-highlight">
            Total Revenue<br>
            ${total_revenue:,.0f}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_profit = df_filtered['Profit'].sum()
        st.markdown(f"""
        <div class="metric-highlight">
            Total Profit<br>
            ${total_profit:,.0f}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_margin = df_filtered['Profit_Margin'].mean()
        st.markdown(f"""
        <div class="metric-highlight">
            Avg Profit Margin<br>
            {avg_margin:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_customers = df_filtered['Units_Sold'].sum()
        st.markdown(f"""
        <div class="metric-highlight">
            Units Sold<br>
            {total_customers:,.0f}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="narrative-text">
    This comprehensive business intelligence analysis examines three years of operational data from a global e-commerce platform. 
    The company successfully evolved from a regional startup to an international marketplace, demonstrating consistent growth 
    across multiple markets and product categories. Key success factors include strategic digital channel optimization, 
    data-driven decision making, and adaptive regional market strategies.
    </div>
    """, unsafe_allow_html=True)
    
    # Overview visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue trend over time
        monthly_revenue = df_filtered.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
        monthly_revenue['Date'] = pd.to_datetime(monthly_revenue[['Year', 'Month']].assign(day=1))
        
        fig_revenue = px.line(
            monthly_revenue,
            x='Date',
            y='Revenue',
            title="Revenue Growth Trajectory",
            template="plotly_white"
        )
        fig_revenue.update_traces(line_color='#3498db', line_width=4)
        fig_revenue.update_layout(height=400)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        # Regional performance
        regional_performance = df_filtered.groupby('Region').agg({
            'Revenue': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        fig_regional = px.bar(
            regional_performance,
            x='Region',
            y='Revenue',
            title="Revenue Distribution by Region",
            template="plotly_white",
            color='Revenue',
            color_continuous_scale='Blues'
        )
        fig_regional.update_layout(height=400, xaxis_tickangle=45)
        st.plotly_chart(fig_regional, use_container_width=True)

elif selected_chapter == "Foundation Phase (2021)":
    st.markdown('<h2 class="chapter-header">Foundation Phase (2021)</h2>', unsafe_allow_html=True)
    
    # Focus on 2021 data
    df_2021 = df_filtered[df_filtered['Year'] == 2021]
    
    st.markdown("""
    <div class="narrative-text">
    The foundation year of 2021 established the core business infrastructure and market positioning. Initial operations 
    focused on product-market fit, establishing supply chains, and building customer acquisition channels. Strategic 
    decisions made during this period set the trajectory for subsequent growth phases.
    </div>
    """, unsafe_allow_html=True)
    
    if len(df_2021) > 0:
        # Key insights for 2021
        st.markdown(f"""
        <div class="insight-box">
            <h3>Foundation Year Performance Metrics</h3>
            <ul>
                <li><strong>Revenue Foundation:</strong> Generated ${df_2021['Revenue'].sum():,.0f} in total revenue</li>
                <li><strong>Primary Market:</strong> {df_2021['Region'].value_counts().index[0]} represented our core market focus</li>
                <li><strong>Lead Product Category:</strong> {df_2021['Product_Category'].value_counts().index[0]} dominated our product portfolio</li>
                <li><strong>Digital Channel Adoption:</strong> {(df_2021['Channel'].isin(['Online', 'Mobile App']).sum() / len(df_2021) * 100):.1f}% of transactions through digital platforms</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly progression in 2021
            monthly_2021 = df_2021.groupby('Month').agg({
                'Revenue': 'sum',
                'Profit_Margin': 'mean'
            }).reset_index()
            
            fig_monthly = px.line(
                monthly_2021,
                x='Month',
                y='Revenue',
                title="2021 Monthly Revenue Development",
                template="plotly_white"
            )
            fig_monthly.update_traces(line_color='#e74c3c', line_width=3, mode='lines+markers')
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            # Channel distribution in 2021
            channel_2021 = df_2021['Channel'].value_counts()
            
            fig_channel = px.pie(
                values=channel_2021.values,
                names=channel_2021.index,
                title="2021 Sales Channel Distribution",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_channel.update_layout(height=400)
            st.plotly_chart(fig_channel, use_container_width=True)
        
        # Challenge identification
        st.markdown("""
        <div class="insight-box">
            <h3>Strategic Challenges Identified</h3>
            <p>Analysis revealed critical areas requiring optimization for sustainable growth:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_margin_2021 = df_2021['Profit_Margin'].mean()
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>Profit Optimization</strong><br>
                Average Margin: {avg_margin_2021:.1f}%<br>
                <em>Requires improvement</em>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_cac_2021 = df_2021['Customer_Acquisition_Cost'].mean()
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>Acquisition Efficiency</strong><br>
                Avg CAC: ${avg_cac_2021:.0f}<br>
                <em>High acquisition costs</em>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            return_rate_2021 = df_2021['Return_Rate'].mean()
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>Quality Metrics</strong><br>
                Return Rate: {return_rate_2021:.1%}<br>
                <em>Quality enhancement needed</em>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.warning("No data available for 2021 with current filters.")

elif selected_chapter == "Growth Phase (2022)":
    st.markdown('<h2 class="chapter-header">Growth Phase (2022)</h2>', unsafe_allow_html=True)
    
    # Focus on 2022 data and comparison with 2021
    df_2022 = df_filtered[df_filtered['Year'] == 2022]
    df_2021 = df_filtered[df_filtered['Year'] == 2021]
    
    st.markdown("""
    <div class="narrative-text">
    2022 represented a transformational growth period characterized by aggressive market expansion, operational scaling, 
    and strategic investments in technology infrastructure. The company leveraged insights from the foundation year 
    to accelerate growth while maintaining operational efficiency.
    </div>
    """, unsafe_allow_html=True)
    
    if len(df_2022) > 0 and len(df_2021) > 0:
        # Growth metrics comparison
        revenue_growth = ((df_2022['Revenue'].sum() - df_2021['Revenue'].sum()) / df_2021['Revenue'].sum()) * 100
        profit_growth = ((df_2022['Profit'].sum() - df_2021['Profit'].sum()) / df_2021['Profit'].sum()) * 100
        
        st.markdown(f"""
        <div class="insight-box">
            <h3>Growth Phase Performance Results</h3>
            <ul>
                <li><strong>Revenue Acceleration:</strong> {revenue_growth:.1f}% increase from 2021 baseline</li>
                <li><strong>Profit Expansion:</strong> {profit_growth:.1f}% increase in total profitability</li>
                <li><strong>Market Presence:</strong> Active operations in {df_2022['Region'].nunique()} regional markets</li>
                <li><strong>Product Portfolio:</strong> {df_2022['Product_Category'].nunique()} active product categories</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Growth visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Year-over-year comparison
            yearly_comparison = pd.DataFrame({
                'Year': [2021, 2022],
                'Revenue': [df_2021['Revenue'].sum(), df_2022['Revenue'].sum()],
                'Profit': [df_2021['Profit'].sum(), df_2022['Profit'].sum()]
            })
            
            fig_yoy = px.bar(
                yearly_comparison,
                x='Year',
                y=['Revenue', 'Profit'],
                title="Year-over-Year Performance Comparison",
                template="plotly_white",
                barmode='group'
            )
            fig_yoy.update_layout(height=400)
            st.plotly_chart(fig_yoy, use_container_width=True)
        
        with col2:
            # Quarterly growth trend
            quarterly_2022 = df_2022.groupby('Quarter')['Revenue'].sum().reset_index()
            
            fig_quarterly = px.line(
                quarterly_2022,
                x='Quarter',
                y='Revenue',
                title="2022 Quarterly Revenue Progression",
                template="plotly_white",
                markers=True
            )
            fig_quarterly.update_traces(line_color='#27ae60', line_width=4, marker_size=10)
            fig_quarterly.update_layout(height=400)
            st.plotly_chart(fig_quarterly, use_container_width=True)
        
        # Channel performance analysis
        st.markdown("### Channel Performance Evolution")
        
        channel_comparison = pd.concat([
            df_2021.groupby('Channel')['Revenue'].sum().reset_index().assign(Year=2021),
            df_2022.groupby('Channel')['Revenue'].sum().reset_index().assign(Year=2022)
        ])
        
        fig_channels = px.bar(
            channel_comparison,
            x='Channel',
            y='Revenue',
            color='Year',
            title="Channel Performance: 2021 vs 2022",
            template="plotly_white",
            barmode='group'
        )
        fig_channels.update_layout(height=400, xaxis_tickangle=45)
        st.plotly_chart(fig_channels, use_container_width=True)
        
        # Growth challenges identified
        st.markdown("""
        <div class="insight-box">
            <h3>Scaling Challenge Analysis</h3>
            <p>Rapid expansion revealed operational challenges requiring strategic attention:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        margin_change = df_2022['Profit_Margin'].mean() - df_2021['Profit_Margin'].mean()
        cac_change = df_2022['Customer_Acquisition_Cost'].mean() - df_2021['Customer_Acquisition_Cost'].mean()
        rating_change = df_2022['Customer_Rating'].mean() - df_2021['Customer_Rating'].mean()
        
        with col1:
            margin_status = "Improved" if margin_change > 0 else "Declined"
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>Margin Performance</strong><br>
                Change: {margin_change:+.1f}%<br>
                <em>Status: {margin_status}</em>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            cac_status = "Increased" if cac_change > 0 else "Decreased"
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>Acquisition Costs</strong><br>
                Change: ${cac_change:+.0f}<br>
                <em>Status: {cac_status}</em>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            rating_status = "Improved" if rating_change > 0 else "Declined"
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>Customer Satisfaction</strong><br>
                Change: {rating_change:+.2f}<br>
                <em>Status: {rating_status}</em>
            </div>
            """, unsafe_allow_html=True)

elif selected_chapter == "Market Analysis":
    st.markdown('<h2 class="chapter-header">Geographic Market Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="narrative-text">
    Geographic expansion analysis reveals distinct regional performance patterns, customer behaviors, and market opportunities. 
    This comprehensive market assessment provides insights for strategic resource allocation and regional optimization strategies.
    </div>
    """, unsafe_allow_html=True)
    
    # Regional performance deep dive
    regional_metrics = df_filtered.groupby('Region').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Profit_Margin': 'mean',
        'Customer_Rating': 'mean',
        'Customer_Acquisition_Cost': 'mean',
        'Customer_Lifetime_Value': 'mean',
        'Return_Rate': 'mean'
    }).reset_index()
    
    regional_metrics['CLV_CAC_Ratio'] = regional_metrics['Customer_Lifetime_Value'] / regional_metrics['Customer_Acquisition_Cost']
    
    # Top performing regions
    top_regions = regional_metrics.nlargest(3, 'Revenue')
    
    st.markdown(f"""
    <div class="insight-box">
        <h3>Regional Performance Overview</h3>
        <ul>
            <li><strong>Leading Market:</strong> {top_regions.iloc[0]['Region']} (${top_regions.iloc[0]['Revenue']:,.0f} revenue)</li>
            <li><strong>Highest Margins:</strong> {regional_metrics.loc[regional_metrics['Profit_Margin'].idxmax(), 'Region']} ({regional_metrics['Profit_Margin'].max():.1f}% margin)</li>
            <li><strong>Best Customer Value:</strong> {regional_metrics.loc[regional_metrics['CLV_CAC_Ratio'].idxmax(), 'Region']} (CLV/CAC: {regional_metrics['CLV_CAC_Ratio'].max():.1f}x)</li>
            <li><strong>Global Presence:</strong> Active operations in {len(regional_metrics)} regional markets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Regional performance visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional revenue and profit
        fig_regional_rev = px.scatter(
            regional_metrics,
            x='Revenue',
            y='Profit',
            size='Customer_Lifetime_Value',
            color='Region',
            title="Regional Performance: Revenue vs Profit",
            template="plotly_white",
            hover_data=['Profit_Margin', 'Customer_Rating']
        )
        fig_regional_rev.update_layout(height=500)
        st.plotly_chart(fig_regional_rev, use_container_width=True)
    
    with col2:
        # CLV/CAC ratio by region
        fig_clv_cac = px.bar(
            regional_metrics.sort_values('CLV_CAC_Ratio', ascending=True),
            x='CLV_CAC_Ratio',
            y='Region',
            orientation='h',
            title="Customer Value Efficiency by Region",
            template="plotly_white",
            color='CLV_CAC_Ratio',
            color_continuous_scale='Viridis'
        )
        fig_clv_cac.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="Target: 3x")
        fig_clv_cac.update_layout(height=500)
        st.plotly_chart(fig_clv_cac, use_container_width=True)
    
    # Regional heatmap
    st.markdown("### Regional Performance Metrics Heatmap")
    
    # Create a correlation matrix of regional metrics
    metrics_for_heatmap = regional_metrics.set_index('Region')[['Revenue', 'Profit_Margin', 'Customer_Rating', 'CLV_CAC_Ratio']].T
    
    fig_heatmap = px.imshow(
        metrics_for_heatmap,
        title="Regional Performance Metrics Comparison",
        template="plotly_white",
        color_continuous_scale='RdYlBu_r',
        aspect="auto"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Market-specific insights
    st.markdown("### Regional Market Performance Details")
    
    for i, region in enumerate(regional_metrics['Region']):
        region_data = regional_metrics[regional_metrics['Region'] == region].iloc[0]
        
        # Determine region performance status
        revenue_rank = regional_metrics['Revenue'].rank(ascending=False)[regional_metrics['Region'] == region].iloc[0]
        margin_rank = regional_metrics['Profit_Margin'].rank(ascending=False)[regional_metrics['Region'] == region].iloc[0]
        
        performance_status = "Market Leader" if revenue_rank <= 2 else "Growth Market" if revenue_rank <= 4 else "Emerging Market"
        
        with st.expander(f"{region} - {performance_status}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Revenue", f"${region_data['Revenue']:,.0f}")
                st.metric("Profit Margin", f"{region_data['Profit_Margin']:.1f}%")
            
            with col2:
                st.metric("Customer Rating", f"{region_data['Customer_Rating']:.2f}/5.0")
                st.metric("Return Rate", f"{region_data['Return_Rate']:.1%}")
            
            with col3:
                st.metric("Avg CAC", f"${region_data['Customer_Acquisition_Cost']:.0f}")
                st.metric("Avg CLV", f"${region_data['Customer_Lifetime_Value']:,.0f}")
            
            with col4:
                st.metric("CLV/CAC Ratio", f"{region_data['CLV_CAC_Ratio']:.1f}x")
                efficiency_status = "Excellent" if region_data['CLV_CAC_Ratio'] > 4 else "Good" if region_data['CLV_CAC_Ratio'] > 3 else "Needs Improvement"
                st.write(f"**Efficiency Status:** {efficiency_status}")

elif selected_chapter == "Optimization Phase (2023)":
    st.markdown('<h2 class="chapter-header">Optimization Phase (2023)</h2>', unsafe_allow_html=True)
    
    # Focus on 2023 data and optimization insights
    df_2023 = df_filtered[df_filtered['Year'] == 2023]
    
    st.markdown("""
    <div class="narrative-text">
    2023 marked the optimization and efficiency phase, leveraging comprehensive data analytics to maximize ROI, 
    enhance customer experience, and streamline operations. Strategic focus shifted from growth to sustainable 
    profitability and operational excellence.
    </div>
    """, unsafe_allow_html=True)
    
    if len(df_2023) > 0:
        # Optimization metrics
        st.markdown(f"""
        <div class="insight-box">
            <h3>Optimization Phase Results</h3>
            <ul>
                <li><strong>Revenue Efficiency:</strong> ${df_2023['Revenue'].sum():,.0f} with enhanced profit optimization</li>
                <li><strong>Customer Experience:</strong> Average rating improved to {df_2023['Customer_Rating'].mean():.2f}/5.0</li>
                <li><strong>Cost Efficiency:</strong> Customer acquisition cost optimized to ${df_2023['Customer_Acquisition_Cost'].mean():.0f}</li>
                <li><strong>Quality Enhancement:</strong> Return rate reduced to {df_2023['Return_Rate'].mean():.1%}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Product category performance analysis
        category_performance = df_2023.groupby('Product_Category').agg({
            'Revenue': 'sum',
            'Profit_Margin': 'mean',
            'Customer_Rating': 'mean',
            'Return_Rate': 'mean',
            'Units_Sold': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Product category revenue vs margin
            fig_category = px.scatter(
                category_performance,
                x='Revenue',
                y='Profit_Margin',
                size='Units_Sold',
                color='Customer_Rating',
                hover_name='Product_Category',
                title="Product Category: Revenue vs Margin (2023)",
                template="plotly_white",
                color_continuous_scale='Viridis'
            )
            fig_category.update_layout(height=500)
            st.plotly_chart(fig_category, use_container_width=True)
        
        with col2:
            # Customer segment performance
            segment_performance = df_2023.groupby('Customer_Segment').agg({
                'Revenue': 'sum',
                'Profit_Margin': 'mean',
                'Customer_Lifetime_Value': 'mean'
            }).reset_index()
            
            fig_segment = px.sunburst(
                df_2023,
                path=['Customer_Segment', 'Channel'],
                values='Revenue',
                title="Revenue Distribution: Segments & Channels",
                template="plotly_white"
            )
            fig_segment.update_layout(height=500)
            st.plotly_chart(fig_segment, use_container_width=True)
        
        # Advanced analytics insights
        st.markdown("### Advanced Performance Analytics")
        
        # Customer behavior analysis
        customer_behavior = df_2023.groupby(['Customer_Segment', 'Channel']).agg({
            'Revenue': 'mean',
            'Customer_Rating': 'mean',
            'Return_Rate': 'mean'
        }).reset_index()
        
        # Channel effectiveness by segment
        fig_behavior = px.bar(
            customer_behavior,
            x='Customer_Segment',
            y='Revenue',
            color='Channel',
            title="Average Revenue by Customer Segment and Channel",
            template="plotly_white",
            barmode='group'
        )
        fig_behavior.update_layout(height=400)
        st.plotly_chart(fig_behavior, use_container_width=True)
        
        # ROI optimization analysis
        roi_analysis = df_2023.groupby(['Product_Category', 'Region']).agg({
            'ROI': 'mean',
            'Marketing_Spend': 'sum',
            'Revenue': 'sum'
        }).reset_index()
        
        fig_roi = px.scatter(
            roi_analysis,
            x='Marketing_Spend',
            y='ROI',
            size='Revenue',
            color='Product_Category',
            facet_col='Region',
            facet_col_wrap=3,
            title="ROI Analysis by Product Category and Region",
            template="plotly_white"
        )
        fig_roi.update_layout(height=600)
        st.plotly_chart(fig_roi, use_container_width=True)
        
        # Key optimization discoveries
        st.markdown("""
        <div class="insight-box">
            <h3>Key Optimization Insights</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        best_category = category_performance.loc[category_performance['Profit_Margin'].idxmax(), 'Product_Category']
        best_margin = category_performance['Profit_Margin'].max()
        
        best_segment = segment_performance.loc[segment_performance['Customer_Lifetime_Value'].idxmax(), 'Customer_Segment']
        best_clv = segment_performance['Customer_Lifetime_Value'].max()
        
        with col1:
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>Top Product Category</strong><br>
                {best_category}<br>
                <em>{best_margin:.1f}% profit margin</em>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>Most Valuable Segment</strong><br>
                {best_segment}<br>
                <em>${best_clv:,.0f} avg CLV</em>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            best_roi_combo = roi_analysis.loc[roi_analysis['ROI'].idxmax()]
            st.markdown(f"""
            <div class="recommendation-item">
                <strong>Best ROI Combination</strong><br>
                {best_roi_combo['Product_Category']}<br>
                <em>{best_roi_combo['Region']} - {best_roi_combo['ROI']:.0f}% ROI</em>
            </div>
            """, unsafe_allow_html=True)

elif selected_chapter == "Strategic Planning":
    st.markdown('<h2 class="chapter-header">Strategic Planning & Future Roadmap</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="narrative-text">
    Based on comprehensive analysis of three years of operational data, this strategic planning framework provides 
    actionable recommendations for sustained growth, operational excellence, and market leadership. These insights 
    will guide resource allocation and strategic decision-making for future business development.
    </div>
    """, unsafe_allow_html=True)
    
    # Strategic recommendations based on data insights
    st.markdown("""
    <div class="conclusion-box">
        <h3>Strategic Recommendations Framework</h3>
        <p>Data-driven analysis reveals five critical strategic focus areas that will maximize growth potential 
        and operational efficiency for sustainable business success.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation categories
    recommendations = [
        {
            "title": "Geographic Market Expansion",
            "priority": "High",
            "description": "Target high-potential emerging markets with proven ROI frameworks",
            "metrics": ["Target CLV/CAC ratio > 4x", "Customer rating > 4.5", "Profit margin > 25%"],
            "timeline": "6-12 months"
        },
        {
            "title": "Digital Channel Enhancement",
            "priority": "High", 
            "description": "Optimize mobile and online platform performance and user experience",
            "metrics": ["Increase mobile revenue by 40%", "Reduce mobile CAC by 20%", "Improve conversion rates"],
            "timeline": "3-6 months"
        },
        {
            "title": "Customer Segment Strategy",
            "priority": "Medium",
            "description": "Focus on Premium segment growth and customer lifetime value optimization",
            "metrics": ["Grow Premium segment by 30%", "Increase CLV by 25%", "Reduce churn by 15%"],
            "timeline": "6-18 months"
        },
        {
            "title": "Product Portfolio Management",
            "priority": "Medium",
            "description": "Optimize high-margin categories and restructure underperforming segments",
            "metrics": ["Focus on Electronics and Health & Beauty", "Improve overall margin by 5%", "Reduce return rates"],
            "timeline": "12-24 months"
        },
        {
            "title": "Analytics & Technology Infrastructure",
            "priority": "High",
            "description": "Implement advanced analytics, automation, and business intelligence systems",
            "metrics": ["Real-time dashboards", "Predictive analytics", "Process automation"],
            "timeline": "6-12 months"
        }
    ]
    
    # Display recommendations
    for i, rec in enumerate(recommendations):
        priority_indicator = "HIGH PRIORITY" if rec["priority"] == "High" else "MEDIUM PRIORITY" if rec["priority"] == "Medium" else "LOW PRIORITY"
        
        with st.expander(f"{rec['title']} | {priority_indicator}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description:** {rec['description']}")
                st.write("**Key Performance Indicators:**")
                for metric in rec['metrics']:
                    st.write(f"• {metric}")
            
            with col2:
                st.write(f"**Priority Level:** {rec['priority']}")
                st.write(f"**Implementation Timeline:** {rec['timeline']}")
    
    # Financial projections based on recommendations
    st.markdown("### Financial Impact Projections")
    
    # Create projection scenarios
    current_revenue = df_filtered['Revenue'].sum()
    current_profit = df_filtered['Profit'].sum()
    
    scenarios = {
        "Conservative": {"revenue_growth": 0.15, "margin_improvement": 0.02},
        "Moderate": {"revenue_growth": 0.25, "margin_improvement": 0.05},
        "Aggressive": {"revenue_growth": 0.40, "margin_improvement": 0.08}
    }
    
    projection_data = []
    for scenario, params in scenarios.items():
        projected_revenue = current_revenue * (1 + params["revenue_growth"])
        current_margin = (current_profit / current_revenue)
        new_margin = current_margin + params["margin_improvement"]
        projected_profit = projected_revenue * new_margin
        
        projection_data.append({
            "Scenario": scenario,
            "Projected Revenue": projected_revenue,
            "Projected Profit": projected_profit,
            "Revenue Growth": params["revenue_growth"] * 100,
            "Margin Improvement": params["margin_improvement"] * 100
        })
    
    projection_df = pd.DataFrame(projection_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_revenue_proj = px.bar(
            projection_df,
            x="Scenario",
            y="Projected Revenue",
            title="Revenue Projection Scenarios",
            template="plotly_white",
            color="Projected Revenue",
            color_continuous_scale="Blues"
        )
        fig_revenue_proj.add_hline(y=current_revenue, line_dash="dash", line_color="red", annotation_text="Current Revenue")
        fig_revenue_proj.update_layout(height=400)
        st.plotly_chart(fig_revenue_proj, use_container_width=True)
    
    with col2:
        fig_profit_proj = px.bar(
            projection_df,
            x="Scenario",
            y="Projected Profit",
            title="Profit Projection Scenarios",
            template="plotly_white",
            color="Projected Profit",
            color_continuous_scale="Greens"
        )
        fig_profit_proj.add_hline(y=current_profit, line_dash="dash", line_color="red", annotation_text="Current Profit")
        fig_profit_proj.update_layout(height=400)
        st.plotly_chart(fig_profit_proj, use_container_width=True)
    
    # Implementation roadmap
    st.markdown("### Implementation Roadmap")
    
    roadmap_data = [
        {"Phase": "Q1 2024", "Focus": "Digital Optimization", "Key Actions": "Mobile platform enhancement, UX optimization", "Expected Impact": "20% mobile conversion increase"},
        {"Phase": "Q2 2024", "Focus": "Analytics Infrastructure", "Key Actions": "BI dashboards, predictive modeling", "Expected Impact": "Data-driven decision framework"},
        {"Phase": "Q3 2024", "Focus": "Market Expansion", "Key Actions": "Geographic expansion, market localization", "Expected Impact": "15% revenue growth"},
        {"Phase": "Q4 2024", "Focus": "Customer Strategy", "Key Actions": "Premium segment programs, retention optimization", "Expected Impact": "25% CLV improvement"},
    ]
    
    roadmap_df = pd.DataFrame(roadmap_data)
    
    for _, phase in roadmap_df.iterrows():
        st.markdown(f"""
        <div class="phase-indicator">
            {phase['Phase']}: {phase['Focus']}
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Actions:** {phase['Key Actions']}")
        with col2:
            st.write(f"**Impact:** {phase['Expected Impact']}")
    
    # Final insights and call to action
    st.markdown("""
    <div class="conclusion-box">
        <h3>Strategic Implementation Framework</h3>
        <ol>
            <li><strong>Data-Driven Excellence:</strong> Comprehensive analytics confirm that strategic decisions supported by robust data analysis deliver superior business outcomes.</li>
            <li><strong>Customer-Centric Growth:</strong> Premium customer segments and high-satisfaction channels drive sustainable long-term growth.</li>
            <li><strong>Market Opportunity:</strong> Emerging markets present significant expansion potential with appropriate localization and market entry strategies.</li>
            <li><strong>Technology Investment:</strong> Advanced analytics infrastructure and automation capabilities are essential for maintaining competitive positioning.</li>
            <li><strong>Continuous Improvement:</strong> Regular performance monitoring and strategic adjustment ensure sustained operational excellence.</li>
        </ol>
        
        <h4>Immediate Action Items:</h4>
        <ul>
            <li>Secure budget allocation for digital platform enhancements and technology infrastructure</li>
            <li>Conduct comprehensive market research for targeted geographic expansion opportunities</li>
            <li>Establish data science and analytics team expansion strategy</li>
            <li>Implement comprehensive KPI monitoring systems for real-time performance tracking</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Interactive data exploration section
st.markdown("---")
st.markdown("### Interactive Business Intelligence Tools")

# Allow users to create custom analysis
analysis_type = st.selectbox(
    "Select Analysis Framework",
    ["Custom Metrics Analysis", "Trend Analysis", "Performance Correlation", "Comparative Analysis"]
)

if analysis_type == "Custom Metrics Analysis":
    col1, col2 = st.columns(2)
    
    with col1:
        x_metric = st.selectbox("X-Axis Metric", ['Revenue', 'Profit', 'Profit_Margin', 'Customer_Rating', 'ROI'])
        y_metric = st.selectbox("Y-Axis Metric", ['Profit', 'Revenue', 'Customer_Lifetime_Value', 'Customer_Acquisition_Cost', 'ROI'])
    
    with col2:
        color_by = st.selectbox("Segment By", ['Region', 'Product_Category', 'Customer_Segment', 'Channel'])
        size_by = st.selectbox("Size By", ['Revenue', 'Profit', 'Units_Sold', 'Marketing_Spend'])
    
    # Create custom visualization
    fig_custom = px.scatter(
        df_filtered,
        x=x_metric,
        y=y_metric,
        color=color_by,
        size=size_by,
        title=f"Business Intelligence Analysis: {x_metric} vs {y_metric}",
        template="plotly_white",
        opacity=0.7
    )
    fig_custom.update_layout(height=500)
    st.plotly_chart(fig_custom, use_container_width=True)

elif analysis_type == "Trend Analysis":
    st.markdown("### Temporal Trend Analysis")
    
    metric_to_analyze = st.selectbox(
        "Select Metric for Trend Analysis",
        ['Revenue', 'Profit', 'Profit_Margin', 'Customer_Rating', 'Customer_Acquisition_Cost']
    )
    
    # Monthly trend analysis
    monthly_trend = df_filtered.groupby(['Year', 'Month'])[metric_to_analyze].mean().reset_index()
    monthly_trend['Date'] = pd.to_datetime(monthly_trend[['Year', 'Month']].assign(day=1))
    
    fig_trend = px.line(
        monthly_trend,
        x='Date',
        y=metric_to_analyze,
        title=f"{metric_to_analyze} Trend Analysis Over Time",
        template="plotly_white"
    )
    fig_trend.update_traces(line_width=3)
    fig_trend.update_layout(height=400)
    st.plotly_chart(fig_trend, use_container_width=True)

elif analysis_type == "Performance Correlation":
    st.markdown("### Performance Correlation Analysis")
    
    # Calculate correlation matrix
    numeric_columns = ['Revenue', 'Profit', 'Profit_Margin', 'Customer_Rating', 'ROI', 'CLV_CAC_Ratio']
    correlation_matrix = df_filtered[numeric_columns].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        title="Business Metrics Correlation Matrix",
        template="plotly_white",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

elif analysis_type == "Comparative Analysis":
    st.markdown("### Comparative Performance Analysis")
    
    comparison_dimension = st.selectbox(
        "Compare By",
        ['Region', 'Product_Category', 'Customer_Segment', 'Channel', 'Year']
    )
    
    comparison_metric = st.selectbox(
        "Metric to Compare",
        ['Revenue', 'Profit', 'Profit_Margin', 'Customer_Rating', 'ROI']
    )
    
    comparison_data = df_filtered.groupby(comparison_dimension)[comparison_metric].sum().reset_index()
    
    fig_comparison = px.bar(
        comparison_data,
        x=comparison_dimension,
        y=comparison_metric,
        title=f"{comparison_metric} Comparison by {comparison_dimension}",
        template="plotly_white"
    )
    fig_comparison.update_layout(height=400, xaxis_tickangle=45)
    st.plotly_chart(fig_comparison, use_container_width=True)

# Export and reporting options
st.markdown("### Business Intelligence Reporting")

col1, col2, col3 = st.columns(3)

with col1:
    # Export filtered data
    csv_data = df_filtered.to_csv(index=False)
    st.download_button(
        label="Export Business Data (CSV)",
        data=csv_data,
        file_name=f"business_intelligence_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    # Export summary report
    summary_stats = df_filtered.describe()
    summary_csv = summary_stats.to_csv()
    st.download_button(
        label="Export Summary Statistics (CSV)",
        data=summary_csv,
        file_name=f"business_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col3:
    # Export insights as JSON
    insights = {
        "total_revenue": float(df_filtered['Revenue'].sum()),
        "total_profit": float(df_filtered['Profit'].sum()),
        "avg_profit_margin": float(df_filtered['Profit_Margin'].mean()),
        "top_region": df_filtered.groupby('Region')['Revenue'].sum().idxmax(),
        "top_product": df_filtered.groupby('Product_Category')['Revenue'].sum().idxmax(),
        "analysis_date": datetime.now().isoformat()
    }
    
    insights_json = pd.Series(insights).to_json(indent=2)
    st.download_button(
        label="Export Key Insights (JSON)",
        data=insights_json,
        file_name=f"business_insights_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Business Intelligence Analytics Platform | Data-Driven Strategic Decision Making<br>"
    "Transform business data into actionable insights for competitive advantage"
    "</div>",
    unsafe_allow_html=True
)