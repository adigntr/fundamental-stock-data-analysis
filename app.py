import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats

# Page Configuration
st.set_page_config(
    page_title="Fundamental Stock Data Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #00acc1;
        margin: 1rem 0;
        color: #333333;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .insight-box h3, .insight-box h4, .insight-box p {
        color: #333333 !important;
    }
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .correlation-note {
        font-size: 0.85rem;
        color: #666;
        font-style: italic;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 Fundamental Stock Analysis Dashboard<br><span style="font-size: 1rem; font-weight: normal;">Advanced Analytics & Insights</span></div>', unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    stocks = pd.read_csv('data/stocks.csv')
    stats = pd.read_csv('data/data_stats.csv')
    valuation = pd.read_csv('data/data_valuation.csv')
    revenue = pd.read_csv('data/data_revenue.csv')
    profit = pd.read_csv('data/data_profit.csv')
    assets = pd.read_csv('data/data_assets.csv')
    equity = pd.read_csv('data/data_equity.csv')
    cash = pd.read_csv('data/data_cash.csv')
    return stocks, stats, valuation, revenue, profit, assets, equity, cash

try:
    stocks, stats, valuation, revenue, profit, assets, equity, cash = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar Configuration
st.sidebar.markdown("## 🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Get list of tickers
all_tickers = stats['ticker'].dropna().unique().tolist()

# Stock Selection
st.sidebar.markdown("### 📊 Stock Selection")
selected_tickers = st.sidebar.multiselect(
    "Select Stocks for Analysis",
    options=all_tickers,
    default=['MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AAPL'] if 'AAPL' in all_tickers else all_tickers[:7],
    help="Select one or more stocks to analyze"
)

if not selected_tickers:
    selected_tickers = all_tickers[:10]

# Display selected count
st.sidebar.info(f"📌 {len(selected_tickers)} stocks selected")

# Year Range Selection
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Time Period")
available_years = sorted(revenue['year'].dropna().unique().astype(int).tolist())

if len(available_years) >= 2:
    min_year, max_year = st.sidebar.select_slider(
        "Select Year Range",
        options=available_years,
        value=(min(available_years), max(available_years)),
        help="Filter data by year range"
    )
    selected_years = [y for y in available_years if min_year <= y <= max_year]
    st.sidebar.success(f"📆 Period: {min_year} - {max_year}")
else:
    selected_years = available_years

# Filter data
stats_filtered = stats[stats['ticker'].isin(selected_tickers)]
valuation_filtered = valuation[valuation['ticker'].isin(selected_tickers)]

# Filter time-series data
revenue_filtered = revenue[(revenue['ticker'].isin(selected_tickers)) & (revenue['year'].isin(selected_years))]
profit_filtered = profit[(profit['ticker'].isin(selected_tickers)) & (profit['year'].isin(selected_years))]
assets_filtered = assets[(assets['ticker'].isin(selected_tickers)) & (assets['year'].isin(selected_years))]
equity_filtered = equity[(equity['ticker'].isin(selected_tickers)) & (equity['year'].isin(selected_years))]
cash_filtered = cash[(cash['ticker'].isin(selected_tickers)) & (cash['year'].isin(selected_years))]

# =============================================================================
# SECTION 1: Executive Summary & KPIs
# =============================================================================
st.markdown("## 📈 Executive Summary")

# Calculate KPIs
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_roe = stats_filtered['Return on Equity'].mean()
    st.metric(
        "Avg ROE",
        f"{avg_roe:.1f}%",
        delta=f"{avg_roe - stats['Return on Equity'].mean():.1f}% vs market" if not stats.empty else None
    )

with col2:
    avg_roa = stats_filtered['Return on Assets'].mean()
    st.metric(
        "Avg ROA",
        f"{avg_roa:.1f}%",
        delta=f"{avg_roa - stats['Return on Assets'].mean():.1f}% vs market" if not stats.empty else None
    )

with col3:
    avg_margin = stats_filtered['Profit Margin'].mean()
    st.metric(
        "Avg Profit Margin",
        f"{avg_margin:.1f}%",
        delta=f"{avg_margin - stats['Profit Margin'].mean():.1f}% vs market" if not stats.empty else None
    )

with col4:
    if not revenue_filtered.empty:
        latest_year_data = revenue_filtered[revenue_filtered['year'] == revenue_filtered['year'].max()]
        total_revenue = latest_year_data['revenue'].sum()
        if total_revenue >= 1e12:
            rev_display = f"${total_revenue/1e12:.2f}T"
        elif total_revenue >= 1e9:
            rev_display = f"${total_revenue/1e9:.0f}B"
        else:
            rev_display = f"${total_revenue/1e6:.0f}M"
        st.metric("Total Revenue", rev_display)
    else:
        st.metric("Total Revenue", "N/A")

with col5:
    num_stocks = len(selected_tickers)
    st.metric("Stocks Analyzed", num_stocks)

st.markdown("---")

# =============================================================================
# SECTION 2: Performance Analysis
# =============================================================================
st.markdown("## 💰 Performance Analysis")

# ROE vs ROA with Regression
col1, col2 = st.columns(2)

with col1:
    st.subheader("ROE vs ROA Analysis")

    scatter_data = stats_filtered.dropna(subset=['Return on Assets', 'Return on Equity', 'Profit Margin'])

    if len(scatter_data) > 1:
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            scatter_data['Return on Assets'],
            scatter_data['Return on Equity']
        )

        fig_scatter = px.scatter(
            scatter_data,
            x='Return on Assets',
            y='Return on Equity',
            text='ticker',
            size='Profit Margin',
            color='Profit Margin',
            color_continuous_scale='Viridis',
            trendline='ols',
            title=f'ROE vs ROA (R² = {r_value**2:.3f})'
        )
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, width='stretch')

        # Display regression stats
        st.markdown(f'<div class="correlation-note">📊 Regression: ROE = {slope:.2f} × ROA + {intercept:.2f} | p-value: {p_value:.4f}</div>', unsafe_allow_html=True)
    else:
        st.info("Need more data points for regression analysis")

with col2:
    st.subheader("Performance Ranking")

    # Create composite score - filter out NaN values first
    ranking_data = stats_filtered.dropna(subset=['Return on Equity', 'Return on Assets', 'Profit Margin']).copy()

    if not ranking_data.empty:
        # Normalize metrics (0-100 scale)
        for col in ['Return on Equity', 'Return on Assets', 'Profit Margin']:
            min_val = ranking_data[col].min()
            max_val = ranking_data[col].max()
            if max_val > min_val:
                ranking_data[f'{col}_norm'] = (ranking_data[col] - min_val) / (max_val - min_val) * 100
            else:
                ranking_data[f'{col}_norm'] = 50

        # Calculate composite score
        ranking_data['Composite Score'] = (
            ranking_data['Return on Equity_norm'] * 0.4 +
            ranking_data['Return on Assets_norm'] * 0.3 +
            ranking_data['Profit Margin_norm'] * 0.3
        )

        ranking_data = ranking_data.sort_values('Composite Score', ascending=True)

        fig_rank = px.bar(
            ranking_data,
            x='Composite Score',
            y='ticker',
            orientation='h',
            color='Composite Score',
            color_continuous_scale='RdYlGn',
            title='Composite Performance Score'
        )
        fig_rank.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_rank, width='stretch')

        st.markdown('<div class="correlation-note">📝 Score = 40% ROE + 30% ROA + 30% Profit Margin</div>', unsafe_allow_html=True)
    else:
        st.info("No complete data available for performance ranking")

st.markdown("---")

# =============================================================================
# SECTION 4: Profitability Deep Dive
# =============================================================================
st.markdown("## 📈 Profitability Analysis")

if not revenue_filtered.empty and not profit_filtered.empty:
    latest_year_rev = revenue_filtered[revenue_filtered['year'] == revenue_filtered['year'].max()]
    latest_year_profit = profit_filtered[profit_filtered['year'] == profit_filtered['year'].max()]

    profitability_year = latest_year_rev.merge(
        latest_year_profit[['ticker', 'profit']],
        on='ticker',
        how='inner'
    )
    profitability_year['Profit Margin %'] = (profitability_year['profit'] / profitability_year['revenue'] * 100)
    profitability_year = profitability_year.rename(columns={'revenue': 'Revenue', 'profit': 'Gross Profit'})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Revenue Analysis ({int(revenue_filtered['year'].max())})")

        # Treemap for revenue
        fig_treemap = px.treemap(
            profitability_year,
            path=['ticker'],
            values='Revenue',
            color='Profit Margin %',
            color_continuous_scale='RdYlGn',
            title='Revenue Distribution (Size) & Profit Margin (Color)'
        )
        fig_treemap.update_layout(height=500)
        st.plotly_chart(fig_treemap, width='stretch')

    with col2:
        st.subheader("Profit Margin Comparison")

        fig_margin = px.bar(
            profitability_year.sort_values('Profit Margin %', ascending=True),
            x='Profit Margin %',
            y='ticker',
            orientation='h',
            color='Profit Margin %',
            color_continuous_scale='Blues',
            title='Profit Margin by Stock'
        )
        fig_margin.update_layout(height=500, showlegend=False)
        fig_margin.add_vline(x=profitability_year['Profit Margin %'].mean(),
                            line_dash="dash", line_color="red",
                            annotation_text=f"Avg: {profitability_year['Profit Margin %'].mean():.1f}%")
        st.plotly_chart(fig_margin, width='stretch')

else:
    st.warning("No data available for the selected year range")

st.markdown("---")

# =============================================================================
# SECTION 5: Growth Analysis
# =============================================================================
st.markdown("## 📉 Growth Trajectory Analysis")

# Prepare data
revenue_melted = revenue_filtered.copy()
revenue_melted.columns = ['ticker', 'year', 'value']
revenue_melted = revenue_melted.sort_values(['ticker', 'year'])
revenue_melted['growth'] = revenue_melted.groupby('ticker')['value'].pct_change() * 100

profit_melted = profit_filtered.copy()
profit_melted.columns = ['ticker', 'year', 'value']
profit_melted = profit_melted.sort_values(['ticker', 'year'])
profit_melted['growth'] = profit_melted.groupby('ticker')['value'].pct_change() * 100

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue Trends")

    # Create figure with historical data
    fig_rev = go.Figure()

    # Add historical lines for each ticker
    for ticker in revenue_melted['ticker'].unique():
        ticker_data = revenue_melted[revenue_melted['ticker'] == ticker]
        fig_rev.add_trace(go.Scatter(
            x=ticker_data['year'],
            y=ticker_data['value'],
            mode='lines+markers',
            name=ticker
        ))

        # Add forecast for next year
        if len(ticker_data) >= 2:
            years = ticker_data['year'].values
            values = ticker_data['value'].values

            # Linear regression for forecast
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(years, values)

            # Forecast next year
            next_year = years.max() + 1
            forecast_value = slope * next_year + intercept

            # Add forecast point with dashed line
            fig_rev.add_trace(go.Scatter(
                x=[years.max(), next_year],
                y=[values[-1], forecast_value],
                mode='lines+markers',
                name=f'{ticker} Forecast',
                line=dict(dash='dash'),
                marker=dict(symbol='star'),
                showlegend=False
            ))

    fig_rev.update_layout(
        title='Revenue Over Time (with 1Y Forecast)',
        height=450,
        yaxis_title='Revenue (USD)',
        yaxis_tickformat=',.0f',
        hovermode='x unified'
    )
    st.plotly_chart(fig_rev, width='stretch')

with col2:
    st.subheader("Gross Profit Trends")

    # Create figure with historical data
    fig_profit = go.Figure()

    # Add historical lines for each ticker
    for ticker in profit_melted['ticker'].unique():
        ticker_data = profit_melted[profit_melted['ticker'] == ticker]
        fig_profit.add_trace(go.Scatter(
            x=ticker_data['year'],
            y=ticker_data['value'],
            mode='lines+markers',
            name=ticker
        ))

        # Add forecast for next year
        if len(ticker_data) >= 2:
            years = ticker_data['year'].values
            values = ticker_data['value'].values

            # Linear regression for forecast
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(years, values)

            # Forecast next year
            next_year = years.max() + 1
            forecast_value = slope * next_year + intercept

            # Add forecast point with dashed line
            fig_profit.add_trace(go.Scatter(
                x=[years.max(), next_year],
                y=[values[-1], forecast_value],
                mode='lines+markers',
                name=f'{ticker} Forecast',
                line=dict(dash='dash'),
                marker=dict(symbol='star'),
                showlegend=False
            ))

    fig_profit.update_layout(
        title='Gross Profit Over Time (with 1Y Forecast)',
        height=450,
        yaxis_title='Gross Profit (USD)',
        yaxis_tickformat=',.0f',
        hovermode='x unified'
    )
    st.plotly_chart(fig_profit, width='stretch')

# CAGR Analysis
st.subheader("Compound Annual Growth Rate (CAGR)")

if not revenue_filtered.empty:
    revenue_wide = revenue_filtered.pivot(index='ticker', columns='year', values='revenue').reset_index()
    revenue_wide.columns = ['ticker'] + [str(int(col)) if isinstance(col, (int, float)) else str(col) for col in revenue_wide.columns[1:]]

    year_cols = [col for col in revenue_wide.columns if col != 'ticker']

    if len(year_cols) >= 2:
        sorted_years = sorted(year_cols)
        first_year = sorted_years[0]
        last_year = sorted_years[-1]
        n_years = int(last_year) - int(first_year)

        if n_years > 0:
            # Calculate CAGR
            cagr_data = revenue_wide[['ticker', first_year, last_year]].copy()
            cagr_data['CAGR %'] = ((cagr_data[last_year] / cagr_data[first_year]) ** (1/n_years) - 1) * 100
            cagr_data = cagr_data.dropna().sort_values('CAGR %', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                fig_cagr = px.bar(
                    cagr_data,
                    x='ticker',
                    y='CAGR %',
                    color='CAGR %',
                    color_continuous_scale='RdYlGn',
                    title=f'Revenue CAGR ({first_year}-{last_year})'
                )
                fig_cagr.update_layout(height=400)
                fig_cagr.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_cagr, width='stretch')

            with col2:
                # Growth classification
                st.markdown("#### Growth Classification")

                high_growth = cagr_data[cagr_data['CAGR %'] > 20]['ticker'].tolist()
                moderate_growth = cagr_data[(cagr_data['CAGR %'] > 5) & (cagr_data['CAGR %'] <= 20)]['ticker'].tolist()
                low_growth = cagr_data[(cagr_data['CAGR %'] > 0) & (cagr_data['CAGR %'] <= 5)]['ticker'].tolist()
                declining = cagr_data[cagr_data['CAGR %'] <= 0]['ticker'].tolist()

                st.markdown(f"🚀 **High Growth (>20%):** {', '.join(high_growth) if high_growth else 'None'}")
                st.markdown(f"📈 **Moderate Growth (5-20%):** {', '.join(moderate_growth) if moderate_growth else 'None'}")
                st.markdown(f"📊 **Low Growth (0-5%):** {', '.join(low_growth) if low_growth else 'None'}")
                st.markdown(f"📉 **Declining (<0%):** {', '.join(declining) if declining else 'None'}")

st.markdown("---")

# =============================================================================
# SECTION 6: Stock Price Chart
# =============================================================================
st.markdown("## 📊 Stock Price Chart")

# Prepare stock price data
stocks_df = stocks.copy()
stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
stocks_df = stocks_df.sort_values('Date')

# Get available tickers in stocks.csv
available_price_tickers = [col for col in stocks_df.columns if col not in ['id', 'Date']]
price_tickers = [t for t in selected_tickers if t in available_price_tickers]

if price_tickers:
    # Single ticker selection for candlestick
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        selected_price_ticker = st.selectbox(
            "Select Stock for Price Chart",
            options=price_tickers,
            index=0,
            help="Select a stock to view its price chart"
        )

    with col2:
        candle_interval = st.selectbox(
            "Candle Interval",
            options=['Daily', 'Monthly', 'Yearly'],
            index=0,
            help="Select candlestick time interval"
        )

    with col3:
        # Date range slider - adjust based on candle interval
        available_dates = sorted(stocks_df['Date'].dropna().unique())
        if len(available_dates) >= 2:
            min_date = available_dates[0]
            max_date = available_dates[-1]

            # Create date options based on candle interval
            if candle_interval == 'Daily':
                # Weekly intervals for daily view
                date_options = pd.date_range(start=min_date, end=max_date, freq='W').tolist()
                date_format = '%d %b %Y'
            elif candle_interval == 'Monthly':
                # Monthly intervals
                date_options = pd.date_range(start=min_date, end=max_date, freq='MS').tolist()
                date_format = '%b %Y'
            else:  # Yearly
                # Yearly intervals
                date_options = pd.date_range(start=min_date, end=max_date, freq='YS').tolist()
                date_format = '%Y'

            if max_date not in date_options:
                date_options.append(max_date)

            start_date, end_date = st.select_slider(
                "Select Date Range",
                options=date_options,
                value=(date_options[0], date_options[-1]),
                format_func=lambda x: x.strftime(date_format),
                help="Filter price data by date range"
            )

            # Filter stocks data by date range
            stocks_df = stocks_df[(stocks_df['Date'] >= start_date) & (stocks_df['Date'] <= end_date)]

    # Prepare candlestick data
    ticker_data = stocks_df[['Date', selected_price_ticker]].copy()
    ticker_data = ticker_data.dropna()
    ticker_data.columns = ['Date', 'Close']

    # Aggregate based on selected interval
    if candle_interval == 'Monthly':
        ticker_data['Period'] = ticker_data['Date'].dt.to_period('M')
        agg_data = ticker_data.groupby('Period').agg({
            'Close': ['first', 'max', 'min', 'last']
        }).reset_index()
        agg_data.columns = ['Period', 'Open', 'High', 'Low', 'Close']
        agg_data['Date'] = agg_data['Period'].dt.to_timestamp()
        ticker_data = agg_data[['Date', 'Open', 'High', 'Low', 'Close']]
    elif candle_interval == 'Yearly':
        ticker_data['Period'] = ticker_data['Date'].dt.to_period('Y')
        agg_data = ticker_data.groupby('Period').agg({
            'Close': ['first', 'max', 'min', 'last']
        }).reset_index()
        agg_data.columns = ['Period', 'Open', 'High', 'Low', 'Close']
        agg_data['Date'] = agg_data['Period'].dt.to_timestamp()
        ticker_data = agg_data[['Date', 'Open', 'High', 'Low', 'Close']]
    else:
        # Daily - simulate OHLC by using consecutive days
        ticker_data['Open'] = ticker_data['Close'].shift(1)
        ticker_data['High'] = ticker_data[['Open', 'Close']].max(axis=1)
        ticker_data['Low'] = ticker_data[['Open', 'Close']].min(axis=1)

        # Add some variance to High/Low for better visualization
        ticker_data['High'] = ticker_data['High'] * 1.005
        ticker_data['Low'] = ticker_data['Low'] * 0.995

        # Remove first row (no previous day for Open)
        ticker_data = ticker_data.dropna()

    # Create candlestick chart
    fig_candle = go.Figure(data=[go.Candlestick(
        x=ticker_data['Date'],
        open=ticker_data['Open'],
        high=ticker_data['High'],
        low=ticker_data['Low'],
        close=ticker_data['Close'],
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        name=selected_price_ticker
    )])

    fig_candle.update_layout(
        title=f'{selected_price_ticker} Stock Price',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=500,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified'
    )

    st.plotly_chart(fig_candle, width='stretch')

    # Display price statistics
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5, col_stat6 = st.columns(6)

    # Get original unfiltered data for return calculations
    original_ticker_data = stocks.copy()
    original_ticker_data['Date'] = pd.to_datetime(original_ticker_data['Date'])
    original_ticker_data = original_ticker_data[['Date', selected_price_ticker]].dropna()
    original_ticker_data = original_ticker_data.sort_values('Date')
    original_ticker_data.columns = ['Date', 'Close']

    latest_date = original_ticker_data['Date'].max()
    latest_price = original_ticker_data['Close'].iloc[-1]

    # Calculate returns for different periods
    def get_return(days):
        target_date = latest_date - pd.Timedelta(days=days)
        past_data = original_ticker_data[original_ticker_data['Date'] <= target_date]
        if not past_data.empty:
            past_price = past_data['Close'].iloc[-1]
            return ((latest_price / past_price) - 1) * 100
        return None

    # YTD calculation
    year_start = pd.Timestamp(year=latest_date.year, month=1, day=1)
    ytd_data = original_ticker_data[original_ticker_data['Date'] <= year_start]
    ytd_return = ((latest_price / ytd_data['Close'].iloc[-1]) - 1) * 100 if not ytd_data.empty else None

    with col_stat1:
        st.metric("Latest Price", f"${latest_price:.2f}")

    with col_stat2:
        return_1w = get_return(7)
        if return_1w is not None:
            st.metric("1W Return", f"{return_1w:.1f}%", delta=f"{return_1w:.1f}%")
        else:
            st.metric("1W Return", "N/A")

    with col_stat3:
        return_1m = get_return(30)
        if return_1m is not None:
            st.metric("1M Return", f"{return_1m:.1f}%", delta=f"{return_1m:.1f}%")
        else:
            st.metric("1M Return", "N/A")

    with col_stat4:
        return_3m = get_return(90)
        if return_3m is not None:
            st.metric("3M Return", f"{return_3m:.1f}%", delta=f"{return_3m:.1f}%")
        else:
            st.metric("3M Return", "N/A")

    with col_stat5:
        if ytd_return is not None:
            st.metric("YTD Return", f"{ytd_return:.1f}%", delta=f"{ytd_return:.1f}%")
        else:
            st.metric("YTD Return", "N/A")

    with col_stat6:
        return_1y = get_return(365)
        if return_1y is not None:
            st.metric("1Y Return", f"{return_1y:.1f}%", delta=f"{return_1y:.1f}%")
        else:
            st.metric("1Y Return", "N/A")

else:
    st.warning("No price data available for selected stocks")

st.markdown("---")

# =============================================================================
# SECTION 7: Key Insights & Recommendations
# =============================================================================
st.markdown("## 🔍 Key Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    # Top performers by ROE
    top_roe = stats_filtered.nlargest(3, 'Return on Equity')[['ticker', 'Return on Equity']]
    roe_list = " | ".join([f"<strong>{row['ticker']}</strong> {row['Return on Equity']:.1f}%" for _, row in top_roe.iterrows()])

    # Highest Profit Margins
    top_margin = stats_filtered.nlargest(3, 'Profit Margin')[['ticker', 'Profit Margin']]
    margin_list = " | ".join([f"<strong>{row['ticker']}</strong> {row['Profit Margin']:.1f}%" for _, row in top_margin.iterrows()])

    st.markdown(f"""
    <div class="insight-box">
    <h4>Portfolio Summary</h4>
    <p><strong>Top ROE:</strong> {roe_list}</p>
    <p><strong>Top Margin:</strong> {margin_list}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Undervalued stocks
    undervalued = valuation_filtered[valuation_filtered['PEG'] < 1].dropna(subset=['PEG'])
    if not undervalued.empty:
        undervalued_list = " | ".join([f"<strong>{row['ticker']}</strong> PEG={row['PEG']:.2f}" for _, row in undervalued.iterrows()])
    else:
        undervalued_list = "No stocks with PEG < 1"

    # Risk indicators
    high_pe = valuation_filtered[valuation_filtered['Trailing P/E'] > 50].dropna(subset=['Trailing P/E'])
    if not high_pe.empty:
        high_pe_list = " | ".join([f"<strong>{row['ticker']}</strong> P/E={row['Trailing P/E']:.1f}" for _, row in high_pe.head(3).iterrows()])
    else:
        high_pe_list = "None"

    st.markdown(f"""
    <div class="insight-box">
    <h4>Investment Considerations</h4>
    <p><strong>Undervalued (PEG &lt; 1):</strong> {undervalued_list}</p>
    <p><strong>High P/E (&gt; 50):</strong> {high_pe_list}</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1.5rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;'>
    <p style='font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;'>📊 Fundamental Stock Analysis Dashboard</p>
    <p style='margin-bottom: 0.5rem;'>Built with Streamlit & Plotly | Data Source: Kaggle</p>
</div>
""", unsafe_allow_html=True)
