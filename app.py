import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats

st.set_page_config(
    page_title="Fundamental Stock Data Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Fundamental Stock Analysis Dashboard</div>', unsafe_allow_html=True)

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

st.sidebar.markdown("## 🎛️ Dashboard Controls")
st.sidebar.markdown("---")

all_tickers = stats['ticker'].dropna().unique().tolist()

st.sidebar.markdown("### 📊 Stock Selection")
selected_tickers = st.sidebar.multiselect(
    "Select Stocks for Analysis",
    options=all_tickers,
    default=['MSFT', 'TSLA', 'GOOGL', 'META', 'AMZN'] if 'MSFT' in all_tickers else all_tickers[:5],
    help="Select one or more stocks to analyze"
)

if not selected_tickers:
    selected_tickers = all_tickers[:10]

st.sidebar.info(f"📌 {len(selected_tickers)} stocks selected")

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

stats_filtered = stats[stats['ticker'].isin(selected_tickers)]
valuation_filtered = valuation[valuation['ticker'].isin(selected_tickers)]
revenue_filtered = revenue[(revenue['ticker'].isin(selected_tickers)) & (revenue['year'].isin(selected_years))]
profit_filtered = profit[(profit['ticker'].isin(selected_tickers)) & (profit['year'].isin(selected_years))]
assets_filtered = assets[(assets['ticker'].isin(selected_tickers)) & (assets['year'].isin(selected_years))]
equity_filtered = equity[(equity['ticker'].isin(selected_tickers)) & (equity['year'].isin(selected_years))]
cash_filtered = cash[(cash['ticker'].isin(selected_tickers)) & (cash['year'].isin(selected_years))]

# Executive Summary
st.markdown("## 📈 Executive Summary")

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

# Performance Analysis
st.markdown("## 💰 Performance Analysis")
st.subheader("Performance Ranking")

col1, col2, col3 = st.columns(3)

performance_data = stats_filtered.dropna(subset=['Return on Equity', 'Return on Assets', 'Profit Margin']).copy()

with col1:
    if not performance_data.empty:
        top_roe = performance_data.nlargest(10, 'Return on Equity')[['ticker', 'Return on Equity']]
        top_roe = top_roe.sort_values('Return on Equity', ascending=True)

        fig_top_roe = px.bar(
            top_roe,
            x='Return on Equity',
            y='ticker',
            orientation='h',
            color='Return on Equity',
            color_continuous_scale='Blues',
            title='Top Performance by ROE'
        )
        fig_top_roe.update_layout(
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
            title=dict(text='Top Performance by ROE', x=0, xanchor='left'),
            xaxis_title='Return on Equity (%)',
            yaxis_title='Stock'
        )
        st.plotly_chart(fig_top_roe, use_container_width=True)
    else:
        st.info("No ROE data available")

with col2:
    if not performance_data.empty:
        top_roa = performance_data.nlargest(10, 'Return on Assets')[['ticker', 'Return on Assets']]
        top_roa = top_roa.sort_values('Return on Assets', ascending=True)

        fig_top_roa = px.bar(
            top_roa,
            x='Return on Assets',
            y='ticker',
            orientation='h',
            color='Return on Assets',
            color_continuous_scale='Greens',
            title='Top Performance by ROA'
        )
        fig_top_roa.update_layout(
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
            title=dict(text='Top Performance by ROA', x=0, xanchor='left'),
            xaxis_title='Return on Assets (%)',
            yaxis_title='Stock'
        )
        st.plotly_chart(fig_top_roa, use_container_width=True)
    else:
        st.info("No ROA data available")

with col3:
    if not performance_data.empty:
        top_margin = performance_data.nlargest(10, 'Profit Margin')[['ticker', 'Profit Margin']]
        top_margin = top_margin.sort_values('Profit Margin', ascending=True)

        fig_top_margin = px.bar(
            top_margin,
            x='Profit Margin',
            y='ticker',
            orientation='h',
            color='Profit Margin',
            color_continuous_scale='Oranges',
            title='Top Performance by Profit Margin'
        )
        fig_top_margin.update_layout(
            height=400,
            showlegend=False,
            coloraxis_showscale=False,
            title=dict(text='Top Performance by Profit Margin', x=0, xanchor='left'),
            xaxis_title='Profit Margin (%)',
            yaxis_title='Stock'
        )
        st.plotly_chart(fig_top_margin, use_container_width=True)
    else:
        st.info("No Profit Margin data available")

st.markdown("---")

# Growth Trajectory Analysis
st.markdown("## 📉 Growth Trajectory Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue & YoY Growth")

    if not revenue_filtered.empty:
        revenue_yoy = revenue_filtered.copy()
        revenue_yoy = revenue_yoy.sort_values(['ticker', 'year'])
        revenue_yoy['YoY Growth %'] = revenue_yoy.groupby('ticker')['revenue'].pct_change() * 100

        revenue_agg = revenue_yoy.groupby('year').agg({
            'revenue': 'sum',
            'YoY Growth %': 'mean'
        }).reset_index()

        fig_rev_yoy = go.Figure()

        fig_rev_yoy.add_trace(go.Bar(
            x=revenue_agg['year'],
            y=revenue_agg['revenue'],
            name='Total Revenue',
            marker_color='#667eea',
            yaxis='y'
        ))

        if len(revenue_agg) >= 2:
            years = revenue_agg['year'].values
            revenues = revenue_agg['revenue'].values
            slope, intercept, _, _, _ = scipy_stats.linregress(years, revenues)
            next_year = int(years.max()) + 1
            forecast_revenue = slope * next_year + intercept

            fig_rev_yoy.add_trace(go.Bar(
                x=[next_year],
                y=[forecast_revenue],
                name='Forecast Revenue',
                marker_color='#667eea',
                opacity=0.5,
                yaxis='y',
                showlegend=False
            ))

            fig_rev_yoy.add_annotation(
                x=next_year,
                y=forecast_revenue,
                text="Forecast",
                showarrow=False,
                yshift=15,
                font=dict(size=10)
            )

        fig_rev_yoy.add_trace(go.Scatter(
            x=revenue_agg['year'],
            y=revenue_agg['YoY Growth %'],
            name='YoY Growth %',
            mode='lines+markers',
            marker=dict(size=10, color='#ff6b6b'),
            line=dict(width=3, color='#ff6b6b'),
            yaxis='y2'
        ))

        if len(revenue_agg) >= 2:
            yoy_values = revenue_agg['YoY Growth %'].dropna().values
            yoy_years = revenue_agg[revenue_agg['YoY Growth %'].notna()]['year'].values
            if len(yoy_values) >= 2:
                slope_yoy, intercept_yoy, _, _, _ = scipy_stats.linregress(yoy_years, yoy_values)
                next_year = int(revenue_agg['year'].max()) + 1
                forecast_yoy = slope_yoy * next_year + intercept_yoy
                last_yoy = revenue_agg['YoY Growth %'].dropna().iloc[-1]
                last_year = int(revenue_agg[revenue_agg['YoY Growth %'].notna()]['year'].iloc[-1])

                fig_rev_yoy.add_trace(go.Scatter(
                    x=[last_year, next_year],
                    y=[last_yoy, forecast_yoy],
                    mode='lines',
                    line=dict(dash='dash', width=2, color='rgba(255, 107, 107, 0.5)'),
                    yaxis='y2',
                    showlegend=False
                ))

                fig_rev_yoy.add_trace(go.Scatter(
                    x=[next_year],
                    y=[forecast_yoy],
                    mode='markers',
                    marker=dict(size=10, color='#ff6b6b', opacity=0.5),
                    yaxis='y2',
                    showlegend=False
                ))

        fig_rev_yoy.update_layout(
            height=400,
            yaxis=dict(
                title=dict(text='Revenue (USD)', font=dict(color='#667eea')),
                tickfont=dict(color='#667eea'),
                tickformat=',.0f'
            ),
            yaxis2=dict(
                title=dict(text='YoY Growth %', font=dict(color='#ff6b6b')),
                tickfont=dict(color='#ff6b6b'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified'
        )

        st.plotly_chart(fig_rev_yoy, use_container_width=True)
    else:
        st.info("No revenue data available for selected stocks")

with col2:
    st.subheader("Net Income & YoY Growth")

    if not profit_filtered.empty:
        profit_yoy = profit_filtered.copy()
        profit_yoy = profit_yoy.sort_values(['ticker', 'year'])
        profit_yoy['YoY Growth %'] = profit_yoy.groupby('ticker')['profit'].pct_change() * 100

        profit_agg = profit_yoy.groupby('year').agg({
            'profit': 'sum',
            'YoY Growth %': 'mean'
        }).reset_index()

        fig_profit_yoy = go.Figure()

        fig_profit_yoy.add_trace(go.Bar(
            x=profit_agg['year'],
            y=profit_agg['profit'],
            name='Total Net Income',
            marker_color='#764ba2',
            yaxis='y'
        ))

        if len(profit_agg) >= 2:
            years = profit_agg['year'].values
            profits = profit_agg['profit'].values
            slope, intercept, _, _, _ = scipy_stats.linregress(years, profits)
            next_year = int(years.max()) + 1
            forecast_profit = slope * next_year + intercept

            fig_profit_yoy.add_trace(go.Bar(
                x=[next_year],
                y=[forecast_profit],
                name='Forecast Net Income',
                marker_color='#764ba2',
                opacity=0.5,
                yaxis='y',
                showlegend=False
            ))

            fig_profit_yoy.add_annotation(
                x=next_year,
                y=forecast_profit,
                text="Forecast",
                showarrow=False,
                yshift=15,
                font=dict(size=10)
            )

        fig_profit_yoy.add_trace(go.Scatter(
            x=profit_agg['year'],
            y=profit_agg['YoY Growth %'],
            name='YoY Growth %',
            mode='lines+markers',
            marker=dict(size=10, color='#feca57'),
            line=dict(width=3, color='#feca57'),
            yaxis='y2'
        ))

        if len(profit_agg) >= 2:
            yoy_values = profit_agg['YoY Growth %'].dropna().values
            yoy_years = profit_agg[profit_agg['YoY Growth %'].notna()]['year'].values
            if len(yoy_values) >= 2:
                slope_yoy, intercept_yoy, _, _, _ = scipy_stats.linregress(yoy_years, yoy_values)
                next_year = int(profit_agg['year'].max()) + 1
                forecast_yoy = slope_yoy * next_year + intercept_yoy
                last_yoy = profit_agg['YoY Growth %'].dropna().iloc[-1]
                last_year = int(profit_agg[profit_agg['YoY Growth %'].notna()]['year'].iloc[-1])

                fig_profit_yoy.add_trace(go.Scatter(
                    x=[last_year, next_year],
                    y=[last_yoy, forecast_yoy],
                    mode='lines',
                    line=dict(dash='dash', width=2, color='rgba(254, 202, 87, 0.5)'),
                    yaxis='y2',
                    showlegend=False
                ))

                fig_profit_yoy.add_trace(go.Scatter(
                    x=[next_year],
                    y=[forecast_yoy],
                    mode='markers',
                    marker=dict(size=10, color='#feca57', opacity=0.5),
                    yaxis='y2',
                    showlegend=False
                ))

        fig_profit_yoy.update_layout(
            height=400,
            yaxis=dict(
                title=dict(text='Net Income (USD)', font=dict(color='#764ba2')),
                tickfont=dict(color='#764ba2'),
                tickformat=',.0f'
            ),
            yaxis2=dict(
                title=dict(text='YoY Growth %', font=dict(color='#feca57')),
                tickfont=dict(color='#feca57'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified'
        )

        st.plotly_chart(fig_profit_yoy, use_container_width=True)
    else:
        st.info("No profit data available for selected stocks")

# Assets vs Equity
st.subheader("Assets vs Equity Trend")

if not assets_filtered.empty and not equity_filtered.empty:
    assets_equity = assets_filtered.merge(
        equity_filtered[['ticker', 'year', 'equity']],
        on=['ticker', 'year'],
        how='inner'
    )

    assets_equity_agg = assets_equity.groupby('year').agg({
        'assets': 'sum',
        'equity': 'sum'
    }).reset_index()

    fig_assets_equity = go.Figure()

    fig_assets_equity.add_trace(go.Bar(
        x=assets_equity_agg['year'],
        y=assets_equity_agg['equity'],
        name='Total Equity',
        marker_color='#0984e3'
    ))

    fig_assets_equity.add_trace(go.Bar(
        x=assets_equity_agg['year'],
        y=assets_equity_agg['assets'],
        name='Total Assets',
        marker_color='#00b894'
    ))

    fig_assets_equity.update_layout(
        height=400,
        barmode='group',
        yaxis=dict(
            title='Amount (USD)',
            tickformat=',.0f'
        ),
        xaxis_title='',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        hovermode='x unified',
        margin=dict(l=20, r=20, t=40, b=40)
    )

    st.plotly_chart(fig_assets_equity, use_container_width=True)
else:
    st.info("No assets or equity data available for selected stocks")

# CAGR Analysis
col1, col2 = st.columns([4, 1])

with col1:
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
                cagr_data = revenue_wide[['ticker', first_year, last_year]].copy()
                cagr_data['CAGR %'] = ((cagr_data[last_year] / cagr_data[first_year]) ** (1/n_years) - 1) * 100
                cagr_data = cagr_data.dropna().sort_values('CAGR %', ascending=False)

                fig_cagr = px.bar(
                    cagr_data,
                    x='ticker',
                    y='CAGR %',
                    color='CAGR %',
                    color_continuous_scale='RdYlGn'
                )
                fig_cagr.update_layout(
                    height=400,
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=20, t=40, b=40),
                    xaxis_title='',
                    yaxis_title='CAGR %'
                )
                fig_cagr.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_cagr, use_container_width=True)

with col2:
    st.subheader("Growth Classification")

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
                cagr_data = revenue_wide[['ticker', first_year, last_year]].copy()
                cagr_data['CAGR %'] = ((cagr_data[last_year] / cagr_data[first_year]) ** (1/n_years) - 1) * 100
                cagr_data = cagr_data.dropna().sort_values('CAGR %', ascending=False)

                high_growth = cagr_data[cagr_data['CAGR %'] > 20]['ticker'].tolist()
                moderate_growth = cagr_data[(cagr_data['CAGR %'] > 5) & (cagr_data['CAGR %'] <= 20)]['ticker'].tolist()
                low_growth = cagr_data[(cagr_data['CAGR %'] > 0) & (cagr_data['CAGR %'] <= 5)]['ticker'].tolist()
                declining = cagr_data[cagr_data['CAGR %'] <= 0]['ticker'].tolist()

                st.markdown(f"🚀 **High Growth (>20%)**")
                st.markdown(f"{', '.join(high_growth) if high_growth else 'None'}")
                st.markdown("")
                st.markdown(f"📈 **Moderate Growth (5-20%)**")
                st.markdown(f"{', '.join(moderate_growth) if moderate_growth else 'None'}")
                st.markdown("")
                st.markdown(f"📊 **Low Growth (0-5%)**")
                st.markdown(f"{', '.join(low_growth) if low_growth else 'None'}")
                st.markdown("")
                st.markdown(f"📉 **Declining (<0%)**")
                st.markdown(f"{', '.join(declining) if declining else 'None'}")

st.markdown("---")

# Valuation Analysis
st.markdown("## 💎 Valuation Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("PEG Ratio by Stock")

    if not valuation_filtered.empty:
        peg_data = valuation_filtered[['ticker', 'PEG']].dropna().copy()
        peg_data = peg_data.sort_values('PEG', ascending=True)

        if not peg_data.empty:
            fig_peg = px.bar(
                peg_data,
                x='ticker',
                y='PEG',
                color='PEG',
                color_continuous_scale='RdYlGn_r',
                title=''
            )
            fig_peg.update_layout(
                height=400,
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=40, b=40),
                xaxis_title='',
                yaxis_title='PEG Ratio'
            )
            fig_peg.add_hline(y=1, line_dash="dash", line_color="gray",
                            annotation_text="Fair Value (PEG=1)",
                            annotation_position="top right")
            st.plotly_chart(fig_peg, use_container_width=True)
        else:
            st.info("No PEG data available")
    else:
        st.info("No valuation data available")

with col2:
    st.subheader("Trailing P/E Ratio by Stock")

    if not valuation_filtered.empty:
        pe_data = valuation_filtered[['ticker', 'Trailing P/E']].dropna().copy()
        pe_data = pe_data.sort_values('Trailing P/E', ascending=True)

        if not pe_data.empty:
            fig_pe = px.bar(
                pe_data,
                x='ticker',
                y='Trailing P/E',
                color='Trailing P/E',
                color_continuous_scale='RdYlGn_r',
                title=''
            )
            fig_pe.update_layout(
                height=400,
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=40, b=40),
                xaxis_title='',
                yaxis_title='P/E Ratio'
            )
            fig_pe.add_hline(y=25, line_dash="dash", line_color="gray",
                           annotation_text="Market Avg (P/E=25)",
                           annotation_position="top right")
            fig_pe.add_hline(y=50, line_dash="dot", line_color="red",
                           annotation_text="High (P/E=50)",
                           annotation_position="top right")
            st.plotly_chart(fig_pe, use_container_width=True)
        else:
            st.info("No P/E data available")
    else:
        st.info("No valuation data available")

st.markdown("---")

# Key Insights & Recommendations
st.markdown("## 🔍 Key Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    top_roe = stats_filtered.nlargest(3, 'Return on Equity')[['ticker', 'Return on Equity']]
    roe_list = " | ".join([f"<strong>{row['ticker']}</strong> {row['Return on Equity']:.1f}%" for _, row in top_roe.iterrows()])

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
    undervalued = valuation_filtered[valuation_filtered['PEG'] < 1].dropna(subset=['PEG'])
    if not undervalued.empty:
        undervalued_list = " | ".join([f"<strong>{row['ticker']}</strong> PEG={row['PEG']:.2f}" for _, row in undervalued.iterrows()])
    else:
        undervalued_list = "No stocks with PEG < 1"

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
    <p style='margin-bottom: 0;'>Created by Guntur Adi Wardana</p>
</div>
""", unsafe_allow_html=True)
