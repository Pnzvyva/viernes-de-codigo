import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, t
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 8: Gestión de Riesgos")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Tipos de Riesgo", "Valor en Riesgo (VaR)", "Pruebas de Estrés", "Métricas de Riesgo"])
    
    with tab1:
        st.subheader("Understanding Financial Risk")
        
        st.write("""
        Risk management is the process of identifying, analyzing, and controlling threats to an organization's 
        capital and earnings. In finance, risk comes in many forms and must be carefully measured and managed.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Market Risk:**
            - Interest rate risk
            - Equity price risk
            - Currency risk
            - Commodity risk
            - Volatility risk
            """)
        
        with col2:
            st.write("""
            **Credit Risk:**
            - Default risk
            - Credit spread risk
            - Counterparty risk
            - Settlement risk
            """)
        
        st.write("""
        **Other Risk Types:**
        - **Operational Risk**: Internal processes, systems, or human errors
        - **Liquidity Risk**: Inability to meet short-term obligations
        - **Legal Risk**: Losses from legal or regulatory issues
        - **Reputation Risk**: Loss of confidence affecting business
        """)
        
        st.subheader("Risk Measurement Framework")
        
        risk_measures = {
            "Standard Deviation": {
                "Description": "Measures dispersion of returns around the mean",
                "Formula": "σ = √(Σ(r - μ)² / (n-1))",
                "Use Case": "Basic volatility measure"
            },
            "Beta": {
                "Description": "Measures systematic risk relative to market",
                "Formula": "β = Cov(r_asset, r_market) / Var(r_market)",
                "Use Case": "Market risk assessment"
            },
            "Value at Risk": {
                "Description": "Maximum expected loss over given time horizon",
                "Formula": "VaR = -Percentile(returns, α)",
                "Use Case": "Risk limits and capital allocation"
            },
            "Expected Shortfall": {
                "Description": "Expected loss beyond VaR threshold",
                "Formula": "ES = E[Loss | Loss > VaR]",
                "Use Case": "Tail risk measurement"
            }
        }
        
        for measure, details in risk_measures.items():
            with st.expander(f"📊 {measure}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Description:** {details['Description']}")
                    st.write(f"**Use Case:** {details['Use Case']}")
                with col2:
                    st.write(f"**Formula:** {details['Formula']}")
        
        # Risk-Return Visualization
        st.subheader("Risk-Return Analysis")
        
        if st.button("Generate Risk-Return Analysis"):
            # Sample portfolio data
            np.random.seed(42)
            n_portfolios = 1000
            n_assets = 4
            
            # Generate random weights
            weights = np.random.random((n_portfolios, n_assets))
            weights = weights / weights.sum(axis=1)[:, np.newaxis]
            
            # Asset parameters
            expected_returns = np.array([0.08, 0.12, 0.15, 0.06])
            volatilities = np.array([0.12, 0.18, 0.25, 0.08])
            
            # Correlation matrix
            correlation_matrix = np.array([
                [1.0, 0.3, 0.2, -0.1],
                [0.3, 1.0, 0.5, 0.0],
                [0.2, 0.5, 1.0, 0.1],
                [-0.1, 0.0, 0.1, 1.0]
            ])
            
            # Covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            
            # Calculate portfolio metrics
            portfolio_returns = []
            portfolio_risks = []
            
            for w in weights:
                port_return = np.sum(w * expected_returns)
                port_risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                portfolio_returns.append(port_return * 100)
                portfolio_risks.append(port_risk * 100)
            
            # Create scatter plot
            fig = go.Figure()
            
            # Color by Sharpe ratio
            risk_free_rate = 3.0
            sharpe_ratios = [(ret - risk_free_rate) / risk for ret, risk in zip(portfolio_returns, portfolio_risks)]
            
            fig.add_trace(go.Scatter(
                x=portfolio_risks,
                y=portfolio_returns,
                mode='markers',
                marker=dict(
                    color=sharpe_ratios,
                    colorscale='Viridis',
                    size=6,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                text=[f"Return: {ret:.1f}%<br>Risk: {risk:.1f}%<br>Sharpe: {sharpe:.2f}" 
                      for ret, risk, sharpe in zip(portfolio_returns, portfolio_risks, sharpe_ratios)],
                name='Portfolios'
            ))
            
            # Add individual assets
            asset_names = ['Bonds', 'Large Cap', 'Small Cap', 'Cash']
            fig.add_trace(go.Scatter(
                x=volatilities * 100,
                y=expected_returns * 100,
                mode='markers+text',
                text=asset_names,
                textposition="top center",
                marker=dict(size=12, color='red'),
                name='Individual Assets'
            ))
            
            fig.update_layout(
                title="Portfolio Risk-Return Analysis",
                xaxis_title="Risk (Standard Deviation %)",
                yaxis_title="Expected Return (%)",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Value at Risk (VaR)")
        
        st.write("""
        Value at Risk (VaR) is a statistical measure that quantifies the level of financial risk 
        within a firm, portfolio, or position over a specific time frame.
        """)
        
        with st.expander("📊 VaR Methods"):
            st.write("""
            **1. Historical Method:** Uses actual historical returns
            **2. Parametric Method:** Assumes normal distribution
            **3. Monte Carlo Method:** Uses simulated scenarios
            """)
        
        var_method = st.selectbox("Select VaR Method:", 
                                 ["Historical VaR", "Parametric VaR", "Monte Carlo VaR"])
        
        if var_method == "Historical VaR":
            st.write("**Historical Value at Risk**")
            st.write("Uses actual historical returns to estimate potential losses.")
            
            # Generate sample historical returns
            if st.button("Generate Historical Returns Data"):
                np.random.seed(42)
                n_days = 252  # One year of daily data
                
                # Generate correlated returns for a portfolio
                daily_returns = np.random.normal(0.0008, 0.02, n_days)  # ~20% annual volatility
                
                # Add some extreme events
                extreme_events = np.random.choice(n_days, size=10, replace=False)
                daily_returns[extreme_events] = np.random.normal(-0.03, 0.01, 10)
                
                df_returns = pd.DataFrame({
                    'Date': pd.date_range(start='2023-01-01', periods=n_days, freq='B'),
                    'Daily_Return': daily_returns,
                    'Cumulative_Return': np.cumsum(daily_returns)
                })
                
                # Plot returns
                fig = px.line(df_returns, x='Date', y='Cumulative_Return', 
                             title='Cumulative Portfolio Returns')
                st.plotly_chart(fig, use_container_width=True)
                
                # VaR Calculation
                col1, col2 = st.columns(2)
                with col1:
                    confidence_level = st.selectbox("Confidence Level", [90, 95, 99], index=1)
                    portfolio_value = st.number_input("Portfolio Value ($)", value=1000000, min_value=1000)
                with col2:
                    time_horizon = st.selectbox("Time Horizon (days)", [1, 5, 10], index=0)
                
                alpha = (100 - confidence_level) / 100
                
                # Scale returns for time horizon
                scaled_returns = daily_returns * np.sqrt(time_horizon)
                
                # Calculate Historical VaR
                var_historical = np.percentile(scaled_returns, alpha * 100) * portfolio_value
                
                # Calculate Expected Shortfall (Conditional VaR)
                tail_losses = scaled_returns[scaled_returns <= np.percentile(scaled_returns, alpha * 100)]
                expected_shortfall = np.mean(tail_losses) * portfolio_value
                
                # Display results
                st.subheader("VaR Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Historical VaR", f"${abs(var_historical):,.0f}")
                with col2:
                    st.metric("Expected Shortfall", f"${abs(expected_shortfall):,.0f}")
                with col3:
                    st.metric("VaR as % of Portfolio", f"{abs(var_historical)/portfolio_value*100:.2f}%")
                
                # VaR visualization
                fig = go.Figure()
                
                # Histogram of returns
                fig.add_trace(go.Histogram(x=scaled_returns*100, nbinsx=50, 
                                         name='Return Distribution', opacity=0.7))
                
                # VaR line
                fig.add_vline(x=np.percentile(scaled_returns, alpha * 100)*100, 
                             line_dash="dash", line_color="red",
                             annotation_text=f"{confidence_level}% VaR")
                
                # Expected Shortfall region
                tail_threshold = np.percentile(scaled_returns, alpha * 100)
                tail_returns = scaled_returns[scaled_returns <= tail_threshold]
                
                fig.add_trace(go.Histogram(x=tail_returns*100, nbinsx=20, 
                                         name='Tail Losses', opacity=0.8, 
                                         marker_color='red'))
                
                fig.update_layout(
                    title=f"Return Distribution and {confidence_level}% VaR",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif var_method == "Parametric VaR":
            st.write("**Parametric Value at Risk**")
            st.write("Assumes returns follow a normal distribution.")
            
            col1, col2 = st.columns(2)
            with col1:
                portfolio_value = st.number_input("Portfolio Value ($)", value=1000000, min_value=1000, key="param_pv")
                expected_return = st.number_input("Expected Daily Return (%)", value=0.05, min_value=-5.0, max_value=5.0) / 100
            with col2:
                volatility = st.number_input("Daily Volatility (%)", value=1.5, min_value=0.1, max_value=10.0) / 100
                confidence_level = st.selectbox("Confidence Level (%)", [90, 95, 99], index=1, key="param_conf")
            
            if st.button("Calculate Parametric VaR"):
                alpha = (100 - confidence_level) / 100
                z_score = norm.ppf(alpha)
                
                # Parametric VaR
                var_parametric = -(expected_return + z_score * volatility) * portfolio_value
                
                # For comparison, also calculate assuming zero mean
                var_zero_mean = -z_score * volatility * portfolio_value
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Parametric VaR", f"${var_parametric:,.0f}")
                    st.metric("Z-Score", f"{z_score:.3f}")
                with col2:
                    st.metric("VaR (Zero Mean)", f"${var_zero_mean:,.0f}")
                    st.metric("VaR as % of Portfolio", f"{var_parametric/portfolio_value*100:.2f}%")
                
                # Distribution visualization
                x_range = np.linspace(-4*volatility, 4*volatility, 1000)
                pdf = norm.pdf(x_range, expected_return, volatility)
                
                fig = go.Figure()
                
                # Normal distribution
                fig.add_trace(go.Scatter(x=x_range*100, y=pdf, mode='lines', 
                                       name='Return Distribution'))
                
                # VaR threshold
                var_threshold = expected_return + z_score * volatility
                fig.add_vline(x=var_threshold*100, line_dash="dash", line_color="red",
                             annotation_text=f"{confidence_level}% VaR")
                
                # Shade tail area
                tail_x = x_range[x_range <= var_threshold]
                tail_y = norm.pdf(tail_x, expected_return, volatility)
                
                fig.add_trace(go.Scatter(x=tail_x*100, y=tail_y, fill='tonexty', 
                                       fillcolor='rgba(255,0,0,0.3)', mode='none',
                                       name=f'Tail Risk ({alpha*100:.0f}%)'))
                
                fig.update_layout(
                    title="Normal Distribution and Parametric VaR",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Probability Density"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif var_method == "Monte Carlo VaR":
            st.write("**Monte Carlo Value at Risk**")
            st.write("Uses random simulation to generate potential future scenarios.")
            
            col1, col2 = st.columns(2)
            with col1:
                portfolio_value = st.number_input("Portfolio Value ($)", value=1000000, min_value=1000, key="mc_pv")
                expected_return = st.number_input("Expected Daily Return (%)", value=0.05, min_value=-5.0, max_value=5.0, key="mc_ret") / 100
                volatility = st.number_input("Daily Volatility (%)", value=1.5, min_value=0.1, max_value=10.0, key="mc_vol") / 100
            with col2:
                n_simulations = st.number_input("Number of Simulations", value=10000, min_value=1000, max_value=100000)
                confidence_level = st.selectbox("Confidence Level (%)", [90, 95, 99], index=1, key="mc_conf")
                time_horizon = st.number_input("Time Horizon (days)", value=1, min_value=1, max_value=30)
            
            if st.button("Run Monte Carlo Simulation"):
                np.random.seed(42)
                
                # Generate random returns
                dt = 1  # Daily time step
                random_returns = np.random.normal(
                    expected_return * time_horizon,
                    volatility * np.sqrt(time_horizon),
                    n_simulations
                )
                
                # Calculate portfolio values
                portfolio_values = portfolio_value * (1 + random_returns)
                portfolio_changes = portfolio_values - portfolio_value
                
                # Calculate VaR
                alpha = (100 - confidence_level) / 100
                var_monte_carlo = -np.percentile(portfolio_changes, alpha * 100)
                
                # Expected Shortfall
                tail_losses = portfolio_changes[portfolio_changes <= -var_monte_carlo]
                expected_shortfall = -np.mean(tail_losses)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Monte Carlo VaR", f"${var_monte_carlo:,.0f}")
                with col2:
                    st.metric("Expected Shortfall", f"${expected_shortfall:,.0f}")
                with col3:
                    st.metric("Worst Case Loss", f"${-min(portfolio_changes):,.0f}")
                
                # Simulation results visualization
                fig = go.Figure()
                
                # Histogram of portfolio changes
                fig.add_trace(go.Histogram(x=portfolio_changes/1000, nbinsx=50, 
                                         name='Simulated P&L ($000)', opacity=0.7))
                
                # VaR line
                fig.add_vline(x=-var_monte_carlo/1000, line_dash="dash", line_color="red",
                             annotation_text=f"{confidence_level}% VaR")
                
                # Expected Shortfall line
                fig.add_vline(x=-expected_shortfall/1000, line_dash="dot", line_color="orange",
                             annotation_text="Expected Shortfall")
                
                fig.update_layout(
                    title=f"Monte Carlo Simulation Results ({n_simulations:,} simulations)",
                    xaxis_title="Portfolio Change ($000)",
                    yaxis_title="Frequency"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Stress Testing")
        
        st.write("""
        Stress testing evaluates how portfolios perform under extreme but plausible adverse scenarios. 
        It complements VaR by examining tail risk and model limitations.
        """)
        
        st.subheader("Scenario Analysis")
        
        # Define stress scenarios
        scenarios = {
            "Base Case": {"Stock": 0.0, "Bond": 0.0, "Currency": 0.0},
            "Market Crash": {"Stock": -30.0, "Bond": 5.0, "Currency": -10.0},
            "Interest Rate Shock": {"Stock": -10.0, "Bond": -15.0, "Currency": 0.0},
            "Currency Crisis": {"Stock": -15.0, "Bond": -5.0, "Currency": -25.0},
            "Stagflation": {"Stock": -20.0, "Bond": -10.0, "Currency": -15.0},
            "Recovery": {"Stock": 25.0, "Bond": -3.0, "Currency": 5.0}
        }
        
        # Portfolio allocation input
        st.subheader("Portfolio Allocation")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            stock_allocation = st.slider("Stock Allocation (%)", 0, 100, 60)
            portfolio_value = st.number_input("Portfolio Value ($)", value=1000000, min_value=1000, key="stress_pv")
        with col2:
            bond_allocation = st.slider("Bond Allocation (%)", 0, 100, 30)
        with col3:
            currency_allocation = 100 - stock_allocation - bond_allocation
            st.metric("Currency Allocation (%)", currency_allocation)
        
        if stock_allocation + bond_allocation <= 100:
            if st.button("Run Stress Test"):
                # Calculate portfolio impact for each scenario
                stress_results = []
                
                for scenario_name, shocks in scenarios.items():
                    portfolio_impact = (
                        (stock_allocation/100) * (shocks["Stock"]/100) * portfolio_value +
                        (bond_allocation/100) * (shocks["Bond"]/100) * portfolio_value +
                        (currency_allocation/100) * (shocks["Currency"]/100) * portfolio_value
                    )
                    
                    new_portfolio_value = portfolio_value + portfolio_impact
                    
                    stress_results.append({
                        'Scenario': scenario_name,
                        'Portfolio Impact ($)': portfolio_impact,
                        'Portfolio Impact (%)': (portfolio_impact / portfolio_value) * 100,
                        'New Portfolio Value ($)': new_portfolio_value
                    })
                
                # Display results
                df_stress = pd.DataFrame(stress_results)
                st.dataframe(df_stress, use_container_width=True)
                
                # Visualization
                fig = px.bar(df_stress, x='Scenario', y='Portfolio Impact (%)', 
                           title='Stress Test Results',
                           color='Portfolio Impact (%)',
                           color_continuous_scale='RdYlGn_r')
                
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk metrics
                st.subheader("Risk Assessment")
                
                worst_case = min(stress_results, key=lambda x: x['Portfolio Impact ($)'])
                best_case = max(stress_results, key=lambda x: x['Portfolio Impact ($)'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Worst Case Loss", 
                             f"${abs(worst_case['Portfolio Impact ($)']):,.0f}",
                             f"{worst_case['Portfolio Impact (%)']:.1f}%")
                with col2:
                    st.metric("Best Case Gain", 
                             f"${best_case['Portfolio Impact ($)']:,.0f}",
                             f"{best_case['Portfolio Impact (%)']:.1f}%")
                with col3:
                    range_impact = best_case['Portfolio Impact ($)'] - worst_case['Portfolio Impact ($)']
                    st.metric("Impact Range", f"${range_impact:,.0f}")
        else:
            st.error("Portfolio allocations must sum to 100% or less.")
        
        # Historical Stress Testing
        st.subheader("Historical Stress Test")
        
        st.write("Test portfolio performance during historical crisis periods:")
        
        historical_crises = {
            "Black Monday (1987)": {"Date": "Oct 1987", "Stock": -22.6, "Bond": 2.1},
            "Dot-com Crash (2000)": {"Date": "Mar 2000", "Stock": -39.3, "Bond": 12.6},
            "9/11 Attacks (2001)": {"Date": "Sep 2001", "Stock": -11.6, "Bond": 4.5},
            "Financial Crisis (2008)": {"Date": "Oct 2008", "Stock": -36.8, "Bond": 20.1},
            "Flash Crash (2010)": {"Date": "May 2010", "Stock": -6.1, "Bond": 1.2},
            "COVID-19 Crash (2020)": {"Date": "Mar 2020", "Stock": -33.8, "Bond": 8.0}
        }
        
        selected_crisis = st.selectbox("Select Historical Crisis:", list(historical_crises.keys()))
        
        if st.button("Analyze Historical Impact"):
            crisis_data = historical_crises[selected_crisis]
            
            # Calculate impact (simplified - only stocks and bonds)
            total_allocation = stock_allocation + bond_allocation
            if total_allocation > 0:
                stock_weight = stock_allocation / total_allocation
                bond_weight = bond_allocation / total_allocation
                
                historical_impact = (
                    stock_weight * (crisis_data["Stock"]/100) * portfolio_value +
                    bond_weight * (crisis_data["Bond"]/100) * portfolio_value
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Historical Impact", 
                             f"${historical_impact:,.0f}",
                             f"{(historical_impact/portfolio_value)*100:.2f}%")
                with col2:
                    st.metric("Crisis Period", crisis_data["Date"])
                
                if historical_impact < 0:
                    st.warning(f"⚠️ Your portfolio would have lost ${abs(historical_impact):,.0f} during {selected_crisis}")
                else:
                    st.success(f"✅ Your portfolio would have gained ${historical_impact:,.0f} during {selected_crisis}")
    
    with tab4:
        st.subheader("Advanced Risk Metrics")
        
        st.write("""
        Beyond basic volatility and VaR, sophisticated risk metrics provide deeper insights 
        into portfolio behavior and risk characteristics.
        """)
        
        # Generate sample return data
        if st.button("Generate Sample Portfolio Data"):
            np.random.seed(42)
            
            # Create sample daily returns for different asset classes
            n_days = 252
            dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
            
            # Simulate correlated returns
            correlation_matrix = np.array([
                [1.0, 0.7, -0.2, 0.4],
                [0.7, 1.0, -0.1, 0.5],
                [-0.2, -0.1, 1.0, 0.1],
                [0.4, 0.5, 0.1, 1.0]
            ])
            
            means = np.array([0.0008, 0.0006, 0.0002, 0.0004])  # Daily returns
            volatilities = np.array([0.016, 0.020, 0.008, 0.025])  # Daily volatilities
            
            # Generate correlated random returns
            random_normal = np.random.multivariate_normal(means, 
                                                         np.outer(volatilities, volatilities) * correlation_matrix, 
                                                         n_days)
            
            df_returns = pd.DataFrame(random_normal, 
                                    columns=['US_Stocks', 'Intl_Stocks', 'Bonds', 'Commodities'],
                                    index=dates)
            
            # Portfolio weights
            weights = np.array([0.4, 0.3, 0.2, 0.1])  # 40% US stocks, 30% Intl, 20% bonds, 10% commodities
            
            # Calculate portfolio returns
            portfolio_returns = (df_returns * weights).sum(axis=1)
            
            # Calculate various risk metrics
            st.subheader("Risk Metrics Summary")
            
            # Basic metrics
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.03) / annual_volatility  # Assuming 3% risk-free rate
            
            # Downside metrics
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annual_return - 0.03) / downside_deviation if len(downside_returns) > 0 else np.nan
            
            # Maximum Drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # VaR and Expected Shortfall
            var_95 = np.percentile(portfolio_returns, 5)
            expected_shortfall = portfolio_returns[portfolio_returns <= var_95].mean()
            
            # Skewness and Kurtosis
            from scipy.stats import skew, kurtosis
            portfolio_skewness = skew(portfolio_returns)
            portfolio_kurtosis = kurtosis(portfolio_returns)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Annual Return", f"{annual_return*100:.2f}%")
                st.metric("Volatility", f"{annual_volatility*100:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
                st.metric("Sortino Ratio", f"{sortino_ratio:.3f}")
            with col3:
                st.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")
                st.metric("95% VaR (Daily)", f"{var_95*100:.2f}%")
            with col4:
                st.metric("Skewness", f"{portfolio_skewness:.3f}")
                st.metric("Kurtosis", f"{portfolio_kurtosis:.3f}")
            
            # Detailed risk metrics table
            st.subheader("Detailed Risk Analysis")
            
            risk_metrics = {
                'Metric': [
                    'Value at Risk (95%)', 'Expected Shortfall (95%)', 'Downside Deviation',
                    'Maximum Drawdown', 'Calmar Ratio', 'Skewness', 'Excess Kurtosis'
                ],
                'Value': [
                    f"{var_95*100:.2f}%",
                    f"{expected_shortfall*100:.2f}%",
                    f"{downside_deviation*100:.2f}%",
                    f"{max_drawdown*100:.2f}%",
                    f"{(annual_return / abs(max_drawdown)):.3f}" if max_drawdown != 0 else "N/A",
                    f"{portfolio_skewness:.3f}",
                    f"{portfolio_kurtosis:.3f}"
                ],
                'Interpretation': [
                    'Expected daily loss exceeded 5% of the time',
                    'Average loss when VaR is exceeded',
                    'Volatility of negative returns only',
                    'Worst peak-to-trough decline',
                    'Return per unit of max drawdown',
                    'Asymmetry of return distribution',
                    'Tail thickness vs normal distribution'
                ]
            }
            
            df_risk_metrics = pd.DataFrame(risk_metrics)
            st.dataframe(df_risk_metrics, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Rolling volatility
                rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252)
                fig1 = px.line(x=rolling_vol.index, y=rolling_vol.values, 
                              title='30-Day Rolling Volatility')
                fig1.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Drawdown chart
                fig2 = px.line(x=drawdowns.index, y=drawdowns.values * 100, 
                              title='Portfolio Drawdowns')
                fig2.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)")
                st.plotly_chart(fig2, use_container_width=True)
        
        # Risk assessment quiz
        st.subheader("Risk Management Assessment")
        
        q1 = st.radio(
            "Which measure captures tail risk better than VaR?",
            ["Standard deviation", "Expected shortfall", "Beta", "Correlation"],
            key="risk_q1"
        )
        
        q2 = st.radio(
            "Maximum drawdown measures:",
            ["Average loss", "Peak-to-trough decline", "Daily volatility", "Skewness"],
            key="risk_q2"
        )
        
        q3 = st.radio(
            "A portfolio with negative skewness has:",
            ["More upside potential", "More downside risk", "Higher returns", "Lower volatility"],
            key="risk_q3"
        )
        
        if st.button("Submit Assessment"):
            score = 0
            
            if q1 == "Expected shortfall":
                st.success("Q1: Correct! ✅")
                score += 1
            else:
                st.error("Q1: Incorrect. Expected shortfall captures the severity of tail losses beyond VaR.")
            
            if q2 == "Peak-to-trough decline":
                st.success("Q2: Correct! ✅")
                score += 1
            else:
                st.error("Q2: Incorrect. Maximum drawdown measures the worst peak-to-trough decline.")
            
            if q3 == "More downside risk":
                st.success("Q3: Correct! ✅")
                score += 1
            else:
                st.error("Q3: Incorrect. Negative skewness indicates more frequent large losses.")
            
            st.write(f"Your score: {score}/3")
            
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Class 8: Risk Management")
                progress_tracker.set_class_score("Class 8: Risk Management", (score/3) * 100)
                st.success("🎉 Excellent! You understand risk management!")
    
    # Download materials
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Risk Management Resources")
    
    risk_code = """
# Risk Management Tools

import numpy as np
from scipy.stats import norm

def historical_var(returns, confidence_level=0.05):
    '''Calculate historical Value at Risk'''
    return np.percentile(returns, confidence_level * 100)

def parametric_var(expected_return, volatility, confidence_level=0.05):
    '''Calculate parametric VaR assuming normal distribution'''
    z_score = norm.ppf(confidence_level)
    return expected_return + z_score * volatility

def expected_shortfall(returns, confidence_level=0.05):
    '''Calculate Expected Shortfall (Conditional VaR)'''
    var = historical_var(returns, confidence_level)
    return np.mean(returns[returns <= var])

def maximum_drawdown(returns):
    '''Calculate maximum drawdown from return series'''
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return drawdowns.min()

def sharpe_ratio(returns, risk_free_rate=0.0):
    '''Calculate Sharpe ratio'''
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

def sortino_ratio(returns, risk_free_rate=0.0):
    '''Calculate Sortino ratio (downside deviation)'''
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    return excess_returns.mean() / downside_std

# Example usage
sample_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
var_95 = historical_var(sample_returns, 0.05)
es_95 = expected_shortfall(sample_returns, 0.05)
max_dd = maximum_drawdown(sample_returns)
"""
    
    st.sidebar.download_button(
        label="💻 Download Risk Code",
        data=risk_code,
        file_name="risk_management.py",
        mime="text/python"
    )
