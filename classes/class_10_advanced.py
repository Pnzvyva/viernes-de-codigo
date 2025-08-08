import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 10: Temas Avanzados en Ingeniería Financiera")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Derivados Exóticos", "Modelos de Riesgo Crediticio", "Trading Algorítmico", "Síntesis del Curso"])
    
    with tab1:
        st.subheader("Exotic Derivatives and Structured Products")
        
        st.write("""
        Exotic derivatives are non-standard financial instruments with complex payoff structures. 
        They are often used for specific hedging needs or to express sophisticated market views.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Path-Dependent Options:**
            - Asian options (average price)
            - Barrier options (knock-in/out)
            - Lookback options (extrema)
            - Cliquet options (ratchet)
            """)
        
        with col2:
            st.write("""
            **Multi-Asset Derivatives:**
            - Rainbow options
            - Basket options
            - Exchange options
            - Correlation swaps
            """)
        
        exotic_type = st.selectbox("Select Exotic Derivative:", 
                                  ["Rainbow Option", "Cliquet Option", "Variance Swap"])
        
        if exotic_type == "Rainbow Option":
            st.write("**Rainbow Option (Best/Worst of N Assets)**")
            st.write("Option payoff depends on the best or worst performing asset among multiple underlying assets.")
            
            n_assets_rainbow = st.selectbox("Number of Assets:", [2, 3, 4], index=1)
            
            # Asset parameters
            col1, col2 = st.columns(2)
            with col1:
                K_rainbow = st.number_input("Strike Price", value=100.0, key="rainbow_K")
                T_rainbow = st.number_input("Time to Expiration (years)", value=1.0, key="rainbow_T")
                r_rainbow = st.number_input("Risk-free Rate (%)", value=5.0, key="rainbow_r") / 100
                rainbow_type = st.radio("Rainbow Type:", ["Best of", "Worst of"])
            with col2:
                payoff_type = st.radio("Payoff Type:", ["Call", "Put"])
                n_sims_rainbow = st.number_input("Monte Carlo Simulations", value=50000, key="rainbow_sims")
            
            # Individual asset parameters
            st.subheader("Asset Parameters")
            asset_params_rainbow = []
            for i in range(n_assets_rainbow):
                col1, col2, col3 = st.columns(3)
                with col1:
                    S0 = st.number_input(f"Asset {i+1} Price", value=100.0, key=f"rainbow_S0_{i}")
                with col2:
                    vol = st.number_input(f"Volatility {i+1} (%)", value=20.0 + i*5, key=f"rainbow_vol_{i}") / 100
                with col3:
                    div = st.number_input(f"Dividend Yield {i+1} (%)", value=2.0, key=f"rainbow_div_{i}") / 100
                asset_params_rainbow.append([S0, vol, div])
            
            # Correlation matrix
            st.subheader("Correlation Structure")
            avg_correlation = st.slider("Average Correlation", -0.5, 0.9, 0.3, 0.1)
            
            if st.button("Price Rainbow Option"):
                # Create correlation matrix
                corr_rainbow = np.full((n_assets_rainbow, n_assets_rainbow), avg_correlation)
                np.fill_diagonal(corr_rainbow, 1.0)
                
                # Extract parameters
                S0_rainbow = np.array([params[0] for params in asset_params_rainbow])
                vols_rainbow = np.array([params[1] for params in asset_params_rainbow])
                divs_rainbow = np.array([params[2] for params in asset_params_rainbow])
                
                # Monte Carlo simulation
                np.random.seed(42)
                dt = T_rainbow
                
                # Generate correlated random numbers
                L = np.linalg.cholesky(corr_rainbow)
                Z = np.random.standard_normal((n_sims_rainbow, n_assets_rainbow))
                Z_corr = Z @ L.T
                
                # Generate final asset prices
                ST_rainbow = np.zeros((n_sims_rainbow, n_assets_rainbow))
                for i in range(n_assets_rainbow):
                    ST_rainbow[:, i] = S0_rainbow[i] * np.exp(
                        (r_rainbow - divs_rainbow[i] - 0.5 * vols_rainbow[i]**2) * T_rainbow +
                        vols_rainbow[i] * np.sqrt(T_rainbow) * Z_corr[:, i]
                    )
                
                # Calculate rainbow payoffs
                if rainbow_type == "Best of":
                    selected_prices = np.max(ST_rainbow, axis=1)
                else:
                    selected_prices = np.min(ST_rainbow, axis=1)
                
                if payoff_type == "Call":
                    payoffs = np.maximum(selected_prices - K_rainbow, 0)
                else:
                    payoffs = np.maximum(K_rainbow - selected_prices, 0)
                
                # Price option
                rainbow_price = np.exp(-r_rainbow * T_rainbow) * np.mean(payoffs)
                standard_error = np.exp(-r_rainbow * T_rainbow) * np.std(payoffs) / np.sqrt(n_sims_rainbow)
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rainbow Option Price", f"${rainbow_price:.4f}")
                    st.metric("Standard Error", f"±${1.96 * standard_error:.4f}")
                with col2:
                    exercise_prob = np.mean(payoffs > 0) * 100
                    st.metric("Exercise Probability", f"{exercise_prob:.1f}%")
                    avg_payoff = np.mean(payoffs[payoffs > 0])
                    st.metric("Avg Payoff (if exercised)", f"${avg_payoff:.2f}")
                with col3:
                    avg_selected = np.mean(selected_prices)
                    st.metric(f"Average {rainbow_type} Price", f"${avg_selected:.2f}")
                
                # Distribution analysis
                fig_rainbow = px.histogram(selected_prices, nbins=50, 
                                          title=f"Distribution of {rainbow_type} Asset Prices")
                fig_rainbow.add_vline(x=K_rainbow, line_dash="dash", annotation_text="Strike Price")
                st.plotly_chart(fig_rainbow, use_container_width=True)
        
        elif exotic_type == "Cliquet Option":
            st.write("**Cliquet (Ratchet) Option**")
            st.write("Option that locks in gains at regular reset dates, providing downside protection with upside participation.")
            
            col1, col2 = st.columns(2)
            with col1:
                S0_cliquet = st.number_input("Initial Stock Price", value=100.0, key="cliquet_S0")
                T_cliquet = st.number_input("Total Time (years)", value=3.0, key="cliquet_T")
                n_resets = st.number_input("Number of Resets", value=12, min_value=2, key="cliquet_resets")
                global_floor = st.number_input("Global Floor (%)", value=-10.0, key="cliquet_floor") / 100
            with col2:
                local_floor = st.number_input("Local Floor (%)", value=-5.0, key="cliquet_local_floor") / 100
                local_cap = st.number_input("Local Cap (%)", value=15.0, key="cliquet_local_cap") / 100
                volatility_cliquet = st.number_input("Volatility (%)", value=25.0, key="cliquet_vol") / 100
                r_cliquet = st.number_input("Risk-free Rate (%)", value=4.0, key="cliquet_r") / 100
            
            if st.button("Price Cliquet Option"):
                dt = T_cliquet / n_resets
                n_sims_cliquet = 50000
                
                np.random.seed(42)
                
                # Generate stock price paths
                Z = np.random.standard_normal((n_sims_cliquet, n_resets))
                S = np.zeros((n_sims_cliquet, n_resets + 1))
                S[:, 0] = S0_cliquet
                
                # Track cliquet performance
                cliquet_returns = np.zeros((n_sims_cliquet, n_resets))
                
                for i in range(n_resets):
                    # Generate stock prices for this period
                    S[:, i + 1] = S[:, i] * np.exp(
                        (r_cliquet - 0.5 * volatility_cliquet**2) * dt +
                        volatility_cliquet * np.sqrt(dt) * Z[:, i]
                    )
                    
                    # Calculate period return
                    period_return = (S[:, i + 1] - S[:, i]) / S[:, i]
                    
                    # Apply local floor and cap
                    capped_return = np.minimum(period_return, local_cap)
                    floored_return = np.maximum(capped_return, local_floor)
                    
                    cliquet_returns[:, i] = floored_return
                
                # Calculate total cliquet performance
                total_cliquet_return = np.sum(cliquet_returns, axis=1)
                
                # Apply global floor
                final_payoff = np.maximum(total_cliquet_return, global_floor)
                
                # Price the cliquet
                cliquet_price = np.exp(-r_cliquet * T_cliquet) * np.mean(final_payoff) * S0_cliquet
                
                # Compare with vanilla call
                vanilla_price = FinancialCalculations.black_scholes_call(
                    S0_cliquet, S0_cliquet, T_cliquet, r_cliquet, volatility_cliquet
                )
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cliquet Option Price", f"${cliquet_price:.2f}")
                    avg_return = np.mean(total_cliquet_return) * 100
                    st.metric("Average Total Return", f"{avg_return:.2f}%")
                with col2:
                    st.metric("Vanilla Call Price", f"${vanilla_price:.2f}")
                    downside_protection = np.mean(final_payoff >= global_floor) * 100
                    st.metric("Downside Protection", f"{downside_protection:.1f}%")
                with col3:
                    max_return = np.max(total_cliquet_return) * 100
                    st.metric("Maximum Return", f"{max_return:.2f}%")
                    min_return = np.min(final_payoff) * 100
                    st.metric("Minimum Return", f"{min_return:.2f}%")
                
                # Distribution of returns
                fig_cliquet = px.histogram(total_cliquet_return * 100, nbins=50, 
                                          title="Cliquet Total Return Distribution")
                fig_cliquet.add_vline(x=global_floor * 100, line_dash="dash", 
                                     annotation_text="Global Floor")
                fig_cliquet.update_layout(xaxis_title="Total Return (%)", yaxis_title="Frequency")
                st.plotly_chart(fig_cliquet, use_container_width=True)
        
        elif exotic_type == "Variance Swap":
            st.write("**Variance Swap**")
            st.write("Swap that allows trading realized volatility against implied volatility.")
            
            col1, col2 = st.columns(2)
            with col1:
                S0_var = st.number_input("Initial Stock Price", value=100.0, key="var_S0")
                T_var = st.number_input("Time to Maturity (years)", value=0.25, key="var_T")
                n_obs = st.number_input("Number of Observations", value=63, key="var_obs")  # ~3 months daily
                strike_vol = st.number_input("Strike Volatility (%)", value=20.0, key="var_strike") / 100
            with col2:
                true_vol = st.number_input("True Volatility (%)", value=25.0, key="var_true") / 100
                notional = st.number_input("Notional Amount ($)", value=1000000.0, key="var_notional")
                r_var = st.number_input("Risk-free Rate (%)", value=5.0, key="var_r") / 100
            
            if st.button("Analyze Variance Swap"):
                dt = T_var / n_obs
                n_sims_var = 10000
                
                np.random.seed(42)
                
                # Generate stock price paths
                Z = np.random.standard_normal((n_sims_var, n_obs))
                S_var = np.zeros((n_sims_var, n_obs + 1))
                S_var[:, 0] = S0_var
                
                # Generate paths
                for i in range(n_obs):
                    S_var[:, i + 1] = S_var[:, i] * np.exp(
                        (r_var - 0.5 * true_vol**2) * dt + true_vol * np.sqrt(dt) * Z[:, i]
                    )
                
                # Calculate realized variance for each path
                log_returns = np.log(S_var[:, 1:] / S_var[:, :-1])
                realized_variance = np.sum(log_returns**2, axis=1) * (252 / n_obs)  # Annualized
                
                # Variance swap payoff
                variance_diff = realized_variance - strike_vol**2
                swap_payoff = notional * variance_diff
                
                # Present value
                pv_payoff = np.exp(-r_var * T_var) * swap_payoff
                fair_value = np.mean(pv_payoff)
                
                # Theoretical fair strike (for at-the-money variance swap)
                theoretical_var = true_vol**2
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Variance Swap Fair Value", f"${fair_value:,.0f}")
                    avg_realized_vol = np.sqrt(np.mean(realized_variance))
                    st.metric("Average Realized Vol", f"{avg_realized_vol*100:.2f}%")
                with col2:
                    st.metric("Strike Volatility", f"{strike_vol*100:.2f}%")
                    st.metric("True Volatility", f"{true_vol*100:.2f}%")
                with col3:
                    win_probability = np.mean(variance_diff > 0) * 100
                    st.metric("Win Probability", f"{win_probability:.1f}%")
                    max_payoff = np.max(swap_payoff)
                    st.metric("Max Payoff", f"${max_payoff:,.0f}")
                
                # Distribution of realized volatility
                realized_vol = np.sqrt(realized_variance)
                fig_var = px.histogram(realized_vol * 100, nbins=50, 
                                      title="Distribution of Realized Volatility")
                fig_var.add_vline(x=strike_vol * 100, line_dash="dash", 
                                 annotation_text="Strike Volatility")
                fig_var.add_vline(x=true_vol * 100, line_dash="dot", 
                                 annotation_text="True Volatility")
                fig_var.update_layout(xaxis_title="Realized Volatility (%)", yaxis_title="Frequency")
                st.plotly_chart(fig_var, use_container_width=True)
                
                # P&L distribution
                fig_pnl = px.histogram(pv_payoff/1000, nbins=50, title="Variance Swap P&L Distribution")
                fig_pnl.add_vline(x=0, line_dash="dash", annotation_text="Break-even")
                fig_pnl.update_layout(xaxis_title="P&L ($000)", yaxis_title="Frequency")
                st.plotly_chart(fig_pnl, use_container_width=True)
    
    with tab2:
        st.subheader("Credit Risk Modeling")
        
        st.write("""
        Credit risk is the risk of financial loss due to counterparty default. Modern credit risk 
        models use probability distributions to estimate default likelihood and loss severity.
        """)
        
        credit_model = st.selectbox("Select Credit Model:", 
                                   ["Merton Model", "Reduced Form Model", "Portfolio Credit Risk"])
        
        if credit_model == "Merton Model":
            st.write("**Merton Structural Credit Model**")
            st.write("Models default as occurring when firm value falls below debt level.")
            
            with st.expander("📊 Merton Model Theory"):
                st.write("""
                The Merton model treats equity as a call option on firm value with strike equal to debt.
                Default occurs when firm value < debt at maturity.
                """)
                st.latex(r"P_{default} = N\left(\frac{\ln(D/V) + (r - 0.5\sigma_V^2)T}{\sigma_V\sqrt{T}}\right)")
                st.write("Where D = debt, V = firm value, σ_V = firm volatility")
            
            col1, col2 = st.columns(2)
            with col1:
                firm_value = st.number_input("Firm Value ($B)", value=10.0, key="merton_V")
                debt_value = st.number_input("Debt Value ($B)", value=6.0, key="merton_D")
                firm_volatility = st.number_input("Firm Volatility (%)", value=25.0, key="merton_vol") / 100
            with col2:
                time_horizon = st.number_input("Time Horizon (years)", value=1.0, key="merton_T")
                risk_free_rate = st.number_input("Risk-free Rate (%)", value=3.0, key="merton_r") / 100
            
            if st.button("Calculate Default Probability"):
                # Merton model calculation
                d1 = (np.log(firm_value / debt_value) + 
                     (risk_free_rate + 0.5 * firm_volatility**2) * time_horizon) / (firm_volatility * np.sqrt(time_horizon))
                d2 = d1 - firm_volatility * np.sqrt(time_horizon)
                
                # Default probability
                default_prob = norm.cdf(-d2)
                
                # Distance to default
                distance_to_default = (np.log(firm_value / debt_value) + 
                                     (risk_free_rate - 0.5 * firm_volatility**2) * time_horizon) / (firm_volatility * np.sqrt(time_horizon))
                
                # Equity value (call option)
                equity_value = firm_value * norm.cdf(d1) - debt_value * np.exp(-risk_free_rate * time_horizon) * norm.cdf(d2)
                
                # Credit spread (approximate)
                risk_neutral_prob = default_prob
                recovery_rate = 0.4  # Assume 40% recovery
                credit_spread = -np.log(1 - risk_neutral_prob * (1 - recovery_rate)) / time_horizon
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Default Probability", f"{default_prob*100:.2f}%")
                    st.metric("Distance to Default", f"{distance_to_default:.3f}")
                with col2:
                    st.metric("Equity Value ($B)", f"${equity_value:.2f}")
                    leverage = debt_value / firm_value
                    st.metric("Leverage Ratio", f"{leverage:.2f}")
                with col3:
                    st.metric("Credit Spread", f"{credit_spread*100:.0f} bps")
                    equity_vol = equity_value / firm_value * firm_volatility / (equity_value / firm_value)
                    st.metric("Implied Equity Vol", f"{equity_vol*100:.1f}%")
                
                # Sensitivity analysis
                firm_value_range = np.linspace(debt_value * 0.8, debt_value * 2.0, 50)
                default_probs = []
                
                for V in firm_value_range:
                    d2_sens = (np.log(V / debt_value) + 
                              (risk_free_rate - 0.5 * firm_volatility**2) * time_horizon) / (firm_volatility * np.sqrt(time_horizon))
                    prob = norm.cdf(-d2_sens)
                    default_probs.append(prob)
                
                fig_merton = go.Figure()
                fig_merton.add_trace(go.Scatter(x=firm_value_range, y=np.array(default_probs)*100, 
                                               mode='lines', name='Default Probability'))
                fig_merton.add_vline(x=debt_value, line_dash="dash", annotation_text="Debt Level")
                fig_merton.add_vline(x=firm_value, line_dash="dot", annotation_text="Current Firm Value")
                
                fig_merton.update_layout(
                    title="Default Probability vs Firm Value",
                    xaxis_title="Firm Value ($B)",
                    yaxis_title="Default Probability (%)"
                )
                st.plotly_chart(fig_merton, use_container_width=True)
        
        elif credit_model == "Reduced Form Model":
            st.write("**Reduced Form (Intensity-Based) Model**")
            st.write("Models default as a random event with stochastic intensity (hazard rate).")
            
            col1, col2 = st.columns(2)
            with col1:
                lambda_0 = st.number_input("Initial Hazard Rate (%)", value=2.0, key="rf_lambda") / 100
                lambda_reversion = st.number_input("Mean Reversion Speed", value=0.5, key="rf_kappa")
                lambda_vol = st.number_input("Hazard Rate Volatility", value=0.3, key="rf_vol")
            with col2:
                time_horizon_rf = st.number_input("Time Horizon (years)", value=5.0, key="rf_T")
                n_paths_rf = st.number_input("Number of Paths", value=10000, key="rf_paths")
                recovery_rate_rf = st.number_input("Recovery Rate (%)", value=40.0, key="rf_recovery") / 100
            
            if st.button("Simulate Credit Events"):
                dt = 1/252  # Daily simulation
                n_steps_rf = int(time_horizon_rf * 252)
                
                np.random.seed(42)
                
                # Simulate hazard rate paths (CIR process)
                lambda_paths = np.zeros((n_paths_rf, n_steps_rf + 1))
                lambda_paths[:, 0] = lambda_0
                
                default_times = np.full(n_paths_rf, np.inf)
                default_indicators = np.zeros(n_paths_rf)
                
                for path in range(n_paths_rf):
                    for step in range(n_steps_rf):
                        # CIR process for hazard rate
                        dW = np.random.normal(0, np.sqrt(dt))
                        lambda_t = max(lambda_paths[path, step], 0.0001)  # Avoid negative rates
                        
                        dlambda = lambda_reversion * (lambda_0 - lambda_t) * dt + lambda_vol * np.sqrt(lambda_t) * dW
                        lambda_paths[path, step + 1] = lambda_t + dlambda
                        
                        # Check for default
                        if default_indicators[path] == 0:  # Not yet defaulted
                            default_prob_step = lambda_t * dt
                            if np.random.uniform() < default_prob_step:
                                default_times[path] = step * dt
                                default_indicators[path] = 1
                
                # Calculate survival probabilities
                time_points = np.linspace(0, time_horizon_rf, 21)
                survival_probs = []
                
                for t in time_points:
                    survived = np.sum(default_times > t) / n_paths_rf
                    survival_probs.append(survived)
                
                # Results
                final_default_rate = np.mean(default_indicators) * 100
                avg_default_time = np.mean(default_times[default_times < np.inf])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Default Rate", f"{final_default_rate:.2f}%")
                    st.metric("Average Default Time", f"{avg_default_time:.2f} years")
                with col2:
                    survival_1y = np.sum(default_times > 1.0) / n_paths_rf * 100
                    st.metric("1-Year Survival", f"{survival_1y:.1f}%")
                    survival_5y = np.sum(default_times > 5.0) / n_paths_rf * 100
                    st.metric("5-Year Survival", f"{survival_5y:.1f}%")
                with col3:
                    avg_lambda = np.mean(lambda_paths[:, -1]) * 100
                    st.metric("Final Avg Hazard Rate", f"{avg_lambda:.2f}%")
                
                # Survival curve
                fig_survival = go.Figure()
                fig_survival.add_trace(go.Scatter(x=time_points, y=np.array(survival_probs)*100, 
                                                 mode='lines+markers', name='Survival Probability'))
                fig_survival.update_layout(
                    title="Survival Probability Curve",
                    xaxis_title="Time (years)",
                    yaxis_title="Survival Probability (%)"
                )
                st.plotly_chart(fig_survival, use_container_width=True)
                
                # Hazard rate evolution
                sample_paths = min(50, n_paths_rf)
                t_grid = np.linspace(0, time_horizon_rf, n_steps_rf + 1)
                
                fig_hazard = go.Figure()
                for i in range(sample_paths):
                    fig_hazard.add_trace(go.Scatter(x=t_grid, y=lambda_paths[i, :]*100, 
                                                   mode='lines', opacity=0.3, showlegend=False))
                
                # Add mean path
                mean_lambda = np.mean(lambda_paths, axis=0)
                fig_hazard.add_trace(go.Scatter(x=t_grid, y=mean_lambda*100, 
                                               mode='lines', line=dict(color='red', width=3),
                                               name='Mean Hazard Rate'))
                
                fig_hazard.update_layout(
                    title="Hazard Rate Evolution",
                    xaxis_title="Time (years)",
                    yaxis_title="Hazard Rate (%)"
                )
                st.plotly_chart(fig_hazard, use_container_width=True)
        
        elif credit_model == "Portfolio Credit Risk":
            st.write("**Portfolio Credit Risk (One-Factor Model)**")
            st.write("Model default correlation using systematic risk factor.")
            
            col1, col2 = st.columns(2)
            with col1:
                n_obligors = st.number_input("Number of Obligors", value=100, min_value=10, max_value=1000)
                avg_pd = st.number_input("Average PD (%)", value=2.0, key="port_pd") / 100
                correlation = st.number_input("Asset Correlation", value=0.15, min_value=0.0, max_value=0.99)
            with col2:
                lgd = st.number_input("Loss Given Default (%)", value=60.0, key="port_lgd") / 100
                exposure = st.number_input("Exposure per Obligor ($)", value=1000000.0, key="port_exp")
                n_scenarios = st.number_input("Number of Scenarios", value=10000, key="port_scenarios")
            
            if st.button("Simulate Portfolio Losses"):
                np.random.seed(42)
                
                # One-factor model simulation
                portfolio_losses = []
                
                for scenario in range(n_scenarios):
                    # Generate systematic factor
                    Z = np.random.normal(0, 1)
                    
                    # Generate idiosyncratic factors
                    epsilon = np.random.normal(0, 1, n_obligors)
                    
                    # Asset values
                    sqrt_rho = np.sqrt(correlation)
                    sqrt_1_rho = np.sqrt(1 - correlation)
                    asset_values = sqrt_rho * Z + sqrt_1_rho * epsilon
                    
                    # Default threshold (inverse normal of PD)
                    threshold = norm.ppf(avg_pd)
                    
                    # Determine defaults
                    defaults = asset_values < threshold
                    n_defaults = np.sum(defaults)
                    
                    # Calculate portfolio loss
                    portfolio_loss = n_defaults * exposure * lgd
                    portfolio_losses.append(portfolio_loss)
                
                portfolio_losses = np.array(portfolio_losses)
                
                # Risk metrics
                expected_loss = np.mean(portfolio_losses)
                portfolio_std = np.std(portfolio_losses)
                var_95 = np.percentile(portfolio_losses, 95)
                var_99 = np.percentile(portfolio_losses, 99)
                expected_shortfall = np.mean(portfolio_losses[portfolio_losses >= var_99])
                
                # Economic capital
                economic_capital = var_99 - expected_loss
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Loss", f"${expected_loss/1e6:.2f}M")
                    st.metric("Portfolio Std Dev", f"${portfolio_std/1e6:.2f}M")
                with col2:
                    st.metric("95% VaR", f"${var_95/1e6:.2f}M")
                    st.metric("99% VaR", f"${var_99/1e6:.2f}M")
                with col3:
                    st.metric("Expected Shortfall", f"${expected_shortfall/1e6:.2f}M")
                    st.metric("Economic Capital", f"${economic_capital/1e6:.2f}M")
                
                # Loss distribution
                fig_port = px.histogram(portfolio_losses/1e6, nbins=50, 
                                       title="Portfolio Loss Distribution")
                fig_port.add_vline(x=expected_loss/1e6, line_dash="dash", annotation_text="Expected Loss")
                fig_port.add_vline(x=var_99/1e6, line_dash="dot", line_color="red", annotation_text="99% VaR")
                fig_port.update_layout(xaxis_title="Portfolio Loss ($M)", yaxis_title="Frequency")
                st.plotly_chart(fig_port, use_container_width=True)
                
                # Default rate distribution
                default_rates = [loss / (n_obligors * exposure * lgd) * 100 for loss in portfolio_losses]
                fig_default = px.histogram(default_rates, nbins=30, title="Default Rate Distribution")
                fig_default.add_vline(x=avg_pd*100, line_dash="dash", annotation_text="Expected Default Rate")
                fig_default.update_layout(xaxis_title="Default Rate (%)", yaxis_title="Frequency")
                st.plotly_chart(fig_default, use_container_width=True)
    
    with tab3:
        st.subheader("Algorithmic Trading Concepts")
        
        st.write("""
        Algorithmic trading uses computer algorithms to execute trading strategies based on 
        predefined rules, market data analysis, and mathematical models.
        """)
        
        algo_topic = st.selectbox("Select Topic:", 
                                 ["Mean Reversion Strategy", "Momentum Strategy", "Pairs Trading", "Portfolio Optimization"])
        
        if algo_topic == "Mean Reversion Strategy":
            st.write("**Mean Reversion Trading Strategy**")
            st.write("Strategy based on the assumption that prices will revert to their historical mean.")
            
            col1, col2 = st.columns(2)
            with col1:
                lookback_period = st.number_input("Lookback Period (days)", value=20, min_value=5, max_value=100)
                entry_threshold = st.number_input("Entry Threshold (std devs)", value=2.0, min_value=0.5, max_value=5.0)
                exit_threshold = st.number_input("Exit Threshold (std devs)", value=0.5, min_value=0.1, max_value=2.0)
            with col2:
                initial_capital = st.number_input("Initial Capital ($)", value=100000.0, min_value=1000.0)
                position_size = st.number_input("Position Size (%)", value=10.0, min_value=1.0, max_value=50.0) / 100
                transaction_cost = st.number_input("Transaction Cost (%)", value=0.1, min_value=0.0) / 100
            
            if st.button("Backtest Mean Reversion"):
                # Generate synthetic price data
                np.random.seed(42)
                n_days = 252 * 2  # 2 years of data
                
                # Generate mean-reverting price series
                prices = [100.0]
                theta = 0.1  # Mean reversion speed
                mu = 100.0   # Long-term mean
                sigma = 0.02  # Daily volatility
                
                for i in range(n_days - 1):
                    price_change = theta * (mu - prices[-1]) + sigma * np.random.normal()
                    new_price = prices[-1] + price_change
                    prices.append(max(new_price, 1.0))  # Avoid negative prices
                
                prices = np.array(prices)
                dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
                
                # Calculate rolling statistics
                df = pd.DataFrame({'Date': dates, 'Price': prices})
                df['SMA'] = df['Price'].rolling(window=lookback_period).mean()
                df['StdDev'] = df['Price'].rolling(window=lookback_period).std()
                df['Z_Score'] = (df['Price'] - df['SMA']) / df['StdDev']
                
                # Generate signals
                df['Signal'] = 0
                df.loc[df['Z_Score'] > entry_threshold, 'Signal'] = -1  # Sell when too high
                df.loc[df['Z_Score'] < -entry_threshold, 'Signal'] = 1   # Buy when too low
                df.loc[abs(df['Z_Score']) < exit_threshold, 'Signal'] = 0  # Exit when near mean
                
                # Backtest strategy
                position = 0
                cash = initial_capital
                equity_curve = []
                trades = []
                
                for i in range(len(df)):
                    current_price = df.iloc[i]['Price']
                    signal = df.iloc[i]['Signal']
                    
                    # Position management
                    if signal != 0 and position == 0:  # Enter position
                        position_value = cash * position_size
                        shares = position_value / current_price
                        transaction_costs = position_value * transaction_cost
                        
                        if signal == 1:  # Buy
                            position = shares
                            cash -= position_value + transaction_costs
                        else:  # Sell short
                            position = -shares
                            cash += position_value - transaction_costs
                        
                        trades.append({
                            'Date': df.iloc[i]['Date'],
                            'Type': 'Buy' if signal == 1 else 'Sell',
                            'Price': current_price,
                            'Shares': abs(shares),
                            'Value': position_value
                        })
                    
                    elif signal == 0 and position != 0:  # Exit position
                        position_value = abs(position) * current_price
                        transaction_costs = position_value * transaction_cost
                        
                        if position > 0:  # Close long
                            cash += position_value - transaction_costs
                        else:  # Close short
                            cash -= position_value + transaction_costs
                        
                        trades.append({
                            'Date': df.iloc[i]['Date'],
                            'Type': 'Close',
                            'Price': current_price,
                            'Shares': abs(position),
                            'Value': position_value
                        })
                        
                        position = 0
                    
                    # Calculate total equity
                    if position != 0:
                        position_value = position * current_price
                        total_equity = cash + position_value
                    else:
                        total_equity = cash
                    
                    equity_curve.append(total_equity)
                
                # Performance metrics
                df['Equity'] = equity_curve
                df['Returns'] = df['Equity'].pct_change()
                df['BuyHold'] = initial_capital * (df['Price'] / df.iloc[0]['Price'])
                
                total_return = (equity_curve[-1] / initial_capital - 1) * 100
                buy_hold_return = (df.iloc[-1]['BuyHold'] / initial_capital - 1) * 100
                
                annual_return = (equity_curve[-1] / initial_capital) ** (252 / len(df)) - 1
                annual_vol = df['Returns'].std() * np.sqrt(252)
                sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                
                max_dd = ((df['Equity'].expanding().max() - df['Equity']) / df['Equity'].expanding().max()).max()
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{total_return:.2f}%")
                    st.metric("Annual Return", f"{annual_return*100:.2f}%")
                with col2:
                    st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
                    st.metric("Annual Volatility", f"{annual_vol*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
                    st.metric("Max Drawdown", f"{max_dd*100:.2f}%")
                
                # Plot results
                fig_backtest = go.Figure()
                
                fig_backtest.add_trace(go.Scatter(x=df['Date'], y=df['Price'], 
                                                 mode='lines', name='Stock Price'))
                fig_backtest.add_trace(go.Scatter(x=df['Date'], y=df['SMA'], 
                                                 mode='lines', name='Moving Average'))
                
                # Add entry/exit points
                buy_signals = df[df['Signal'] == 1]
                sell_signals = df[df['Signal'] == -1]
                
                fig_backtest.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Price'],
                                                 mode='markers', marker=dict(color='green', size=8),
                                                 name='Buy Signal'))
                fig_backtest.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Price'],
                                                 mode='markers', marker=dict(color='red', size=8),
                                                 name='Sell Signal'))
                
                fig_backtest.update_layout(title="Mean Reversion Strategy - Price and Signals",
                                          xaxis_title="Date", yaxis_title="Price ($)")
                st.plotly_chart(fig_backtest, use_container_width=True)
                
                # Equity curve
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(x=df['Date'], y=df['Equity'], 
                                               mode='lines', name='Strategy'))
                fig_equity.add_trace(go.Scatter(x=df['Date'], y=df['BuyHold'], 
                                               mode='lines', name='Buy & Hold'))
                fig_equity.update_layout(title="Equity Curves Comparison",
                                        xaxis_title="Date", yaxis_title="Portfolio Value ($)")
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Trade summary
                if trades:
                    st.subheader("Recent Trades")
                    trades_df = pd.DataFrame(trades[-10:])  # Show last 10 trades
                    st.dataframe(trades_df, use_container_width=True)
        
        elif algo_topic == "Pairs Trading":
            st.write("**Pairs Trading Strategy**")
            st.write("Market-neutral strategy that trades the spread between two correlated assets.")
            
            # Generate synthetic correlated asset data
            if st.button("Generate Pairs Trading Example"):
                np.random.seed(42)
                n_days = 252
                dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
                
                # Generate two correlated assets
                correlation = 0.8
                returns_A = np.random.normal(0.0005, 0.02, n_days)
                returns_B = correlation * returns_A + np.sqrt(1 - correlation**2) * np.random.normal(0.0005, 0.02, n_days)
                
                # Add temporary divergence
                divergence_period = slice(100, 150)
                returns_B[divergence_period] += np.linspace(0, -0.05, 50)
                returns_B[slice(150, 200)] += np.linspace(-0.05, 0, 50)
                
                price_A = 100 * np.cumprod(1 + returns_A)
                price_B = 100 * np.cumprod(1 + returns_B)
                
                df_pairs = pd.DataFrame({
                    'Date': dates,
                    'Asset_A': price_A,
                    'Asset_B': price_B
                })
                
                # Calculate spread and z-score
                df_pairs['Spread'] = df_pairs['Asset_A'] - df_pairs['Asset_B']
                df_pairs['Spread_MA'] = df_pairs['Spread'].rolling(window=20).mean()
                df_pairs['Spread_Std'] = df_pairs['Spread'].rolling(window=20).std()
                df_pairs['Z_Score'] = (df_pairs['Spread'] - df_pairs['Spread_MA']) / df_pairs['Spread_Std']
                
                # Generate trading signals
                entry_threshold = 2.0
                exit_threshold = 0.5
                
                df_pairs['Signal'] = 0
                df_pairs.loc[df_pairs['Z_Score'] > entry_threshold, 'Signal'] = -1  # Short spread
                df_pairs.loc[df_pairs['Z_Score'] < -entry_threshold, 'Signal'] = 1   # Long spread
                df_pairs.loc[abs(df_pairs['Z_Score']) < exit_threshold, 'Signal'] = 0
                
                # Visualizations
                fig_pairs = go.Figure()
                fig_pairs.add_trace(go.Scatter(x=df_pairs['Date'], y=df_pairs['Asset_A'], 
                                              mode='lines', name='Asset A'))
                fig_pairs.add_trace(go.Scatter(x=df_pairs['Date'], y=df_pairs['Asset_B'], 
                                              mode='lines', name='Asset B'))
                fig_pairs.update_layout(title="Pairs Trading - Asset Prices",
                                       xaxis_title="Date", yaxis_title="Price ($)")
                st.plotly_chart(fig_pairs, use_container_width=True)
                
                # Spread analysis
                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(x=df_pairs['Date'], y=df_pairs['Z_Score'], 
                                               mode='lines', name='Z-Score'))
                fig_spread.add_hline(y=entry_threshold, line_dash="dash", line_color="red", 
                                    annotation_text="Entry Threshold")
                fig_spread.add_hline(y=-entry_threshold, line_dash="dash", line_color="red")
                fig_spread.add_hline(y=0, line_dash="dot", line_color="gray")
                
                fig_spread.update_layout(title="Spread Z-Score",
                                        xaxis_title="Date", yaxis_title="Z-Score")
                st.plotly_chart(fig_spread, use_container_width=True)
                
                # Performance metrics
                correlation_coef = np.corrcoef(returns_A, returns_B)[0, 1]
                mean_reversion_speed = abs(df_pairs['Z_Score'].diff().mean())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Correlation Coefficient", f"{correlation_coef:.3f}")
                    st.metric("Mean Spread", f"{df_pairs['Spread'].mean():.2f}")
                with col2:
                    st.metric("Spread Volatility", f"{df_pairs['Spread'].std():.2f}")
                    st.metric("Max Z-Score", f"{df_pairs['Z_Score'].abs().max():.2f}")
                with col3:
                    signal_frequency = (df_pairs['Signal'] != 0).sum() / len(df_pairs) * 100
                    st.metric("Signal Frequency", f"{signal_frequency:.1f}%")
    
    with tab4:
        st.subheader("Course Synthesis and Advanced Applications")
        
        st.write("""
        This final section integrates concepts from all previous classes to demonstrate 
        how financial engineering techniques work together in real-world applications.
        """)
        
        synthesis_topic = st.selectbox("Select Integration Topic:", 
                                      ["Complete Derivatives Desk", "Risk Management Framework", "Course Review"])
        
        if synthesis_topic == "Complete Derivatives Desk":
            st.write("**Integrated Derivatives Trading Desk**")
            st.write("Combine pricing, hedging, and risk management for a complete trading operation.")
            
            # Portfolio setup
            st.subheader("Desk Portfolio")
            
            # Sample derivatives book
            book_data = {
                'Instrument': ['SPX Call Options', 'EUR/USD Forwards', 'Interest Rate Swaps', 'Credit Default Swaps'],
                'Notional ($M)': [50.0, 100.0, 200.0, 25.0],
                'Delta': [25.0, -15.0, 0.0, -5.0],
                'Gamma': [2.5, 0.0, 0.0, 0.0],
                'Vega': [15.0, 0.0, 50.0, 8.0],
                'Theta': [-0.8, 0.1, -0.2, -0.1]
            }
            
            book_df = pd.DataFrame(book_data)
            st.dataframe(book_df, use_container_width=True)
            
            # Risk calculations
            total_delta = book_df['Delta'].sum()
            total_gamma = book_df['Gamma'].sum()
            total_vega = book_df['Vega'].sum()
            total_theta = book_df['Theta'].sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Delta", f"{total_delta:.1f}")
            with col2:
                st.metric("Total Gamma", f"{total_gamma:.1f}")
            with col3:
                st.metric("Total Vega", f"{total_vega:.1f}")
            with col4:
                st.metric("Total Theta", f"{total_theta:.1f}")
            
            # Market scenario analysis
            st.subheader("Scenario Analysis")
            
            scenarios = {
                'Base Case': {'Stock': 0, 'FX': 0, 'Rates': 0, 'Vol': 0, 'Credit': 0},
                'Risk-On': {'Stock': 5, 'FX': 2, 'Rates': 10, 'Vol': -20, 'Credit': -50},
                'Risk-Off': {'Stock': -10, 'FX': -5, 'Rates': -25, 'Vol': 50, 'Credit': 100},
                'Central Bank': {'Stock': 2, 'FX': -3, 'Rates': 25, 'Vol': 10, 'Credit': 20}
            }
            
            scenario_pnl = []
            for scenario_name, changes in scenarios.items():
                # Simplified P&L calculation
                pnl = (total_delta * changes['Stock'] * 0.01 +  # Delta P&L
                      total_vega * changes['Vol'] * 0.01 +      # Vega P&L  
                      total_theta * 1 +                         # Theta P&L (1 day)
                      0.5 * total_gamma * (changes['Stock'] * 0.01)**2)  # Gamma P&L
                
                scenario_pnl.append({'Scenario': scenario_name, 'P&L ($000)': pnl})
            
            scenario_df = pd.DataFrame(scenario_pnl)
            st.dataframe(scenario_df, use_container_width=True)
            
            # P&L visualization
            fig_pnl = px.bar(scenario_df, x='Scenario', y='P&L ($000)', 
                            title='Scenario P&L Analysis',
                            color='P&L ($000)', 
                            color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        elif synthesis_topic == "Risk Management Framework":
            st.write("**Comprehensive Risk Management Framework**")
            st.write("Integrate all risk types and management techniques studied in the course.")
            
            # Risk dashboard
            st.subheader("Risk Dashboard")
            
            # Sample risk metrics
            risk_metrics = {
                'Market Risk': {
                    'VaR (1-day, 95%)': '$2.5M',
                    'Expected Shortfall': '$4.2M',
                    'Beta': '1.15',
                    'Duration': '4.2 years'
                },
                'Credit Risk': {
                    'Expected Loss': '$800K',
                    'Unexpected Loss': '$3.2M',
                    'Default Probability': '2.3%',
                    'Credit VaR': '$4.1M'
                },
                'Operational Risk': {
                    'Op Risk Capital': '$5.0M',
                    'Frequency Score': '3.2/5',
                    'Severity Score': '2.8/5',
                    'Key Risk Indicators': '85% Green'
                },
                'Liquidity Risk': {
                    'LCR': '125%',
                    'NSFR': '118%',
                    'Funding Gap': '$0M',
                    'Liquidity Buffer': '$25M'
                }
            }
            
            # Display risk metrics in tabs
            risk_tabs = st.tabs(list(risk_metrics.keys()))
            
            for i, (risk_type, metrics) in enumerate(risk_metrics.items()):
                with risk_tabs[i]:
                    cols = st.columns(len(metrics))
                    for j, (metric, value) in enumerate(metrics.items()):
                        with cols[j]:
                            # Color coding for different risk levels
                            if 'VaR' in metric or 'Loss' in metric:
                                delta_color = 'inverse'
                            else:
                                delta_color = 'normal'
                            st.metric(metric, value)
            
            # Stress testing results
            st.subheader("Integrated Stress Testing Results")
            
            stress_results = {
                'Stress Scenario': ['2008 Financial Crisis', 'COVID-19 Pandemic', 'Interest Rate Shock', 
                                   'Sovereign Debt Crisis', 'Cyber Attack'],
                'Market Risk Impact ($M)': [-15.2, -8.9, -12.1, -6.8, -2.1],
                'Credit Risk Impact ($M)': [-8.5, -12.3, -4.2, -9.7, -1.5],
                'Operational Impact ($M)': [-1.0, -2.1, -0.5, -1.2, -8.9],
                'Total Impact ($M)': [-24.7, -23.3, -16.8, -17.7, -12.5]
            }
            
            stress_df = pd.DataFrame(stress_results)
            st.dataframe(stress_df, use_container_width=True)
            
            # Risk appetite framework
            st.subheader("Risk Appetite Framework")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Risk Limits:**")
                st.write("- Market Risk VaR: $5.0M (Current: $2.5M) ✅")
                st.write("- Credit Risk VaR: $8.0M (Current: $4.1M) ✅") 
                st.write("- Operational Risk Capital: $10.0M (Current: $5.0M) ✅")
                st.write("- Liquidity Coverage Ratio: >110% (Current: 125%) ✅")
            
            with col2:
                st.write("**Risk Monitoring:**")
                st.write("- Daily VaR reporting ✅")
                st.write("- Weekly stress testing ✅")
                st.write("- Monthly ALCO review ✅")
                st.write("- Quarterly model validation ✅")
        
        elif synthesis_topic == "Course Review":
            st.write("**Financial Engineering Course Review**")
            st.write("Summary of key concepts and their practical applications.")
            
            # Course modules summary
            modules_summary = {
                'Class 1: Introduction': {
                    'Key Concepts': ['Time value of money', 'Risk-return relationship', 'Financial markets'],
                    'Applications': ['Investment decisions', 'Valuation basics', 'Risk assessment'],
                    'Tools': ['Present value calculations', 'Future value formulas', 'Basic statistics']
                },
                'Class 2: Time Value of Money': {
                    'Key Concepts': ['Present/future value', 'Annuities', 'Compounding'],
                    'Applications': ['Bond pricing', 'Loan calculations', 'Investment analysis'],
                    'Tools': ['Financial calculators', 'Excel functions', 'Python libraries']
                },
                'Class 3: Bond Valuation': {
                    'Key Concepts': ['Bond pricing', 'Yield measures', 'Duration/convexity'],
                    'Applications': ['Fixed income trading', 'Portfolio management', 'Risk hedging'],
                    'Tools': ['Bond pricing models', 'Yield calculations', 'Duration analysis']
                },
                'Class 4: Stock Valuation': {
                    'Key Concepts': ['DDM', 'DCF analysis', 'Relative valuation'],
                    'Applications': ['Equity research', 'Investment banking', 'Portfolio construction'],
                    'Tools': ['Valuation models', 'Financial ratios', 'Scenario analysis']
                },
                'Class 5: Derivatives': {
                    'Key Concepts': ['Forwards/futures', 'Swaps', 'Risk management'],
                    'Applications': ['Hedging', 'Speculation', 'Arbitrage'],
                    'Tools': ['Pricing formulas', 'Greek calculations', 'Monte Carlo methods']
                },
                'Class 6: Options': {
                    'Key Concepts': ['Black-Scholes', 'Option Greeks', 'Trading strategies'],
                    'Applications': ['Options trading', 'Portfolio hedging', 'Volatility trading'],
                    'Tools': ['Option pricing models', 'Greeks analysis', 'Strategy backtesting']
                },
                'Class 7: Portfolio Theory': {
                    'Key Concepts': ['MPT', 'Efficient frontier', 'CAPM'],
                    'Applications': ['Asset allocation', 'Performance measurement', 'Risk budgeting'],
                    'Tools': ['Optimization algorithms', 'Risk models', 'Performance attribution']
                },
                'Class 8: Risk Management': {
                    'Key Concepts': ['VaR', 'Stress testing', 'Risk metrics'],
                    'Applications': ['Risk limits', 'Capital allocation', 'Regulatory compliance'],
                    'Tools': ['Risk systems', 'Backtesting', 'Model validation']
                },
                'Class 9: Monte Carlo': {
                    'Key Concepts': ['Random simulation', 'Path-dependent pricing', 'Scenario generation'],
                    'Applications': ['Complex derivatives', 'Risk assessment', 'Model validation'],
                    'Tools': ['Random number generation', 'Variance reduction', 'Convergence analysis']
                },
                'Class 10: Advanced Topics': {
                    'Key Concepts': ['Exotic derivatives', 'Credit risk', 'Algorithmic trading'],
                    'Applications': ['Structured products', 'Credit modeling', 'Quantitative strategies'],
                    'Tools': ['Advanced models', 'Machine learning', 'High-frequency systems']
                }
            }
            
            # Interactive course map
            selected_module = st.selectbox("Select Module for Detail:", list(modules_summary.keys()))
            
            module_info = modules_summary[selected_module]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Key Concepts")
                for concept in module_info['Key Concepts']:
                    st.write(f"• {concept}")
            
            with col2:
                st.subheader("Applications")
                for app in module_info['Applications']:
                    st.write(f"• {app}")
            
            with col3:
                st.subheader("Tools")
                for tool in module_info['Tools']:
                    st.write(f"• {tool}")
            
            # Career pathways
            st.subheader("Career Applications")
            
            career_paths = {
                'Investment Banking': ['Valuation models', 'DCF analysis', 'Comparable company analysis'],
                'Asset Management': ['Portfolio optimization', 'Risk management', 'Performance attribution'],
                'Trading': ['Derivatives pricing', 'Risk management', 'Algorithmic strategies'],
                'Risk Management': ['VaR models', 'Stress testing', 'Regulatory compliance'],
                'Quantitative Research': ['Mathematical modeling', 'Statistical analysis', 'Algorithm development'],
                'Corporate Finance': ['Capital budgeting', 'Cost of capital', 'Financial planning']
            }
            
            selected_career = st.selectbox("Explore Career Path:", list(career_paths.keys()))
            
            st.write(f"**Key Skills for {selected_career}:**")
            for skill in career_paths[selected_career]:
                st.write(f"✓ {skill}")
        
        # Final Assessment
        st.subheader("Final Course Assessment")
        
        q1 = st.radio(
            "Which technique is most appropriate for valuing a path-dependent derivative?",
            ["Black-Scholes formula", "Monte Carlo simulation", "Binomial tree", "Closed-form solution"],
            key="final_q1"
        )
        
        q2 = st.radio(
            "In portfolio optimization, the efficient frontier represents:",
            ["All possible portfolios", "Maximum return portfolios", "Minimum risk portfolios", "Optimal risk-return combinations"],
            key="final_q2"
        )
        
        q3 = st.radio(
            "Which risk measure captures tail risk better than VaR?",
            ["Standard deviation", "Beta", "Expected Shortfall", "Correlation"],
            key="final_q3"
        )
        
        q4 = st.radio(
            "The primary purpose of financial derivatives is:",
            ["Speculation only", "Risk management and price discovery", "Arbitrage only", "Regulation compliance"],
            key="final_q4"
        )
        
        if st.button("Submit Final Assessment"):
            score = 0
            
            if q1 == "Monte Carlo simulation":
                st.success("Q1: Correct! ✅")
                score += 1
            else:
                st.error("Q1: Incorrect. Monte Carlo simulation is ideal for path-dependent derivatives.")
            
            if q2 == "Optimal risk-return combinations":
                st.success("Q2: Correct! ✅")
                score += 1
            else:
                st.error("Q2: Incorrect. The efficient frontier shows optimal risk-return combinations.")
            
            if q3 == "Expected Shortfall":
                st.success("Q3: Correct! ✅")
                score += 1
            else:
                st.error("Q3: Incorrect. Expected Shortfall measures the severity of tail losses.")
            
            if q4 == "Risk management and price discovery":
                st.success("Q4: Correct! ✅")
                score += 1
            else:
                st.error("Q4: Incorrect. Derivatives serve multiple purposes including risk management.")
            
            st.write(f"Your final score: {score}/4 ({score/4*100:.0f}%)")
            
            if score >= 3:
                st.balloons()
                progress_tracker.mark_class_completed("Class 10: Advanced Topics")
                progress_tracker.set_class_score("Class 10: Advanced Topics", (score/4) * 100)
                
                # Check if all classes completed
                completed = progress_tracker.get_completed_classes()
                if len(completed) == 10:
                    st.success("🎉🎉🎉 CONGRATULATIONS! 🎉🎉🎉")
                    st.success("You have successfully completed the entire Financial Engineering course!")
                    st.success("You are now equipped with comprehensive knowledge of:")
                    st.success("✓ Mathematical Finance • ✓ Derivatives Pricing • ✓ Risk Management")
                    st.success("✓ Portfolio Theory • ✓ Monte Carlo Methods • ✓ Advanced Applications")
                    
                    # Generate certificate data
                    completion_date = pd.Timestamp.now().strftime("%B %d, %Y")
                    st.markdown(f"""
                    ---
                    ## 🏆 CERTIFICATE OF COMPLETION 🏆
                    
                    **This certifies that you have successfully completed**
                    
                    # Financial Engineering Course
                    **Fundamentals and Advanced Applications**
                    
                    Completed on: **{completion_date}**
                    
                    **Course Coverage:**
                    - 10 comprehensive classes
                    - Mathematical foundations
                    - Practical applications
                    - Interactive exercises
                    - Real-world case studies
                    
                    ---
                    """)
                else:
                    st.success("🎉 Excellent! You've completed the advanced topics!")
            else:
                st.info("Review the course materials and try the assessment again.")
    
    # Download materials
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Advanced Resources")
    
    advanced_code = """
# Advanced Financial Engineering Applications

import numpy as np
from scipy.optimize import minimize

def merton_default_probability(V, D, r, sigma_V, T):
    '''Merton model default probability'''
    from scipy.stats import norm
    
    d2 = (np.log(V/D) + (r - 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
    return norm.cdf(-d2)

def rainbow_option_mc(S0_list, K, T, r, sigma_list, corr_matrix, n_sims, option_type='best_of_calls'):
    '''Price rainbow option using Monte Carlo'''
    n_assets = len(S0_list)
    
    # Generate correlated random numbers
    L = np.linalg.cholesky(corr_matrix)
    Z = np.random.standard_normal((n_sims, n_assets))
    Z_corr = Z @ L.T
    
    # Generate final asset prices
    ST = np.zeros((n_sims, n_assets))
    for i in range(n_assets):
        ST[:, i] = S0_list[i] * np.exp(
            (r - 0.5*sigma_list[i]**2)*T + sigma_list[i]*np.sqrt(T)*Z_corr[:, i]
        )
    
    # Calculate payoffs
    if option_type == 'best_of_calls':
        best_prices = np.max(ST, axis=1)
        payoffs = np.maximum(best_prices - K, 0)
    elif option_type == 'worst_of_calls':
        worst_prices = np.min(ST, axis=1)
        payoffs = np.maximum(worst_prices - K, 0)
    
    # Price option
    option_price = np.exp(-r*T) * np.mean(payoffs)
    return option_price

def mean_reversion_strategy(prices, lookback=20, entry_std=2.0, exit_std=0.5):
    '''Generate mean reversion trading signals'''
    signals = np.zeros(len(prices))
    
    for i in range(lookback, len(prices)):
        window = prices[i-lookback:i]
        mean_price = np.mean(window)
        std_price = np.std(window)
        
        if std_price > 0:
            z_score = (prices[i] - mean_price) / std_price
            
            if z_score > entry_std:
                signals[i] = -1  # Sell signal
            elif z_score < -entry_std:
                signals[i] = 1   # Buy signal
            elif abs(z_score) < exit_std:
                signals[i] = 0   # Exit signal
    
    return signals

def portfolio_var_monte_carlo(weights, expected_returns, cov_matrix, 
                             portfolio_value, confidence_level=0.05, n_sims=100000):
    '''Calculate portfolio VaR using Monte Carlo'''
    
    # Generate random portfolio returns
    portfolio_returns = np.random.multivariate_normal(expected_returns, cov_matrix, n_sims)
    portfolio_pnl = portfolio_value * (portfolio_returns @ weights)
    
    # Calculate VaR and Expected Shortfall
    var = -np.percentile(portfolio_pnl, confidence_level * 100)
    es = -np.mean(portfolio_pnl[portfolio_pnl <= -var])
    
    return var, es

# Example usage - Rainbow option
S0 = [100, 110, 95]  # Initial prices
sigma = [0.2, 0.25, 0.3]  # Volatilities
corr = np.array([[1.0, 0.5, 0.3],
                 [0.5, 1.0, 0.4],
                 [0.3, 0.4, 1.0]])

rainbow_price = rainbow_option_mc(S0, 100, 1.0, 0.05, sigma, corr, 50000)
print(f"Rainbow option price: ${rainbow_price:.4f}")
"""
    
    st.sidebar.download_button(
        label="💻 Download Advanced Code",
        data=advanced_code,
        file_name="advanced_financial_engineering.py",
        mime="text/python"
    )
