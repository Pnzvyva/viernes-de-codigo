import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 9: Métodos de Monte Carlo")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Fundamentos de Monte Carlo", "Simulación de Precios", "Valoración de Opciones", "Aplicaciones de Riesgo"])
    
    with tab1:
        st.subheader("Introduction to Monte Carlo Methods")
        
        st.write("""
        Monte Carlo methods are computational algorithms that rely on repeated random sampling 
        to obtain numerical results. In finance, they are used to model complex systems and 
        calculate the value of instruments with multiple sources of uncertainty.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Key Characteristics:**
            - Uses random number generation
            - Approximates solutions through sampling
            - Particularly useful for high-dimensional problems
            - Provides confidence intervals
            - Handles complex payoff structures
            """)
        
        with col2:
            st.write("""
            **Financial Applications:**
            - Option pricing (especially exotic options)
            - Risk management (VaR, stress testing)
            - Portfolio optimization
            - Credit risk modeling
            - Asset liability management
            """)
        
        st.subheader("Random Number Generation")
        
        with st.expander("🎲 Understanding Random Numbers"):
            st.write("""
            Monte Carlo simulations require high-quality random numbers. In practice, we use:
            - **Pseudo-random numbers**: Deterministic algorithms that produce sequences appearing random
            - **Quasi-random numbers**: Low-discrepancy sequences for better convergence
            - **Antithetic variates**: Variance reduction technique using negatively correlated samples
            """)
        
        # Simple Monte Carlo example: Estimating π
        st.subheader("Simple Example: Estimating π")
        
        st.write("We can estimate π by randomly throwing darts at a square containing a circle.")
        
        n_samples = st.slider("Number of samples:", 1000, 50000, 10000, step=1000)
        
        if st.button("Run π Estimation"):
            np.random.seed(42)
            
            # Generate random points in [-1,1] x [-1,1]
            x = np.random.uniform(-1, 1, n_samples)
            y = np.random.uniform(-1, 1, n_samples)
            
            # Check if points are inside unit circle
            inside_circle = (x**2 + y**2) <= 1
            
            # Estimate π
            pi_estimate = 4 * np.sum(inside_circle) / n_samples
            
            # Create visualization
            sample_size = min(2000, n_samples)  # Limit points for visualization
            idx = np.random.choice(n_samples, sample_size, replace=False)
            
            colors = ['red' if inside_circle[i] else 'blue' for i in idx]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x[idx], y=y[idx],
                mode='markers',
                marker=dict(color=colors, size=3),
                name='Random Points'
            ))
            
            # Add circle
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            fig.add_trace(go.Scatter(
                x=circle_x, y=circle_y,
                mode='lines',
                line=dict(color='black', width=2),
                name='Unit Circle'
            ))
            
            fig.update_layout(
                title=f"Monte Carlo Estimation of π (n={n_samples:,})",
                xaxis_title="X",
                yaxis_title="Y",
                width=600, height=600,
                xaxis=dict(range=[-1.1, 1.1]),
                yaxis=dict(range=[-1.1, 1.1])
            )
            fig.update_yaxis(scaleanchor="x", scaleratio=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated π", f"{pi_estimate:.4f}")
            with col2:
                st.metric("Actual π", f"{np.pi:.4f}")
            with col3:
                error = abs(pi_estimate - np.pi)
                st.metric("Error", f"{error:.4f}")
        
        st.subheader("Monte Carlo Process")
        
        with st.expander("📊 General Monte Carlo Algorithm"):
            st.write("""
            **Step 1:** Define the problem and identify random variables
            **Step 2:** Generate random samples from appropriate distributions
            **Step 3:** Calculate the outcome for each sample
            **Step 4:** Aggregate results (mean, percentiles, etc.)
            **Step 5:** Estimate accuracy and confidence intervals
            """)
        
        # Law of Large Numbers demonstration
        st.subheader("Convergence Properties")
        
        if st.button("Demonstrate Convergence"):
            np.random.seed(42)
            
            # Simulate convergence of sample mean
            true_mean = 0.05  # 5% daily return
            true_std = 0.02   # 2% daily volatility
            
            max_samples = 10000
            sample_sizes = np.logspace(1, np.log10(max_samples), 100, dtype=int)
            
            # Generate all samples at once
            all_samples = np.random.normal(true_mean, true_std, max_samples)
            
            # Calculate running means
            running_means = []
            running_stds = []
            
            for n in sample_sizes:
                sample_mean = np.mean(all_samples[:n])
                sample_std = np.std(all_samples[:n], ddof=1)
                running_means.append(sample_mean)
                running_stds.append(sample_std)
            
            # Plot convergence
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sample_sizes, y=running_means,
                mode='lines', name='Sample Mean',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=sample_sizes, y=running_stds,
                mode='lines', name='Sample Std Dev',
                line=dict(color='red'), yaxis='y2'
            ))
            
            # Add true values as horizontal lines
            fig.add_hline(y=true_mean, line_dash="dash", line_color="blue", 
                         annotation_text="True Mean")
            fig.add_hline(y=true_std, line_dash="dash", line_color="red", 
                         annotation_text="True Std Dev")
            
            fig.update_layout(
                title="Monte Carlo Convergence",
                xaxis_title="Sample Size",
                yaxis_title="Sample Mean",
                yaxis2=dict(title="Sample Std Dev", overlaying='y', side='right')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Stock Price Simulation")
        
        st.write("""
        Stock prices are commonly modeled using Geometric Brownian Motion (GBM), 
        which assumes that price changes follow a log-normal distribution.
        """)
        
        with st.expander("📈 Geometric Brownian Motion"):
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.write("Discrete form:")
            st.latex(r"S_{t+1} = S_t \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}Z\right)")
            st.write("Where Z ~ N(0,1) is a standard normal random variable")
        
        st.subheader("Single Asset Simulation")
        
        # Parameters for simulation
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.number_input("Initial Stock Price ($)", value=100.0, min_value=0.01)
            mu = st.number_input("Drift (μ) - Annual %", value=8.0, min_value=-50.0, max_value=50.0) / 100
        with col2:
            sigma = st.number_input("Volatility (σ) - Annual %", value=20.0, min_value=1.0, max_value=100.0) / 100
            T = st.number_input("Time Horizon (years)", value=1.0, min_value=0.1, max_value=10.0)
        with col3:
            n_paths = st.number_input("Number of Paths", value=1000, min_value=10, max_value=10000)
            n_steps = st.number_input("Time Steps", value=252, min_value=10, max_value=1000)
        
        if st.button("Simulate Stock Paths"):
            # Time parameters
            dt = T / n_steps
            t = np.linspace(0, T, n_steps + 1)
            
            # Generate random paths
            np.random.seed(42)
            Z = np.random.standard_normal((n_paths, n_steps))
            
            # Initialize price array
            S = np.zeros((n_paths, n_steps + 1))
            S[:, 0] = S0
            
            # Generate paths using GBM formula
            for i in range(n_steps):
                S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
            
            # Create DataFrame for visualization
            sample_paths = min(50, n_paths)  # Show limited paths for clarity
            path_indices = np.random.choice(n_paths, sample_paths, replace=False)
            
            fig = go.Figure()
            
            # Plot sample paths
            for i in path_indices:
                fig.add_trace(go.Scatter(
                    x=t, y=S[i, :],
                    mode='lines',
                    opacity=0.3,
                    line=dict(width=1),
                    showlegend=False,
                    hovertemplate='Time: %{x:.2f}<br>Price: $%{y:.2f}<extra></extra>'
                ))
            
            # Add mean path
            mean_path = np.mean(S, axis=0)
            fig.add_trace(go.Scatter(
                x=t, y=mean_path,
                mode='lines',
                line=dict(color='red', width=3),
                name='Mean Path'
            ))
            
            # Add theoretical mean
            theoretical_mean = S0 * np.exp(mu * t)
            fig.add_trace(go.Scatter(
                x=t, y=theoretical_mean,
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='Theoretical Mean'
            ))
            
            fig.update_layout(
                title=f"Monte Carlo Stock Price Simulation ({n_paths:,} paths)",
                xaxis_title="Time (years)",
                yaxis_title="Stock Price ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            final_prices = S[:, -1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
            with col2:
                st.metric("Median Final Price", f"${np.median(final_prices):.2f}")
            with col3:
                st.metric("Std Dev", f"${np.std(final_prices):.2f}")
            with col4:
                theoretical_final = S0 * np.exp(mu * T)
                st.metric("Theoretical Mean", f"${theoretical_final:.2f}")
            
            # Distribution of final prices
            fig_hist = px.histogram(final_prices, nbins=50, title="Distribution of Final Stock Prices")
            fig_hist.add_vline(x=np.mean(final_prices), line_dash="dash", 
                              annotation_text="Sample Mean")
            fig_hist.add_vline(x=theoretical_final, line_dash="dot", 
                              annotation_text="Theoretical Mean")
            fig_hist.update_layout(xaxis_title="Final Stock Price ($)", yaxis_title="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Multi-asset simulation
        st.subheader("Multi-Asset Simulation")
        
        st.write("Simulate correlated stock prices using Cholesky decomposition.")
        
        # Parameters for multi-asset
        n_assets = st.selectbox("Number of Assets:", [2, 3, 4], index=0)
        
        if n_assets >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Asset Parameters:**")
                asset_params = []
                for i in range(n_assets):
                    st.write(f"Asset {i+1}:")
                    price = st.number_input(f"Initial Price {i+1}", value=100.0, key=f"price_{i}")
                    drift = st.number_input(f"Drift {i+1} (%)", value=8.0 + i*2, key=f"drift_{i}") / 100
                    vol = st.number_input(f"Volatility {i+1} (%)", value=20.0 + i*5, key=f"vol_{i}") / 100
                    asset_params.append([price, drift, vol])
            
            with col2:
                st.write("**Correlation Matrix:**")
                # Initialize correlation matrix
                corr_matrix = np.eye(n_assets)
                
                # Input correlation coefficients
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        corr = st.number_input(
                            f"Correlation {i+1}-{j+1}:",
                            value=0.3, min_value=-0.99, max_value=0.99,
                            step=0.1, key=f"corr_{i}_{j}"
                        )
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
                
                # Display correlation matrix
                st.write("Correlation Matrix:")
                st.dataframe(pd.DataFrame(corr_matrix, 
                           columns=[f"Asset {i+1}" for i in range(n_assets)],
                           index=[f"Asset {i+1}" for i in range(n_assets)]))
            
            if st.button("Simulate Multi-Asset Paths"):
                # Simulation parameters
                T_multi = 1.0
                n_steps_multi = 252
                n_paths_multi = 1000
                dt = T_multi / n_steps_multi
                
                # Extract parameters
                S0_multi = np.array([param[0] for param in asset_params])
                mu_multi = np.array([param[1] for param in asset_params])
                sigma_multi = np.array([param[2] for param in asset_params])
                
                # Cholesky decomposition for correlation
                L = np.linalg.cholesky(corr_matrix)
                
                # Generate correlated random numbers
                np.random.seed(42)
                Z_independent = np.random.standard_normal((n_paths_multi, n_steps_multi, n_assets))
                Z_correlated = np.zeros_like(Z_independent)
                
                for path in range(n_paths_multi):
                    for step in range(n_steps_multi):
                        Z_correlated[path, step, :] = L @ Z_independent[path, step, :]
                
                # Initialize price arrays
                S_multi = np.zeros((n_paths_multi, n_steps_multi + 1, n_assets))
                S_multi[:, 0, :] = S0_multi
                
                # Generate correlated paths
                for path in range(n_paths_multi):
                    for step in range(n_steps_multi):
                        for asset in range(n_assets):
                            S_multi[path, step + 1, asset] = S_multi[path, step, asset] * np.exp(
                                (mu_multi[asset] - 0.5 * sigma_multi[asset]**2) * dt + 
                                sigma_multi[asset] * np.sqrt(dt) * Z_correlated[path, step, asset]
                            )
                
                # Plot results
                fig_multi = go.Figure()
                t = np.linspace(0, T_multi, n_steps_multi + 1)
                colors = ['blue', 'red', 'green', 'purple']
                
                for asset in range(n_assets):
                    # Plot mean path for each asset
                    mean_path = np.mean(S_multi[:, :, asset], axis=0)
                    fig_multi.add_trace(go.Scatter(
                        x=t, y=mean_path,
                        mode='lines',
                        line=dict(color=colors[asset % len(colors)], width=3),
                        name=f'Asset {asset+1} Mean'
                    ))
                
                fig_multi.update_layout(
                    title="Multi-Asset Monte Carlo Simulation (Mean Paths)",
                    xaxis_title="Time (years)",
                    yaxis_title="Stock Price ($)"
                )
                
                st.plotly_chart(fig_multi, use_container_width=True)
                
                # Correlation analysis of final prices
                final_prices_multi = S_multi[:, -1, :]
                realized_corr = np.corrcoef(final_prices_multi.T)
                
                st.subheader("Realized vs Target Correlations")
                
                comparison_data = []
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        comparison_data.append({
                            'Asset Pair': f"{i+1}-{j+1}",
                            'Target Correlation': corr_matrix[i, j],
                            'Realized Correlation': realized_corr[i, j],
                            'Difference': realized_corr[i, j] - corr_matrix[i, j]
                        })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    with tab3:
        st.subheader("Monte Carlo Option Pricing")
        
        st.write("""
        Monte Carlo methods are particularly powerful for pricing complex options where 
        analytical solutions don't exist, such as path-dependent options.
        """)
        
        option_type = st.selectbox("Select Option Type:", 
                                  ["European Options", "Asian Options", "Barrier Options", "Lookback Options"])
        
        if option_type == "European Options":
            st.write("**European Option Pricing via Monte Carlo**")
            st.write("Compare Monte Carlo results with analytical Black-Scholes prices.")
            
            col1, col2 = st.columns(2)
            with col1:
                S0_opt = st.number_input("Current Stock Price", value=100.0, key="eur_S0")
                K_opt = st.number_input("Strike Price", value=100.0, key="eur_K")
                T_opt = st.number_input("Time to Expiration (years)", value=0.25, key="eur_T")
            with col2:
                r_opt = st.number_input("Risk-free Rate (%)", value=5.0, key="eur_r") / 100
                sigma_opt = st.number_input("Volatility (%)", value=20.0, key="eur_sigma") / 100
                n_sims = st.number_input("Number of Simulations", value=100000, min_value=1000, key="eur_sims")
            
            call_or_put = st.radio("Option Type:", ["Call", "Put"], key="eur_type")
            
            if st.button("Price European Option"):
                np.random.seed(42)
                
                # Generate random stock prices at expiration
                Z = np.random.standard_normal(n_sims)
                ST = S0_opt * np.exp((r_opt - 0.5 * sigma_opt**2) * T_opt + sigma_opt * np.sqrt(T_opt) * Z)
                
                # Calculate payoffs
                if call_or_put == "Call":
                    payoffs = np.maximum(ST - K_opt, 0)
                    bs_price = FinancialCalculations.black_scholes_call(S0_opt, K_opt, T_opt, r_opt, sigma_opt)
                else:
                    payoffs = np.maximum(K_opt - ST, 0)
                    bs_price = FinancialCalculations.black_scholes_put(S0_opt, K_opt, T_opt, r_opt, sigma_opt)
                
                # Discount to present value
                mc_price = np.exp(-r_opt * T_opt) * np.mean(payoffs)
                
                # Calculate standard error
                standard_error = np.exp(-r_opt * T_opt) * np.std(payoffs) / np.sqrt(n_sims)
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Monte Carlo Price", f"${mc_price:.4f}")
                with col2:
                    st.metric("Black-Scholes Price", f"${bs_price:.4f}")
                with col3:
                    error = abs(mc_price - bs_price)
                    st.metric("Pricing Error", f"${error:.4f}")
                
                st.info(f"Standard Error: ±${1.96 * standard_error:.4f} (95% confidence interval)")
                
                # Convergence analysis
                sample_sizes = np.logspace(2, np.log10(n_sims), 50, dtype=int)
                running_prices = []
                
                for n in sample_sizes:
                    running_price = np.exp(-r_opt * T_opt) * np.mean(payoffs[:n])
                    running_prices.append(running_price)
                
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=sample_sizes, y=running_prices,
                    mode='lines', name='Monte Carlo Price'
                ))
                fig_conv.add_hline(y=bs_price, line_dash="dash", 
                                  annotation_text="Black-Scholes Price")
                
                fig_conv.update_layout(
                    title="Monte Carlo Convergence",
                    xaxis_title="Number of Simulations",
                    yaxis_title="Option Price ($)",
                    xaxis_type="log"
                )
                
                st.plotly_chart(fig_conv, use_container_width=True)
        
        elif option_type == "Asian Options":
            st.write("**Asian (Average Price) Option Pricing**")
            st.write("Option payoff depends on the average stock price over the option's life.")
            
            col1, col2 = st.columns(2)
            with col1:
                S0_asian = st.number_input("Current Stock Price", value=100.0, key="asian_S0")
                K_asian = st.number_input("Strike Price", value=100.0, key="asian_K")
                T_asian = st.number_input("Time to Expiration (years)", value=0.25, key="asian_T")
                n_steps_asian = st.number_input("Monitoring Steps", value=50, min_value=10, key="asian_steps")
            with col2:
                r_asian = st.number_input("Risk-free Rate (%)", value=5.0, key="asian_r") / 100
                sigma_asian = st.number_input("Volatility (%)", value=20.0, key="asian_sigma") / 100
                n_sims_asian = st.number_input("Number of Simulations", value=50000, min_value=1000, key="asian_sims")
                
            asian_type = st.radio("Asian Option Type:", ["Arithmetic Average", "Geometric Average"])
            call_put_asian = st.radio("Call or Put:", ["Call", "Put"], key="asian_cp")
            
            if st.button("Price Asian Option"):
                dt = T_asian / n_steps_asian
                np.random.seed(42)
                
                # Generate stock price paths
                Z = np.random.standard_normal((n_sims_asian, n_steps_asian))
                S = np.zeros((n_sims_asian, n_steps_asian + 1))
                S[:, 0] = S0_asian
                
                for i in range(n_steps_asian):
                    S[:, i + 1] = S[:, i] * np.exp(
                        (r_asian - 0.5 * sigma_asian**2) * dt + sigma_asian * np.sqrt(dt) * Z[:, i]
                    )
                
                # Calculate averages
                if asian_type == "Arithmetic Average":
                    averages = np.mean(S, axis=1)
                else:  # Geometric Average
                    averages = np.exp(np.mean(np.log(S), axis=1))
                
                # Calculate payoffs
                if call_put_asian == "Call":
                    payoffs = np.maximum(averages - K_asian, 0)
                else:
                    payoffs = np.maximum(K_asian - averages, 0)
                
                # Price option
                asian_price = np.exp(-r_asian * T_asian) * np.mean(payoffs)
                standard_error = np.exp(-r_asian * T_asian) * np.std(payoffs) / np.sqrt(n_sims_asian)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Asian Option Price", f"${asian_price:.4f}")
                    st.metric("Standard Error", f"±${1.96 * standard_error:.4f}")
                with col2:
                    st.metric("Average of Averages", f"${np.mean(averages):.2f}")
                    st.metric("Payoff Rate", f"{np.mean(payoffs > 0)*100:.1f}%")
                
                # Distribution of averages
                fig_asian = px.histogram(averages, nbins=50, title=f"Distribution of {asian_type} Prices")
                fig_asian.add_vline(x=K_asian, line_dash="dash", annotation_text="Strike Price")
                st.plotly_chart(fig_asian, use_container_width=True)
        
        elif option_type == "Barrier Options":
            st.write("**Barrier Option Pricing**")
            st.write("Options that are activated or deactivated when the stock price crosses a barrier.")
            
            col1, col2 = st.columns(2)
            with col1:
                S0_barrier = st.number_input("Current Stock Price", value=100.0, key="barrier_S0")
                K_barrier = st.number_input("Strike Price", value=100.0, key="barrier_K")
                barrier_level = st.number_input("Barrier Level", value=120.0, key="barrier_level")
                T_barrier = st.number_input("Time to Expiration (years)", value=0.25, key="barrier_T")
            with col2:
                r_barrier = st.number_input("Risk-free Rate (%)", value=5.0, key="barrier_r") / 100
                sigma_barrier = st.number_input("Volatility (%)", value=25.0, key="barrier_sigma") / 100
                n_sims_barrier = st.number_input("Number of Simulations", value=50000, min_value=1000, key="barrier_sims")
                n_steps_barrier = st.number_input("Monitoring Steps", value=100, min_value=50, key="barrier_steps")
            
            barrier_type = st.selectbox("Barrier Type:", ["Up-and-Out Call", "Up-and-In Call", "Down-and-Out Put"])
            
            if st.button("Price Barrier Option"):
                dt = T_barrier / n_steps_barrier
                np.random.seed(42)
                
                # Generate stock price paths
                Z = np.random.standard_normal((n_sims_barrier, n_steps_barrier))
                S = np.zeros((n_sims_barrier, n_steps_barrier + 1))
                S[:, 0] = S0_barrier
                
                for i in range(n_steps_barrier):
                    S[:, i + 1] = S[:, i] * np.exp(
                        (r_barrier - 0.5 * sigma_barrier**2) * dt + sigma_barrier * np.sqrt(dt) * Z[:, i]
                    )
                
                # Check barrier conditions
                if barrier_type == "Up-and-Out Call":
                    barrier_crossed = np.any(S >= barrier_level, axis=1)
                    payoffs = np.where(~barrier_crossed, np.maximum(S[:, -1] - K_barrier, 0), 0)
                elif barrier_type == "Up-and-In Call":
                    barrier_crossed = np.any(S >= barrier_level, axis=1)
                    payoffs = np.where(barrier_crossed, np.maximum(S[:, -1] - K_barrier, 0), 0)
                else:  # Down-and-Out Put
                    barrier_crossed = np.any(S <= barrier_level, axis=1)
                    payoffs = np.where(~barrier_crossed, np.maximum(K_barrier - S[:, -1], 0), 0)
                
                # Price option
                barrier_price = np.exp(-r_barrier * T_barrier) * np.mean(payoffs)
                
                # Compare with vanilla option
                if "Call" in barrier_type:
                    vanilla_price = FinancialCalculations.black_scholes_call(S0_barrier, K_barrier, T_barrier, r_barrier, sigma_barrier)
                else:
                    vanilla_price = FinancialCalculations.black_scholes_put(S0_barrier, K_barrier, T_barrier, r_barrier, sigma_barrier)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Barrier Option Price", f"${barrier_price:.4f}")
                with col2:
                    st.metric("Vanilla Option Price", f"${vanilla_price:.4f}")
                with col3:
                    knockout_rate = np.mean(barrier_crossed) * 100
                    st.metric("Barrier Hit Rate", f"{knockout_rate:.1f}%")
                
                # Show sample paths
                sample_paths = min(100, n_sims_barrier)
                path_indices = np.random.choice(n_sims_barrier, sample_paths, replace=False)
                
                fig_barrier = go.Figure()
                
                t = np.linspace(0, T_barrier, n_steps_barrier + 1)
                for i in path_indices[:10]:  # Show only first 10 for clarity
                    color = 'red' if barrier_crossed[i] else 'blue'
                    fig_barrier.add_trace(go.Scatter(
                        x=t, y=S[i, :],
                        mode='lines',
                        line=dict(color=color, width=1),
                        opacity=0.6,
                        showlegend=False
                    ))
                
                fig_barrier.add_hline(y=barrier_level, line_dash="dash", 
                                     annotation_text="Barrier Level")
                fig_barrier.add_hline(y=S0_barrier, line_dash="dot", 
                                     annotation_text="Initial Price")
                
                fig_barrier.update_layout(
                    title=f"{barrier_type} - Sample Paths (Red = Barrier Hit)",
                    xaxis_title="Time (years)",
                    yaxis_title="Stock Price ($)"
                )
                
                st.plotly_chart(fig_barrier, use_container_width=True)
    
    with tab4:
        st.subheader("Risk Management Applications")
        
        st.write("""
        Monte Carlo methods are extensively used in risk management for scenario analysis, 
        stress testing, and calculating risk metrics like VaR and Expected Shortfall.
        """)
        
        risk_app = st.selectbox("Select Risk Application:", 
                               ["Portfolio VaR", "Stress Testing", "Credit Risk Modeling"])
        
        if risk_app == "Portfolio VaR":
            st.write("**Portfolio Value-at-Risk using Monte Carlo**")
            
            # Portfolio setup
            st.subheader("Portfolio Configuration")
            
            n_assets_var = st.selectbox("Number of Assets:", [2, 3, 4, 5], index=2)
            
            # Asset parameters
            asset_weights = []
            asset_returns = []
            asset_vols = []
            
            for i in range(n_assets_var):
                col1, col2, col3 = st.columns(3)
                with col1:
                    weight = st.number_input(f"Asset {i+1} Weight (%)", value=100/n_assets_var, 
                                           min_value=0.0, max_value=100.0, key=f"var_weight_{i}")
                    asset_weights.append(weight/100)
                with col2:
                    ret = st.number_input(f"Expected Return (%)", value=8.0 + i*2, key=f"var_ret_{i}")
                    asset_returns.append(ret/100)
                with col3:
                    vol = st.number_input(f"Volatility (%)", value=15.0 + i*5, key=f"var_vol_{i}")
                    asset_vols.append(vol/100)
            
            # Correlation matrix
            st.subheader("Asset Correlations")
            corr_var = np.eye(n_assets_var)
            
            for i in range(n_assets_var):
                for j in range(i+1, n_assets_var):
                    corr_val = st.number_input(f"Correlation {i+1}-{j+1}:", 
                                             value=0.3, min_value=-0.99, max_value=0.99,
                                             key=f"var_corr_{i}_{j}")
                    corr_var[i, j] = corr_val
                    corr_var[j, i] = corr_val
            
            portfolio_value = st.number_input("Portfolio Value ($)", value=1000000.0, min_value=1000.0)
            time_horizon = st.selectbox("Time Horizon:", ["1 Day", "1 Week", "1 Month"])
            confidence_level = st.selectbox("Confidence Level:", ["95%", "99%", "99.9%"])
            
            if st.button("Calculate Portfolio VaR"):
                # Convert time horizon to fraction of year
                horizon_map = {"1 Day": 1/252, "1 Week": 5/252, "1 Month": 21/252}
                dt = horizon_map[time_horizon]
                
                # Convert confidence level
                conf_map = {"95%": 0.05, "99%": 0.01, "99.9%": 0.001}
                alpha = conf_map[confidence_level]
                
                # Portfolio parameters
                weights = np.array(asset_weights)
                mu = np.array(asset_returns)
                sigma = np.array(asset_vols)
                
                # Portfolio statistics
                portfolio_return = np.sum(weights * mu)
                cov_matrix = np.outer(sigma, sigma) * corr_var
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Monte Carlo simulation
                n_sims_var = 100000
                np.random.seed(42)
                
                # Generate correlated returns
                L = np.linalg.cholesky(corr_var)
                Z = np.random.standard_normal((n_sims_var, n_assets_var))
                corr_returns = Z @ L.T
                
                # Scale by individual asset parameters
                asset_returns_sim = np.zeros_like(corr_returns)
                for i in range(n_assets_var):
                    asset_returns_sim[:, i] = mu[i] * dt + sigma[i] * np.sqrt(dt) * corr_returns[:, i]
                
                # Calculate portfolio returns
                portfolio_returns = asset_returns_sim @ weights
                portfolio_values = portfolio_value * (1 + portfolio_returns)
                portfolio_pnl = portfolio_values - portfolio_value
                
                # Calculate VaR and ES
                var_mc = -np.percentile(portfolio_pnl, alpha * 100)
                tail_losses = portfolio_pnl[portfolio_pnl <= -var_mc]
                es_mc = -np.mean(tail_losses)
                
                # Parametric VaR for comparison
                var_param = -portfolio_value * (portfolio_return * dt - 1.96 * portfolio_vol * np.sqrt(dt))
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Monte Carlo VaR", f"${var_mc:,.0f}")
                    st.metric("Expected Shortfall", f"${es_mc:,.0f}")
                with col2:
                    st.metric("Parametric VaR", f"${var_param:,.0f}")
                    st.metric("VaR Difference", f"${abs(var_mc - var_param):,.0f}")
                with col3:
                    st.metric("Portfolio Return", f"{portfolio_return*100:.2f}%")
                    st.metric("Portfolio Volatility", f"{portfolio_vol*100:.2f}%")
                
                # Distribution of P&L
                fig_var = px.histogram(portfolio_pnl/1000, nbins=50, 
                                      title=f"Portfolio P&L Distribution ({time_horizon})")
                fig_var.add_vline(x=-var_mc/1000, line_dash="dash", line_color="red",
                                 annotation_text=f"{confidence_level} VaR")
                fig_var.add_vline(x=-es_mc/1000, line_dash="dot", line_color="orange",
                                 annotation_text="Expected Shortfall")
                fig_var.update_layout(xaxis_title="P&L ($000)", yaxis_title="Frequency")
                st.plotly_chart(fig_var, use_container_width=True)
        
        elif risk_app == "Stress Testing":
            st.write("**Monte Carlo Stress Testing**")
            st.write("Generate extreme market scenarios and assess portfolio impact.")
            
            # Stress test parameters
            stress_scenarios = {
                "Market Crash": {"equity": -0.30, "bond": 0.05, "volatility_multiplier": 3.0},
                "Interest Rate Shock": {"equity": -0.15, "bond": -0.20, "volatility_multiplier": 2.0},
                "Inflation Spike": {"equity": -0.20, "bond": -0.15, "volatility_multiplier": 2.5},
                "Liquidity Crisis": {"equity": -0.25, "bond": -0.10, "volatility_multiplier": 4.0}
            }
            
            selected_stress = st.selectbox("Select Stress Scenario:", list(stress_scenarios.keys()))
            portfolio_allocation = st.slider("Equity Allocation (%)", 0, 100, 60)
            bond_allocation = 100 - portfolio_allocation
            
            if st.button("Run Stress Test"):
                scenario = stress_scenarios[selected_stress]
                n_sims_stress = 10000
                
                # Base case parameters
                base_equity_return = 0.08
                base_bond_return = 0.03
                equity_vol = 0.20
                bond_vol = 0.05
                correlation = 0.1
                
                # Stressed parameters
                stressed_equity_return = base_equity_return + scenario["equity"]
                stressed_bond_return = base_bond_return + scenario["bond"]
                stressed_equity_vol = equity_vol * scenario["volatility_multiplier"]
                stressed_bond_vol = bond_vol * scenario["volatility_multiplier"]
                
                np.random.seed(42)
                
                # Generate correlated returns
                cov_matrix = np.array([
                    [stressed_equity_vol**2, correlation * stressed_equity_vol * stressed_bond_vol],
                    [correlation * stressed_equity_vol * stressed_bond_vol, stressed_bond_vol**2]
                ])
                
                mean_returns = np.array([stressed_equity_return, stressed_bond_return])
                
                # Monte Carlo simulation
                returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_sims_stress)
                
                # Portfolio weights
                weights = np.array([portfolio_allocation/100, bond_allocation/100])
                portfolio_returns_stress = returns @ weights
                
                # Calculate portfolio statistics
                mean_return = np.mean(portfolio_returns_stress)
                portfolio_vol_stress = np.std(portfolio_returns_stress)
                var_95_stress = np.percentile(portfolio_returns_stress, 5)
                
                # Results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Return", f"{mean_return*100:.2f}%")
                    st.metric("Volatility", f"{portfolio_vol_stress*100:.2f}%")
                with col2:
                    st.metric("95% VaR", f"{var_95_stress*100:.2f}%")
                    prob_loss = np.mean(portfolio_returns_stress < 0) * 100
                    st.metric("Probability of Loss", f"{prob_loss:.1f}%")
                with col3:
                    worst_case = np.min(portfolio_returns_stress)
                    best_case = np.max(portfolio_returns_stress)
                    st.metric("Worst Case", f"{worst_case*100:.2f}%")
                    st.metric("Best Case", f"{best_case*100:.2f}%")
                
                # Distribution plot
                fig_stress = px.histogram(portfolio_returns_stress*100, nbins=50, 
                                         title=f"Stressed Portfolio Returns - {selected_stress}")
                fig_stress.add_vline(x=0, line_dash="dash", annotation_text="Break-even")
                fig_stress.add_vline(x=var_95_stress*100, line_dash="dot", line_color="red",
                                    annotation_text="95% VaR")
                fig_stress.update_layout(xaxis_title="Portfolio Return (%)", yaxis_title="Frequency")
                st.plotly_chart(fig_stress, use_container_width=True)
        
        # Assessment
        st.subheader("Monte Carlo Methods Assessment")
        
        q1 = st.radio(
            "What is the main advantage of Monte Carlo methods in finance?",
            ["Speed of calculation", "Handling complex payoffs", "Perfect accuracy", "No randomness"],
            key="mc_q1"
        )
        
        q2 = st.radio(
            "The standard error of Monte Carlo estimates decreases as:",
            ["n (sample size)", "√n", "1/√n", "1/n"],
            key="mc_q2"
        )
        
        q3 = st.radio(
            "Which option type is most suitable for Monte Carlo pricing?",
            ["European vanilla options", "Path-dependent options", "Simple binary options", "All equally suitable"],
            key="mc_q3"
        )
        
        if st.button("Submit Assessment"):
            score = 0
            
            if q1 == "Handling complex payoffs":
                st.success("Q1: Correct! ✅")
                score += 1
            else:
                st.error("Q1: Incorrect. Monte Carlo excels at handling complex, path-dependent payoffs.")
            
            if q2 == "1/√n":
                st.success("Q2: Correct! ✅")
                score += 1
            else:
                st.error("Q2: Incorrect. Monte Carlo standard error decreases as 1/√n.")
            
            if q3 == "Path-dependent options":
                st.success("Q3: Correct! ✅")
                score += 1
            else:
                st.error("Q3: Incorrect. Monte Carlo is most valuable for path-dependent options.")
            
            st.write(f"Your score: {score}/3")
            
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Class 9: Monte Carlo Methods")
                progress_tracker.set_class_score("Class 9: Monte Carlo Methods", (score/3) * 100)
                st.success("🎉 Excellent! You've mastered Monte Carlo methods!")
    
    # Download materials
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Monte Carlo Resources")
    
    monte_carlo_code = """
# Monte Carlo Methods in Finance

import numpy as np
from scipy.stats import norm

def geometric_brownian_motion(S0, mu, sigma, T, n_steps, n_paths):
    '''Simulate stock price paths using GBM'''
    dt = T / n_steps
    
    # Generate random shocks
    Z = np.random.standard_normal((n_paths, n_steps))
    
    # Initialize price array
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    
    # Generate paths
    for i in range(n_steps):
        S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
    
    return S

def monte_carlo_option_price(S0, K, T, r, sigma, n_sims, option_type='call'):
    '''Price European option using Monte Carlo'''
    np.random.seed(42)
    
    # Generate final stock prices
    Z = np.random.standard_normal(n_sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Discount to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)
    standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    
    return option_price, standard_error

def asian_option_mc(S0, K, T, r, sigma, n_steps, n_sims, option_type='call', avg_type='arithmetic'):
    '''Price Asian option using Monte Carlo'''
    dt = T / n_steps
    
    # Generate stock price paths
    paths = geometric_brownian_motion(S0, r, sigma, T, n_steps, n_sims)
    
    # Calculate averages
    if avg_type == 'arithmetic':
        averages = np.mean(paths, axis=1)
    else:  # geometric
        averages = np.exp(np.mean(np.log(paths), axis=1))
    
    # Calculate payoffs
    if option_type == 'call':
        payoffs = np.maximum(averages - K, 0)
    else:
        payoffs = np.maximum(K - averages, 0)
    
    # Price option
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def portfolio_var_mc(weights, returns, cov_matrix, portfolio_value, confidence_level=0.05, n_sims=100000):
    '''Calculate portfolio VaR using Monte Carlo'''
    np.random.seed(42)
    
    # Generate correlated returns
    portfolio_returns = np.random.multivariate_normal(returns, cov_matrix, n_sims)
    
    # Calculate portfolio returns
    port_returns = portfolio_returns @ weights
    
    # Calculate P&L
    portfolio_pnl = portfolio_value * port_returns
    
    # Calculate VaR and Expected Shortfall
    var = -np.percentile(portfolio_pnl, confidence_level * 100)
    es = -np.mean(portfolio_pnl[portfolio_pnl <= -var])
    
    return var, es

# Example usage
S0, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
call_price, se = monte_carlo_option_price(S0, K, T, r, sigma, 100000)
print(f"Call option price: ${call_price:.4f} ± ${1.96*se:.4f}")
"""
    
    st.sidebar.download_button(
        label="💻 Download Monte Carlo Code",
        data=monte_carlo_code,
        file_name="monte_carlo_methods.py",
        mime="text/python"
    )
