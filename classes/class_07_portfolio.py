import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 7: Teoría de Portafolios")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Teoría Moderna de Portafolios", "Frontera Eficiente", "CAPM", "Optimización de Portafolios"])
    
    with tab1:
        st.subheader("Modern Portfolio Theory (MPT)")
        
        st.write("""
        Modern Portfolio Theory, developed by Harry Markowitz in 1952, provides a mathematical framework 
        for constructing optimal portfolios that maximize expected return for a given level of risk.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Key MPT Assumptions:**
            - Investors are risk-averse
            - Returns are normally distributed
            - Correlations are stable over time
            - Markets are efficient
            - No transaction costs
            """)
        
        with col2:
            st.write("""
            **Core Concepts:**
            - **Diversification**: Reduces risk without reducing expected return
            - **Efficient Frontier**: Set of optimal portfolios
            - **Risk-Return Tradeoff**: Higher return requires higher risk
            - **Correlation**: Key driver of diversification benefits
            """)
        
        st.subheader("Portfolio Risk and Return")
        
        with st.expander("📊 Portfolio Mathematics"):
            st.write("**Portfolio Expected Return:**")
            st.latex(r"E(R_p) = \sum_{i=1}^{n} w_i E(R_i)")
            
            st.write("**Portfolio Variance (Two Assets):**")
            st.latex(r"\sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\sigma_1\sigma_2\rho_{1,2}")
            
            st.write("**Portfolio Standard Deviation:**")
            st.latex(r"\sigma_p = \sqrt{\sigma_p^2}")
            
            st.write("Where: w = weights, σ = standard deviation, ρ = correlation coefficient")
        
        # Two-asset portfolio example
        st.subheader("Two-Asset Portfolio Example")
        
        col1, col2 = st.columns(2)
        with col1:
            return1 = st.slider("Asset 1 Expected Return (%)", 0.0, 20.0, 10.0, 0.5)
            risk1 = st.slider("Asset 1 Risk (Std Dev) (%)", 1.0, 50.0, 15.0, 1.0)
            weight1 = st.slider("Asset 1 Weight (%)", 0.0, 100.0, 60.0, 5.0)
        with col2:
            return2 = st.slider("Asset 2 Expected Return (%)", 0.0, 20.0, 8.0, 0.5)
            risk2 = st.slider("Asset 2 Risk (Std Dev) (%)", 1.0, 50.0, 20.0, 1.0)
            correlation = st.slider("Correlation", -1.0, 1.0, 0.3, 0.1)
        
        weight2 = 100.0 - weight1
        
        # Calculate portfolio metrics
        portfolio_return = (weight1/100) * return1 + (weight2/100) * return2
        portfolio_variance = ((weight1/100)**2 * (risk1/100)**2 + 
                            (weight2/100)**2 * (risk2/100)**2 + 
                            2 * (weight1/100) * (weight2/100) * (risk1/100) * (risk2/100) * correlation)
        portfolio_risk = np.sqrt(portfolio_variance) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Return", f"{portfolio_return:.2f}%")
        with col2:
            st.metric("Portfolio Risk", f"{portfolio_risk:.2f}%")
        with col3:
            sharpe_ratio = (portfolio_return - 3.0) / portfolio_risk  # Assuming 3% risk-free rate
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.3f}")
        
        # Show diversification benefit
        weighted_avg_risk = (weight1/100) * risk1 + (weight2/100) * risk2
        diversification_benefit = weighted_avg_risk - portfolio_risk
        
        if diversification_benefit > 0:
            st.success(f"✅ Diversification Benefit: {diversification_benefit:.2f}% risk reduction")
        else:
            st.warning("⚠️ No diversification benefit (correlation too high)")
    
    with tab2:
        st.subheader("Efficient Frontier")
        
        st.write("""
        The efficient frontier represents the set of portfolios that offer the highest expected return 
        for each level of risk, or the lowest risk for each level of expected return.
        """)
        
        # Generate efficient frontier
        if st.button("Generate Efficient Frontier"):
            # Sample asset data
            assets = ['Stock A', 'Stock B', 'Stock C', 'Bond']
            expected_returns = np.array([0.12, 0.10, 0.08, 0.04])
            
            # Covariance matrix (annual)
            cov_matrix = np.array([
                [0.0225, 0.0108, 0.0063, 0.0015],
                [0.0108, 0.0196, 0.0048, 0.0012],
                [0.0063, 0.0048, 0.0144, 0.0009],
                [0.0015, 0.0012, 0.0009, 0.0025]
            ])
            
            def portfolio_performance(weights, expected_returns, cov_matrix):
                portfolio_return = np.sum(expected_returns * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return portfolio_return, portfolio_std
            
            def minimize_risk(target_return, expected_returns, cov_matrix):
                n_assets = len(expected_returns)
                
                def objective(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                    {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - target_return}  # target return
                ]
                
                bounds = tuple((0, 1) for _ in range(n_assets))
                initial_guess = np.array([1/n_assets] * n_assets)
                
                result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
                return result
            
            # Generate efficient frontier points
            target_returns = np.linspace(0.04, 0.12, 50)
            efficient_portfolios = []
            
            for target in target_returns:
                try:
                    result = minimize_risk(target, expected_returns, cov_matrix)
                    if result.success:
                        weights = result.x
                        ret, risk = portfolio_performance(weights, expected_returns, cov_matrix)
                        efficient_portfolios.append([risk * 100, ret * 100])
                except:
                    continue
            
            if efficient_portfolios:
                efficient_df = pd.DataFrame(efficient_portfolios, columns=['Risk (%)', 'Return (%)'])
                
                # Plot efficient frontier
                fig = px.scatter(efficient_df, x='Risk (%)', y='Return (%)', 
                               title='Efficient Frontier')
                
                # Add individual assets
                individual_risks = [np.sqrt(cov_matrix[i, i]) * 100 for i in range(len(assets))]
                individual_returns = expected_returns * 100
                
                fig.add_trace(go.Scatter(x=individual_risks, y=individual_returns, 
                                       mode='markers+text', text=assets, textposition="top center",
                                       marker=dict(size=10, color='red'), name='Individual Assets'))
                
                # Add Capital Market Line (CML)
                risk_free_rate = 4.0
                # Find market portfolio (max Sharpe ratio)
                sharpe_ratios = [(ret - risk_free_rate) / risk for risk, ret in efficient_portfolios if risk > 0]
                if sharpe_ratios:
                    max_sharpe_idx = np.argmax(sharpe_ratios)
                    market_risk = efficient_portfolios[max_sharpe_idx][0]
                    market_return = efficient_portfolios[max_sharpe_idx][1]
                    
                    # CML line
                    cml_x = np.linspace(0, max(efficient_df['Risk (%)']) * 1.2, 100)
                    cml_y = risk_free_rate + (market_return - risk_free_rate) * cml_x / market_risk
                    
                    fig.add_trace(go.Scatter(x=cml_x, y=cml_y, mode='lines', 
                                           name='Capital Market Line', line=dict(dash='dash')))
                    
                    # Mark market portfolio
                    fig.add_trace(go.Scatter(x=[market_risk], y=[market_return], 
                                           mode='markers', marker=dict(size=15, color='gold'),
                                           name='Market Portfolio'))
                
                fig.update_layout(xaxis_title="Risk (Standard Deviation %)", 
                                yaxis_title="Expected Return (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display asset information
                st.subheader("Asset Information")
                asset_df = pd.DataFrame({
                    'Asset': assets,
                    'Expected Return (%)': expected_returns * 100,
                    'Risk (%)': [np.sqrt(cov_matrix[i, i]) * 100 for i in range(len(assets))]
                })
                st.dataframe(asset_df, use_container_width=True)
                
                # Show correlation matrix
                st.subheader("Correlation Matrix")
                std_devs = np.sqrt(np.diag(cov_matrix))
                corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
                
                corr_df = pd.DataFrame(corr_matrix, index=assets, columns=assets)
                st.dataframe(corr_df.round(3), use_container_width=True)
    
    with tab3:
        st.subheader("Capital Asset Pricing Model (CAPM)")
        
        st.write("""
        CAPM describes the relationship between systematic risk and expected return for assets, 
        particularly stocks. It's used to estimate the cost of equity and evaluate investment performance.
        """)
        
        with st.expander("📊 CAPM Formula"):
            st.latex(r"E(R_i) = R_f + \beta_i [E(R_m) - R_f]")
            st.write("Where:")
            st.write("- E(Rᵢ) = Expected return of asset i")
            st.write("- Rₓ = Risk-free rate")
            st.write("- βᵢ = Beta of asset i")
            st.write("- E(Rₘ) = Expected market return")
            st.write("- [E(Rₘ) - Rₓ] = Market risk premium")
        
        st.subheader("CAPM Calculator")
        
        col1, col2 = st.columns(2)
        with col1:
            risk_free_rate = st.number_input("Risk-Free Rate (%)", value=3.0, min_value=0.0, max_value=20.0)
            market_return = st.number_input("Expected Market Return (%)", value=10.0, min_value=0.0, max_value=30.0)
        with col2:
            asset_beta = st.number_input("Asset Beta", value=1.2, min_value=-3.0, max_value=5.0)
        
        if market_return > risk_free_rate:
            market_premium = market_return - risk_free_rate
            required_return = risk_free_rate + asset_beta * market_premium
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Risk Premium", f"{market_premium:.2f}%")
            with col2:
                st.metric("Required Return", f"{required_return:.2f}%")
            with col3:
                risk_premium = asset_beta * market_premium
                st.metric("Asset Risk Premium", f"{risk_premium:.2f}%")
        
        # Security Market Line
        st.subheader("Security Market Line (SML)")
        
        if st.button("Plot Security Market Line"):
            beta_range = np.linspace(-0.5, 3.0, 100)
            sml_returns = risk_free_rate + beta_range * (market_return - risk_free_rate)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=beta_range, y=sml_returns, mode='lines', 
                                   name='Security Market Line'))
            
            # Add sample securities
            sample_securities = {
                'Treasury Bills': (0.0, risk_free_rate),
                'Market Portfolio': (1.0, market_return),
                'Your Asset': (asset_beta, required_return),
                'Defensive Stock': (0.6, risk_free_rate + 0.6 * market_premium),
                'Growth Stock': (1.8, risk_free_rate + 1.8 * market_premium)
            }
            
            for security, (beta, ret) in sample_securities.items():
                color = 'red' if security == 'Your Asset' else 'blue'
                fig.add_trace(go.Scatter(x=[beta], y=[ret], mode='markers+text', 
                                       text=[security], textposition="top center",
                                       marker=dict(size=10, color=color), name=security,
                                       showlegend=False))
            
            fig.update_layout(title="Security Market Line", 
                            xaxis_title="Beta", yaxis_title="Expected Return (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Beta calculation example
        st.subheader("Beta Calculation")
        
        st.write("Beta measures systematic risk - how much an asset's return moves with the market.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Beta Interpretation:**")
            st.write("- β = 1.0: Moves with market")
            st.write("- β > 1.0: More volatile than market")
            st.write("- β < 1.0: Less volatile than market")
            st.write("- β < 0: Moves opposite to market")
        
        with col2:
            # Beta calculation from correlation
            st.write("**Beta Formula:**")
            st.latex(r"\beta_i = \rho_{i,m} \times \frac{\sigma_i}{\sigma_m}")
            
            correlation = st.slider("Correlation with Market", -1.0, 1.0, 0.75, 0.05)
            asset_volatility = st.slider("Asset Volatility (%)", 1.0, 50.0, 20.0, 1.0)
            market_volatility = st.slider("Market Volatility (%)", 1.0, 50.0, 15.0, 1.0)
            
            calculated_beta = correlation * (asset_volatility / market_volatility)
            st.metric("Calculated Beta", f"{calculated_beta:.3f}")
    
    with tab4:
        st.subheader("Portfolio Optimization")
        
        st.write("""
        Portfolio optimization involves finding the optimal weights for assets to achieve 
        specific objectives such as maximizing return, minimizing risk, or maximizing the Sharpe ratio.
        """)
        
        optimization_type = st.selectbox("Select Optimization Objective:", 
                                       ["Maximize Sharpe Ratio", "Minimize Risk", "Maximize Return"])
        
        # Sample portfolio data
        st.subheader("Portfolio Assets")
        
        if 'portfolio_assets' not in st.session_state:
            st.session_state.portfolio_assets = pd.DataFrame({
                'Asset': ['US Stocks', 'International Stocks', 'Bonds', 'REITs'],
                'Expected Return (%)': [10.0, 8.5, 4.0, 7.0],
                'Risk (%)': [16.0, 18.0, 4.0, 20.0]
            })
        
        # Allow editing of asset data
        edited_assets = st.data_editor(
            st.session_state.portfolio_assets,
            use_container_width=True,
            num_rows="dynamic"
        )
        
        if len(edited_assets) >= 2:
            # Create correlation matrix input
            st.subheader("Correlation Matrix")
            n_assets = len(edited_assets)
            
            # Initialize correlation matrix
            if 'corr_matrix' not in st.session_state or len(st.session_state.corr_matrix) != n_assets:
                st.session_state.corr_matrix = np.eye(n_assets)
                # Add some realistic correlations
                if n_assets >= 2:
                    st.session_state.corr_matrix[0, 1] = st.session_state.corr_matrix[1, 0] = 0.7  # US-Intl stocks
                if n_assets >= 3:
                    st.session_state.corr_matrix[0, 2] = st.session_state.corr_matrix[2, 0] = -0.1  # Stocks-Bonds
                    st.session_state.corr_matrix[1, 2] = st.session_state.corr_matrix[2, 1] = -0.2
                if n_assets >= 4:
                    st.session_state.corr_matrix[0, 3] = st.session_state.corr_matrix[3, 0] = 0.6  # Stocks-REITs
            
            # Display editable correlation matrix
            corr_df = pd.DataFrame(st.session_state.corr_matrix, 
                                 index=edited_assets['Asset'], 
                                 columns=edited_assets['Asset'])
            
            # Show upper triangular part for editing
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{edited_assets.iloc[i]['Asset']} - {edited_assets.iloc[j]['Asset']}")
                    with col2:
                        corr_value = st.number_input("", 
                                                   min_value=-1.0, max_value=1.0, 
                                                   value=float(st.session_state.corr_matrix[i, j]),
                                                   step=0.05, format="%.2f",
                                                   key=f"corr_{i}_{j}")
                        st.session_state.corr_matrix[i, j] = corr_value
                        st.session_state.corr_matrix[j, i] = corr_value
            
            if st.button("Optimize Portfolio"):
                # Prepare data
                returns = edited_assets['Expected Return (%)'].values / 100
                risks = edited_assets['Risk (%)'].values / 100
                
                # Create covariance matrix
                cov_matrix = np.outer(risks, risks) * st.session_state.corr_matrix
                
                # Risk-free rate for Sharpe ratio
                rf_rate = st.number_input("Risk-Free Rate (%)", value=3.0, min_value=0.0) / 100
                
                def portfolio_performance(weights):
                    portfolio_return = np.sum(returns * weights)
                    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe_ratio = (portfolio_return - rf_rate) / portfolio_risk
                    return portfolio_return, portfolio_risk, sharpe_ratio
                
                # Optimization constraints
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # weights sum to 1
                bounds = tuple((0, 1) for _ in range(n_assets))  # long-only
                initial_guess = np.array([1/n_assets] * n_assets)
                
                if optimization_type == "Maximize Sharpe Ratio":
                    def objective(weights):
                        _, _, sharpe = portfolio_performance(weights)
                        return -sharpe  # negative because we minimize
                    
                elif optimization_type == "Minimize Risk":
                    def objective(weights):
                        _, risk, _ = portfolio_performance(weights)
                        return risk
                    
                else:  # Maximize Return
                    def objective(weights):
                        ret, _, _ = portfolio_performance(weights)
                        return -ret  # negative because we minimize
                
                # Solve optimization
                result = minimize(objective, initial_guess, method='SLSQP', 
                                bounds=bounds, constraints=constraints)
                
                if result.success:
                    optimal_weights = result.x
                    opt_return, opt_risk, opt_sharpe = portfolio_performance(optimal_weights)
                    
                    st.success("✅ Optimization Successful!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Portfolio Return", f"{opt_return*100:.2f}%")
                    with col2:
                        st.metric("Portfolio Risk", f"{opt_risk*100:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{opt_sharpe:.3f}")
                    
                    # Show optimal weights
                    st.subheader("Optimal Asset Allocation")
                    
                    weights_df = pd.DataFrame({
                        'Asset': edited_assets['Asset'],
                        'Weight (%)': optimal_weights * 100
                    })
                    st.dataframe(weights_df, use_container_width=True)
                    
                    # Pie chart of allocation
                    fig = px.pie(weights_df, values='Weight (%)', names='Asset', 
                               title='Optimal Portfolio Allocation')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Compare with equal-weight portfolio
                    equal_weights = np.array([1/n_assets] * n_assets)
                    eq_return, eq_risk, eq_sharpe = portfolio_performance(equal_weights)
                    
                    st.subheader("Comparison with Equal-Weight Portfolio")
                    
                    comparison_df = pd.DataFrame({
                        'Portfolio': ['Optimized', 'Equal-Weight'],
                        'Return (%)': [opt_return*100, eq_return*100],
                        'Risk (%)': [opt_risk*100, eq_risk*100],
                        'Sharpe Ratio': [opt_sharpe, eq_sharpe]
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                else:
                    st.error("❌ Optimization failed. Please check your inputs.")
        
        # Assessment
        st.subheader("Portfolio Theory Assessment")
        
        q1 = st.radio(
            "What does the efficient frontier represent?",
            ["All possible portfolios", "Optimal portfolios", "High-risk portfolios", "Low-risk portfolios"],
            key="port_q1"
        )
        
        q2 = st.radio(
            "According to CAPM, what determines expected return?",
            ["Total risk", "Systematic risk (beta)", "Unsystematic risk", "Correlation"],
            key="port_q2"
        )
        
        q3 = st.radio(
            "Diversification reduces which type of risk?",
            ["Systematic risk", "Market risk", "Unsystematic risk", "Interest rate risk"],
            key="port_q3"
        )
        
        if st.button("Submit Assessment"):
            score = 0
            
            if q1 == "Optimal portfolios":
                st.success("Q1: Correct! ✅")
                score += 1
            else:
                st.error("Q1: Incorrect. The efficient frontier shows optimal risk-return combinations.")
            
            if q2 == "Systematic risk (beta)":
                st.success("Q2: Correct! ✅")
                score += 1
            else:
                st.error("Q2: Incorrect. CAPM says expected return depends on systematic risk (beta).")
            
            if q3 == "Unsystematic risk":
                st.success("Q3: Correct! ✅")
                score += 1
            else:
                st.error("Q3: Incorrect. Diversification reduces unsystematic (specific) risk.")
            
            st.write(f"Your score: {score}/3")
            
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Class 7: Portfolio Theory")
                progress_tracker.set_class_score("Class 7: Portfolio Theory", (score/3) * 100)
                st.success("🎉 Excellent! You understand portfolio theory!")
    
    # Download materials
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Portfolio Resources")
    
    portfolio_code = """
# Portfolio Theory and Optimization

import numpy as np
from scipy.optimize import minimize

def portfolio_performance(weights, returns, cov_matrix):
    '''Calculate portfolio return and risk'''
    portfolio_return = np.sum(returns * weights)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    return portfolio_return, portfolio_std

def optimize_sharpe_ratio(returns, cov_matrix, risk_free_rate):
    '''Optimize portfolio for maximum Sharpe ratio'''
    n_assets = len(returns)
    
    def objective(weights):
        port_return, port_std = portfolio_performance(weights, returns, cov_matrix)
        sharpe_ratio = (port_return - risk_free_rate) / port_std
        return -sharpe_ratio  # Minimize negative Sharpe ratio
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    return result.x if result.success else None

def calculate_capm_return(risk_free_rate, beta, market_return):
    '''Calculate CAPM expected return'''
    return risk_free_rate + beta * (market_return - risk_free_rate)

# Example usage
returns = np.array([0.10, 0.08, 0.04])  # Expected returns
cov_matrix = np.array([[0.0225, 0.0108, 0.0015],
                       [0.0108, 0.0196, 0.0012],
                       [0.0015, 0.0012, 0.0025]])  # Covariance matrix

optimal_weights = optimize_sharpe_ratio(returns, cov_matrix, 0.03)
"""
    
    st.sidebar.download_button(
        label="💻 Download Portfolio Code",
        data=portfolio_code,
        file_name="portfolio_theory.py",
        mime="text/python"
    )
