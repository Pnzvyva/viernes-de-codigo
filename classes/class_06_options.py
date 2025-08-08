import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 6: Valoración de Opciones")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Fundamentos de Opciones", "Modelo Black-Scholes", "Griegas de Opciones", "Estrategias"])
    
    with tab1:
        st.subheader("Understanding Options")
        
        st.write("""
        Options are financial derivatives that give the holder the right, but not the obligation, 
        to buy (call) or sell (put) an underlying asset at a specified price (strike) before or on expiration.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Call Options:**
            - Right to BUY at strike price
            - Profit when S > K
            - Limited loss (premium)
            - Unlimited profit potential
            - Bullish strategy
            """)
        
        with col2:
            st.write("""
            **Put Options:**
            - Right to SELL at strike price
            - Profit when S < K
            - Limited loss (premium)
            - Limited profit (K - premium)
            - Bearish strategy
            """)
        
        st.subheader("Option Payoff Diagrams")
        
        option_type = st.selectbox("Select Option Type:", ["Call Option", "Put Option"])
        
        col1, col2 = st.columns(2)
        with col1:
            strike_price = st.slider("Strike Price ($)", 80, 120, 100)
            premium = st.slider("Option Premium ($)", 1, 20, 5)
        with col2:
            position = st.radio("Position:", ["Long (Buy)", "Short (Sell)"])
        
        # Generate payoff diagram
        spot_range = np.linspace(60, 140, 100)
        
        if option_type == "Call Option":
            if position == "Long (Buy)":
                payoffs = np.maximum(spot_range - strike_price, 0) - premium
                title = f"Long Call (Strike: ${strike_price}, Premium: ${premium})"
            else:
                payoffs = premium - np.maximum(spot_range - strike_price, 0)
                title = f"Short Call (Strike: ${strike_price}, Premium: ${premium})"
        else:  # Put Option
            if position == "Long (Buy)":
                payoffs = np.maximum(strike_price - spot_range, 0) - premium
                title = f"Long Put (Strike: ${strike_price}, Premium: ${premium})"
            else:
                payoffs = premium - np.maximum(strike_price - spot_range, 0)
                title = f"Short Put (Strike: ${strike_price}, Premium: ${premium})"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=payoffs, mode='lines', name='Payoff'))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_vline(x=strike_price, line_dash="dot", annotation_text="Strike Price")
        
        # Highlight profit/loss regions
        profit_mask = payoffs > 0
        loss_mask = payoffs < 0
        
        fig.add_trace(go.Scatter(x=spot_range[profit_mask], y=payoffs[profit_mask], 
                                fill='tonexty', fillcolor='rgba(0,255,0,0.2)', 
                                mode='none', name='Profit', showlegend=False))
        fig.add_trace(go.Scatter(x=spot_range[loss_mask], y=payoffs[loss_mask], 
                                fill='tonexty', fillcolor='rgba(255,0,0,0.2)', 
                                mode='none', name='Loss', showlegend=False))
        
        fig.update_layout(title=title, xaxis_title="Stock Price at Expiration ($)", 
                         yaxis_title="Profit/Loss ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        breakeven = strike_price + premium if "Call" in option_type else strike_price - premium
        if position == "Short (Sell)":
            breakeven = strike_price - premium if "Call" in option_type else strike_price + premium
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Strike Price", f"${strike_price}")
        with col2:
            st.metric("Premium", f"${premium}")
        with col3:
            st.metric("Breakeven", f"${breakeven:.2f}")
    
    with tab2:
        st.subheader("Black-Scholes Option Pricing Model")
        
        st.write("""
        The Black-Scholes model provides a mathematical framework for pricing European options. 
        It assumes constant volatility, risk-free rate, and no dividends.
        """)
        
        with st.expander("📊 Black-Scholes Formula"):
            st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)")
            st.latex(r"P = K e^{-rT} N(-d_2) - S_0 N(-d_1)")
            st.write("Where:")
            st.latex(r"d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}")
            st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")
            st.write("C = Call price, P = Put price, N() = Cumulative normal distribution")
        
        st.subheader("Black-Scholes Calculator")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            S = st.number_input("Current Stock Price (S₀) ($)", value=100.0, min_value=0.01)
            K = st.number_input("Strike Price (K) ($)", value=100.0, min_value=0.01)
        with col2:
            T = st.number_input("Time to Expiration (T) years", value=0.25, min_value=0.001, max_value=10.0)
            r = st.number_input("Risk-Free Rate (r) %", value=5.0, min_value=0.0, max_value=50.0) / 100
        with col3:
            sigma = st.number_input("Volatility (σ) %", value=20.0, min_value=0.1, max_value=200.0) / 100
        
        if st.button("Calculate Option Prices"):
            # Calculate Black-Scholes prices
            call_price = FinancialCalculations.black_scholes_call(S, K, T, r, sigma)
            put_price = FinancialCalculations.black_scholes_put(S, K, T, r, sigma)
            
            # Calculate d1 and d2 for analysis
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"Call Option Price: ${call_price:.2f}")
                st.info(f"N(d₁) = {norm.cdf(d1):.4f}")
                st.info(f"N(d₂) = {norm.cdf(d2):.4f}")
            with col2:
                st.success(f"Put Option Price: ${put_price:.2f}")
                st.info(f"d₁ = {d1:.4f}")
                st.info(f"d₂ = {d2:.4f}")
            
            # Put-Call Parity Check
            parity_check = call_price - put_price - (S - K * np.exp(-r * T))
            st.write(f"**Put-Call Parity Check:** {parity_check:.6f} (should be ≈ 0)")
            
            # Sensitivity Analysis
            st.subheader("Sensitivity Analysis")
            
            analysis_param = st.selectbox("Select Parameter for Analysis:", 
                                        ["Stock Price", "Volatility", "Time to Expiration"])
            
            if analysis_param == "Stock Price":
                spot_range = np.linspace(S * 0.7, S * 1.3, 50)
                call_prices = [FinancialCalculations.black_scholes_call(s, K, T, r, sigma) for s in spot_range]
                put_prices = [FinancialCalculations.black_scholes_put(s, K, T, r, sigma) for s in spot_range]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spot_range, y=call_prices, mode='lines', name='Call Price'))
                fig.add_trace(go.Scatter(x=spot_range, y=put_prices, mode='lines', name='Put Price'))
                fig.add_vline(x=S, line_dash="dot", annotation_text="Current Price")
                fig.add_vline(x=K, line_dash="dash", annotation_text="Strike Price")
                fig.update_layout(title="Option Prices vs Stock Price", 
                                xaxis_title="Stock Price ($)", yaxis_title="Option Price ($)")
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_param == "Volatility":
                vol_range = np.linspace(0.05, 0.8, 50)
                call_prices = [FinancialCalculations.black_scholes_call(S, K, T, r, vol) for vol in vol_range]
                put_prices = [FinancialCalculations.black_scholes_put(S, K, T, r, vol) for vol in vol_range]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vol_range*100, y=call_prices, mode='lines', name='Call Price'))
                fig.add_trace(go.Scatter(x=vol_range*100, y=put_prices, mode='lines', name='Put Price'))
                fig.add_vline(x=sigma*100, line_dash="dot", annotation_text="Current Volatility")
                fig.update_layout(title="Option Prices vs Volatility", 
                                xaxis_title="Volatility (%)", yaxis_title="Option Price ($)")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Option Greeks")
        
        st.write("""
        The Greeks measure the sensitivity of option prices to changes in various parameters. 
        They are essential for risk management and hedging strategies.
        """)
        
        def calculate_greeks(S, K, T, r, sigma, option_type='call'):
            """Calculate option Greeks"""
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type == 'call':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta (per day)
            if option_type == 'call':
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                        r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                        r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega (per 1% change in volatility)
            vega = S * np.sqrt(T) * norm.pdf(d1) / 100
            
            # Rho (per 1% change in interest rate)
            if option_type == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}
        
        # Greeks Calculator
        st.subheader("Greeks Calculator")
        
        # Use same parameters from previous tab or allow new input
        if 'S' not in locals():
            col1, col2, col3 = st.columns(3)
            with col1:
                S_greeks = st.number_input("Stock Price ($)", value=100.0, min_value=0.01, key="greeks_S")
                K_greeks = st.number_input("Strike Price ($)", value=100.0, min_value=0.01, key="greeks_K")
            with col2:
                T_greeks = st.number_input("Time to Expiration (years)", value=0.25, min_value=0.001, key="greeks_T")
                r_greeks = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, key="greeks_r") / 100
            with col3:
                sigma_greeks = st.number_input("Volatility (%)", value=20.0, min_value=0.1, key="greeks_vol") / 100
        else:
            S_greeks, K_greeks, T_greeks, r_greeks, sigma_greeks = S, K, T, r, sigma
        
        option_type_greek = st.radio("Option Type:", ["Call", "Put"], key="greek_type")
        
        if st.button("Calculate Greeks"):
            greeks_call = calculate_greeks(S_greeks, K_greeks, T_greeks, r_greeks, sigma_greeks, 'call')
            greeks_put = calculate_greeks(S_greeks, K_greeks, T_greeks, r_greeks, sigma_greeks, 'put')
            
            # Display Greeks
            st.subheader("Option Greeks")
            
            greeks_data = {
                'Greek': ['Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ)', 'Vega (ν)', 'Rho (ρ)'],
                'Call Option': [f"{greeks_call['Delta']:.4f}", f"{greeks_call['Gamma']:.4f}", 
                               f"{greeks_call['Theta']:.4f}", f"{greeks_call['Vega']:.4f}", f"{greeks_call['Rho']:.4f}"],
                'Put Option': [f"{greeks_put['Delta']:.4f}", f"{greeks_put['Gamma']:.4f}", 
                              f"{greeks_put['Theta']:.4f}", f"{greeks_put['Vega']:.4f}", f"{greeks_put['Rho']:.4f}"],
                'Interpretation': [
                    'Price change per $1 stock move',
                    'Delta change per $1 stock move', 
                    'Price change per day (time decay)',
                    'Price change per 1% volatility change',
                    'Price change per 1% rate change'
                ]
            }
            
            df_greeks = pd.DataFrame(greeks_data)
            st.dataframe(df_greeks, use_container_width=True)
            
            # Greeks visualization
            st.subheader("Greeks Visualization")
            
            spot_range = np.linspace(S_greeks * 0.7, S_greeks * 1.3, 50)
            
            selected_greek = option_type_greek.lower()
            greek_values = []
            
            for spot in spot_range:
                greeks = calculate_greeks(spot, K_greeks, T_greeks, r_greeks, sigma_greeks, selected_greek)
                greek_values.append(greeks)
            
            # Plot selected Greeks
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spot_range, y=[g['Delta'] for g in greek_values], 
                                   mode='lines', name='Delta'))
            fig.add_trace(go.Scatter(x=spot_range, y=[g['Gamma']*10 for g in greek_values], 
                                   mode='lines', name='Gamma (×10)'))
            fig.add_vline(x=S_greeks, line_dash="dot", annotation_text="Current Price")
            fig.add_vline(x=K_greeks, line_dash="dash", annotation_text="Strike Price")
            
            fig.update_layout(title=f"{option_type_greek} Option Greeks vs Stock Price", 
                            xaxis_title="Stock Price ($)", yaxis_title="Greek Value")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Option Trading Strategies")
        
        st.write("""
        Option strategies combine multiple options to create specific risk-reward profiles. 
        Here we'll explore some common multi-leg strategies.
        """)
        
        strategy = st.selectbox("Select Strategy:", 
                               ["Covered Call", "Protective Put", "Bull Call Spread", "Iron Condor"])
        
        if strategy == "Covered Call":
            st.write("**Covered Call Strategy**")
            st.write("Long stock + Short call option. Generates income but caps upside.")
            
            col1, col2 = st.columns(2)
            with col1:
                stock_price = st.number_input("Stock Price ($)", value=100.0, key="cc_stock")
                shares = st.number_input("Number of Shares", value=100, min_value=1, key="cc_shares")
            with col2:
                call_strike = st.number_input("Call Strike ($)", value=105.0, key="cc_strike")
                call_premium = st.number_input("Call Premium ($)", value=3.0, key="cc_premium")
            
            # Calculate payoffs
            spot_range = np.linspace(stock_price * 0.8, stock_price * 1.3, 100)
            stock_payoffs = (spot_range - stock_price) * shares
            call_payoffs = (call_premium - np.maximum(spot_range - call_strike, 0)) * shares
            total_payoffs = stock_payoffs + call_payoffs
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spot_range, y=stock_payoffs, mode='lines', name='Stock Position'))
            fig.add_trace(go.Scatter(x=spot_range, y=call_payoffs, mode='lines', name='Short Call'))
            fig.add_trace(go.Scatter(x=spot_range, y=total_payoffs, mode='lines', name='Combined Position', 
                                   line=dict(width=3)))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            
            fig.update_layout(title="Covered Call Strategy Payoff", 
                            xaxis_title="Stock Price at Expiration ($)", yaxis_title="Profit/Loss ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
            max_profit = call_premium * shares + max(0, call_strike - stock_price) * shares
            breakeven = stock_price - call_premium
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Maximum Profit", f"${max_profit:.2f}")
            with col2:
                st.metric("Maximum Loss", "Unlimited")
            with col3:
                st.metric("Breakeven", f"${breakeven:.2f}")
        
        elif strategy == "Bull Call Spread":
            st.write("**Bull Call Spread Strategy**")
            st.write("Long lower strike call + Short higher strike call. Limited profit and loss.")
            
            col1, col2 = st.columns(2)
            with col1:
                long_strike = st.number_input("Long Call Strike ($)", value=95.0, key="bcs_long")
                long_premium = st.number_input("Long Call Premium ($)", value=6.0, key="bcs_long_prem")
            with col2:
                short_strike = st.number_input("Short Call Strike ($)", value=105.0, key="bcs_short")
                short_premium = st.number_input("Short Call Premium ($)", value=2.0, key="bcs_short_prem")
            
            net_debit = long_premium - short_premium
            st.info(f"Net Debit: ${net_debit:.2f}")
            
            # Calculate payoffs
            spot_range = np.linspace(80, 120, 100)
            long_call_payoffs = np.maximum(spot_range - long_strike, 0) - long_premium
            short_call_payoffs = short_premium - np.maximum(spot_range - short_strike, 0)
            spread_payoffs = long_call_payoffs + short_call_payoffs
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spot_range, y=long_call_payoffs, mode='lines', name='Long Call'))
            fig.add_trace(go.Scatter(x=spot_range, y=short_call_payoffs, mode='lines', name='Short Call'))
            fig.add_trace(go.Scatter(x=spot_range, y=spread_payoffs, mode='lines', name='Bull Call Spread', 
                                   line=dict(width=3)))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            
            fig.update_layout(title="Bull Call Spread Payoff", 
                            xaxis_title="Stock Price at Expiration ($)", yaxis_title="Profit/Loss ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Strategy metrics
            max_profit = short_strike - long_strike - net_debit
            max_loss = net_debit
            breakeven = long_strike + net_debit
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Maximum Profit", f"${max_profit:.2f}")
            with col2:
                st.metric("Maximum Loss", f"${max_loss:.2f}")
            with col3:
                st.metric("Breakeven", f"${breakeven:.2f}")
        
        # Practice problems
        st.subheader("Options Practice Problems")
        
        problem = st.selectbox("Select Problem:", ["Problem 1: Black-Scholes", "Problem 2: Put-Call Parity"])
        
        if problem == "Problem 1: Black-Scholes":
            st.write("""
            **Problem 1:**
            Calculate the Black-Scholes price of a call option with:
            - Stock price: $50
            - Strike price: $52
            - Time to expiration: 3 months (0.25 years)
            - Risk-free rate: 4%
            - Volatility: 25%
            """)
            
            user_answer = st.number_input("Your answer ($):", min_value=0.0, format="%.2f", key="bs_p1")
            
            if st.button("Check Answer", key="check_bs_p1"):
                correct_answer = FinancialCalculations.black_scholes_call(50, 52, 0.25, 0.04, 0.25)
                if abs(user_answer - correct_answer) < 0.10:
                    st.success(f"✅ Correct! The answer is ${correct_answer:.2f}")
                else:
                    st.error(f"❌ Incorrect. The correct answer is ${correct_answer:.2f}")
        
        # Final assessment
        st.subheader("Class Assessment")
        
        q1 = st.radio(
            "Which Greek measures time decay?",
            ["Delta", "Gamma", "Theta", "Vega"],
            key="opt_q1"
        )
        
        q2 = st.radio(
            "Put-call parity states that C - P equals:",
            ["S - K", "S - Ke^(-rT)", "K - S", "Ke^(-rT) - S"],
            key="opt_q2"
        )
        
        q3 = st.radio(
            "Which strategy benefits most from high volatility?",
            ["Covered call", "Long straddle", "Short straddle", "Bull spread"],
            key="opt_q3"
        )
        
        if st.button("Submit Assessment"):
            score = 0
            
            if q1 == "Theta":
                st.success("Q1: Correct! ✅")
                score += 1
            else:
                st.error("Q1: Incorrect. Theta measures time decay (price change per day).")
            
            if q2 == "S - Ke^(-rT)":
                st.success("Q2: Correct! ✅")
                score += 1
            else:
                st.error("Q2: Incorrect. Put-call parity: C - P = S - Ke^(-rT)")
            
            if q3 == "Long straddle":
                st.success("Q3: Correct! ✅")
                score += 1
            else:
                st.error("Q3: Incorrect. Long straddle profits from high volatility in either direction.")
            
            st.write(f"Your score: {score}/3")
            
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Class 6: Options Pricing")
                progress_tracker.set_class_score("Class 6: Options Pricing", (score/3) * 100)
                st.success("🎉 Excellent! You've mastered options pricing!")
    
    # Download materials
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Options Resources")
    
    options_code = """
# Options Pricing and Greeks

import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    '''Black-Scholes call option price'''
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def option_delta(S, K, T, r, sigma, option_type='call'):
    '''Calculate option delta'''
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def option_gamma(S, K, T, r, sigma):
    '''Calculate option gamma'''
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Example usage
call_price = black_scholes_call(100, 100, 0.25, 0.05, 0.2)  # $5.91
delta = option_delta(100, 100, 0.25, 0.05, 0.2, 'call')     # 0.605
"""
    
    st.sidebar.download_button(
        label="💻 Download Options Code",
        data=options_code,
        file_name="options_pricing.py",
        mime="text/python"
    )
