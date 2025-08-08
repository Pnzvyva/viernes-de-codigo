import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 5: Introducción a Derivados")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Fundamentos de Derivados", "Forwards y Futuros", "Swaps", "Aplicaciones"])
    
    with tab1:
        st.subheader("Entendiendo los Derivados")
        
        st.write("""
        Los derivados son instrumentos financieros cuyo valor se deriva de un activo subyacente. 
        Se utilizan para cobertura de riesgos, especulación y oportunidades de arbitraje.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Main Types of Derivatives:**
            - **Forwards**: Customized OTC contracts
            - **Futures**: Standardized exchange-traded
            - **Options**: Rights but not obligations
            - **Swaps**: Exchange of cash flows
            """)
        
        with col2:
            st.write("""
            **Key Characteristics:**
            - Leverage effect
            - Low initial cost
            - Based on underlying assets
            - Can be used for hedging or speculation
            """)
        
        st.subheader("Derivative Markets")
        
        market_comparison = pd.DataFrame({
            'Feature': ['Standardization', 'Trading Venue', 'Counterparty Risk', 'Customization', 'Liquidity'],
            'Exchange-Traded': ['High', 'Organized Exchange', 'Low (Clearinghouse)', 'Limited', 'High'],
            'Over-the-Counter': ['Low', 'Bilateral', 'High', 'High', 'Variable']
        })
        
        st.dataframe(market_comparison, use_container_width=True)
        
        st.subheader("Uses of Derivatives")
        
        uses = {
            "Hedging": {
                "Description": "Reduce or eliminate risk exposure",
                "Example": "Airline hedging fuel price risk with oil futures",
                "Benefit": "Risk reduction and cash flow stability"
            },
            "Speculation": {
                "Description": "Profit from price movements with leverage",
                "Example": "Trader buying call options expecting stock price rise",
                "Benefit": "Amplified returns (with higher risk)"
            },
            "Arbitrage": {
                "Description": "Profit from price differences between markets",
                "Example": "Buying underpriced futures and selling overpriced stock",
                "Benefit": "Risk-free profits"
            }
        }
        
        for use, details in uses.items():
            with st.expander(f"🎯 {use}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Description:** {details['Description']}")
                    st.write(f"**Example:** {details['Example']}")
                with col2:
                    st.write(f"**Benefit:** {details['Benefit']}")
    
    with tab2:
        st.subheader("Forwards and Futures")
        
        st.write("""
        Forward and futures contracts are agreements to buy or sell an asset at a predetermined price 
        on a specific future date. The main difference is that futures are standardized and exchange-traded.
        """)
        
        derivative_type = st.selectbox("Select Contract Type:", ["Forward Contract", "Futures Contract"])
        
        if derivative_type == "Forward Contract":
            st.write("**Forward Contract Pricing**")
            
            st.latex(r"F_0 = S_0 \times e^{(r-q) \times T}")
            st.write("Where: F₀ = Forward price, S₀ = Spot price, r = risk-free rate, q = dividend yield, T = time to maturity")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                spot_price = st.number_input("Current Spot Price ($)", value=100.0, min_value=0.01)
                risk_free_rate = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0)
            with col2:
                dividend_yield = st.number_input("Dividend Yield (%)", value=2.0, min_value=0.0)
                time_to_maturity = st.number_input("Time to Maturity (years)", value=0.25, min_value=0.01, max_value=10.0)
            with col3:
                st.write("")  # Spacing
            
            if st.button("Calculate Forward Price"):
                forward_price = spot_price * np.exp((risk_free_rate/100 - dividend_yield/100) * time_to_maturity)
                st.success(f"Theoretical Forward Price: ${forward_price:.2f}")
                
                # Show calculation steps
                st.write("**Calculation Steps:**")
                st.write(f"F₀ = ${spot_price} × e^(({risk_free_rate/100:.3f} - {dividend_yield/100:.3f}) × {time_to_maturity})")
                st.write(f"F₀ = ${spot_price} × e^({(risk_free_rate/100 - dividend_yield/100) * time_to_maturity:.4f})")
                st.write(f"F₀ = ${spot_price} × {np.exp((risk_free_rate/100 - dividend_yield/100) * time_to_maturity):.4f}")
                st.write(f"F₀ = ${forward_price:.2f}")
                
                # Payoff diagram
                st.subheader("Forward Contract Payoffs")
                
                spot_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
                long_payoff = spot_range - forward_price
                short_payoff = forward_price - spot_range
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spot_range, y=long_payoff, mode='lines', name='Long Position'))
                fig.add_trace(go.Scatter(x=spot_range, y=short_payoff, mode='lines', name='Short Position'))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.add_vline(x=forward_price, line_dash="dash", line_color="red", annotation_text="Forward Price")
                
                fig.update_layout(
                    title="Forward Contract Payoff Diagram",
                    xaxis_title="Spot Price at Maturity ($)",
                    yaxis_title="Payoff ($)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif derivative_type == "Futures Contract":
            st.write("**Futures Contract Analysis**")
            
            st.write("""
            Futures contracts are similar to forwards but with key differences:
            - Daily mark-to-market settlement
            - Margin requirements
            - Standardized contract terms
            - Exchange guarantee
            """)
            
            # Margin calculation example
            st.subheader("Margin Requirements")
            
            col1, col2 = st.columns(2)
            with col1:
                contract_size = st.number_input("Contract Size", value=100, min_value=1)
                futures_price = st.number_input("Futures Price ($)", value=105.0, min_value=0.01)
                initial_margin = st.number_input("Initial Margin (%)", value=10.0, min_value=1.0, max_value=50.0)
            with col2:
                maintenance_margin = st.number_input("Maintenance Margin (%)", value=7.5, min_value=1.0, max_value=50.0)
                price_change = st.number_input("Price Change ($)", value=-2.0, min_value=-50.0, max_value=50.0)
            
            if st.button("Calculate Margin Call"):
                contract_value = contract_size * futures_price
                initial_margin_amount = contract_value * (initial_margin / 100)
                maintenance_margin_amount = contract_value * (maintenance_margin / 100)
                
                # Calculate new position value after price change
                new_price = futures_price + price_change
                new_contract_value = contract_size * new_price
                profit_loss = (new_contract_value - contract_value)  # For long position
                
                margin_account = initial_margin_amount + profit_loss
                
                st.subheader("Margin Account Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Contract Value", f"${contract_value:,.2f}")
                    st.metric("Initial Margin", f"${initial_margin_amount:,.2f}")
                    st.metric("Maintenance Margin", f"${maintenance_margin_amount:,.2f}")
                with col2:
                    st.metric("Price Change", f"${price_change:.2f}")
                    st.metric("P&L", f"${profit_loss:,.2f}")
                    st.metric("Margin Account Balance", f"${margin_account:,.2f}")
                
                if margin_account < maintenance_margin_amount:
                    margin_call = initial_margin_amount - margin_account
                    st.error(f"⚠️ MARGIN CALL: Deposit ${margin_call:,.2f} required!")
                else:
                    st.success("✅ No margin call required")
    
    with tab3:
        st.subheader("Interest Rate Swaps")
        
        st.write("""
        An interest rate swap is an agreement between two parties to exchange interest rate payments 
        on a notional principal amount. The most common type is a fixed-for-floating swap.
        """)
        
        st.subheader("Plain Vanilla Interest Rate Swap")
        
        col1, col2 = st.columns(2)
        with col1:
            notional_amount = st.number_input("Notional Amount ($)", value=1000000.0, min_value=1000.0, format="%.0f")
            fixed_rate = st.number_input("Fixed Rate (%)", value=4.5, min_value=0.0, max_value=20.0)
            swap_tenor = st.number_input("Swap Tenor (years)", value=5, min_value=1, max_value=30)
        with col2:
            floating_rate_current = st.number_input("Current Floating Rate (%)", value=3.8, min_value=0.0, max_value=20.0)
            payment_frequency = st.selectbox("Payment Frequency", ["Annual", "Semi-annual", "Quarterly"])
        
        # Payment frequency mapping
        freq_map = {"Annual": 1, "Semi-annual": 2, "Quarterly": 4}
        payments_per_year = freq_map[payment_frequency]
        
        if st.button("Analyze Swap"):
            # Calculate payment amounts
            fixed_payment = (notional_amount * fixed_rate / 100) / payments_per_year
            floating_payment = (notional_amount * floating_rate_current / 100) / payments_per_year
            net_payment = fixed_payment - floating_payment  # From fixed payer's perspective
            
            st.subheader("Swap Payment Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fixed Payment", f"${fixed_payment:,.2f}")
            with col2:
                st.metric("Floating Payment", f"${floating_payment:,.2f}")
            with col3:
                if net_payment > 0:
                    st.metric("Net Payment (Fixed Payer)", f"${net_payment:,.2f}", delta="Pay")
                else:
                    st.metric("Net Receipt (Fixed Payer)", f"${abs(net_payment):,.2f}", delta="Receive")
            
            # Simulate floating rate scenarios
            st.subheader("Interest Rate Sensitivity")
            
            rate_scenarios = np.linspace(1.0, 8.0, 50)
            net_payments = []
            
            for scenario_rate in rate_scenarios:
                scenario_floating_payment = (notional_amount * scenario_rate / 100) / payments_per_year
                scenario_net = fixed_payment - scenario_floating_payment
                net_payments.append(scenario_net)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rate_scenarios, y=net_payments, mode='lines', name='Net Payment'))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.add_vline(x=floating_rate_current, line_dash="dot", annotation_text="Current Rate")
            
            fig.update_layout(
                title="Swap Net Payment vs Floating Rate",
                xaxis_title="Floating Rate (%)",
                yaxis_title="Net Payment ($)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cash flow table
            st.subheader("Payment Schedule (First Year)")
            
            payment_schedule = []
            for quarter in range(1, payments_per_year + 1):
                payment_schedule.append({
                    'Period': quarter,
                    'Fixed Payment': fixed_payment,
                    'Floating Payment': floating_payment,
                    'Net Payment': net_payment
                })
            
            df_schedule = pd.DataFrame(payment_schedule)
            st.dataframe(df_schedule, use_container_width=True)
    
    with tab4:
        st.subheader("Derivative Applications and Risk Management")
        
        application_type = st.selectbox("Select Application:", 
                                      ["Portfolio Hedging", "Currency Hedging", "Interest Rate Risk Management"])
        
        if application_type == "Portfolio Hedging":
            st.write("**Hedging Stock Portfolio with Index Futures**")
            
            col1, col2 = st.columns(2)
            with col1:
                portfolio_value = st.number_input("Portfolio Value ($)", value=500000.0, min_value=1000.0)
                portfolio_beta = st.number_input("Portfolio Beta", value=1.2, min_value=0.1, max_value=3.0)
            with col2:
                index_futures_price = st.number_input("Index Futures Price", value=3500.0, min_value=100.0)
                futures_multiplier = st.number_input("Futures Multiplier", value=250.0, min_value=1.0)
            
            if st.button("Calculate Hedge Ratio"):
                # Calculate optimal hedge ratio
                hedge_ratio = (portfolio_value * portfolio_beta) / (index_futures_price * futures_multiplier)
                contracts_needed = round(hedge_ratio)
                
                st.success(f"Optimal Hedge Ratio: {hedge_ratio:.3f}")
                st.success(f"Futures Contracts Needed: {contracts_needed}")
                
                # Show hedging effectiveness
                st.subheader("Hedging Analysis")
                
                # Simulate market scenarios
                market_changes = np.linspace(-20, 20, 41)
                unhedged_pnl = []
                hedged_pnl = []
                
                for change in market_changes:
                    # Portfolio change
                    portfolio_change = portfolio_value * (change / 100) * portfolio_beta
                    
                    # Futures change (assuming futures move 1:1 with index)
                    futures_price_change = index_futures_price * (change / 100)
                    futures_pnl = -contracts_needed * futures_price_change * futures_multiplier  # Short position
                    
                    unhedged_pnl.append(portfolio_change)
                    hedged_pnl.append(portfolio_change + futures_pnl)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=market_changes, y=unhedged_pnl, mode='lines', name='Unhedged Portfolio'))
                fig.add_trace(go.Scatter(x=market_changes, y=hedged_pnl, mode='lines', name='Hedged Portfolio'))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                
                fig.update_layout(
                    title="Portfolio Hedging Effectiveness",
                    xaxis_title="Market Change (%)",
                    yaxis_title="Portfolio P&L ($)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif application_type == "Currency Hedging":
            st.write("**Currency Forward Hedging**")
            
            col1, col2 = st.columns(2)
            with col1:
                exposure_amount = st.number_input("Foreign Currency Exposure", value=1000000.0, min_value=1000.0)
                current_exchange_rate = st.number_input("Current Exchange Rate", value=1.20, min_value=0.01)
                domestic_rate = st.number_input("Domestic Interest Rate (%)", value=3.0, min_value=0.0)
            with col2:
                foreign_rate = st.number_input("Foreign Interest Rate (%)", value=1.5, min_value=0.0)
                hedge_horizon = st.number_input("Hedge Horizon (months)", value=6, min_value=1, max_value=60)
            
            if st.button("Calculate Currency Hedge"):
                # Calculate forward rate
                time_fraction = hedge_horizon / 12
                forward_rate = current_exchange_rate * np.exp((domestic_rate/100 - foreign_rate/100) * time_fraction)
                
                st.success(f"Forward Exchange Rate: {forward_rate:.4f}")
                
                # Show hedging outcomes for different exchange rate scenarios
                future_rates = np.linspace(current_exchange_rate * 0.8, current_exchange_rate * 1.2, 50)
                unhedged_values = [exposure_amount * rate for rate in future_rates]
                hedged_value = exposure_amount * forward_rate
                hedged_values = [hedged_value] * len(future_rates)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=future_rates, y=unhedged_values, mode='lines', name='Unhedged Position'))
                fig.add_trace(go.Scatter(x=future_rates, y=hedged_values, mode='lines', name='Hedged Position'))
                fig.add_vline(x=current_exchange_rate, line_dash="dot", annotation_text="Current Rate")
                fig.add_vline(x=forward_rate, line_dash="dash", annotation_text="Forward Rate")
                
                fig.update_layout(
                    title="Currency Hedging Comparison",
                    xaxis_title="Future Exchange Rate",
                    yaxis_title="Position Value (Domestic Currency)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Knowledge assessment
        st.subheader("Class Assessment")
        
        q1 = st.radio(
            "Which derivative has the highest counterparty risk?",
            ["Exchange-traded futures", "Forward contracts", "Listed options", "Standardized swaps"],
            key="deriv_q1"
        )
        
        q2 = st.radio(
            "In an interest rate swap, if floating rates rise, the fixed-rate payer:",
            ["Benefits", "Loses", "Is unaffected", "Needs more margin"],
            key="deriv_q2"
        )
        
        q3 = st.radio(
            "Forward prices are higher than spot prices when:",
            ["Interest rates > Dividend yield", "Interest rates < Dividend yield", 
             "The asset is volatile", "The contract expires soon"],
            key="deriv_q3"
        )
        
        if st.button("Submit Assessment"):
            score = 0
            
            if q1 == "Forward contracts":
                st.success("Q1: Correct! ✅")
                score += 1
            else:
                st.error("Q1: Incorrect. Forward contracts are OTC with high counterparty risk.")
            
            if q2 == "Benefits":
                st.success("Q2: Correct! ✅")
                score += 1
            else:
                st.error("Q2: Incorrect. Fixed-rate payer benefits when floating rates rise.")
            
            if q3 == "Interest rates > Dividend yield":
                st.success("Q3: Correct! ✅")
                score += 1
            else:
                st.error("Q3: Incorrect. Forward price increases with interest rates and decreases with dividends.")
            
            st.write(f"Your score: {score}/3")
            
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Class 5: Introduction to Derivatives")
                progress_tracker.set_class_score("Class 5: Introduction to Derivatives", (score/3) * 100)
                st.success("🎉 Excellent! You understand derivatives fundamentals!")
    
    # Download materials
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Derivatives Resources")
    
    derivatives_code = """
# Derivatives Pricing and Risk Management

import numpy as np

def forward_price(spot_price, risk_free_rate, dividend_yield, time_to_maturity):
    '''Calculate forward price'''
    return spot_price * np.exp((risk_free_rate - dividend_yield) * time_to_maturity)

def swap_fixed_rate(notional, tenor, floating_rates, discount_rates):
    '''Calculate fair fixed rate for interest rate swap'''
    pv_floating = sum([notional * rate * np.exp(-disc_rate * (i+1)) 
                      for i, (rate, disc_rate) in enumerate(zip(floating_rates, discount_rates))])
    
    annuity = sum([np.exp(-disc_rate * (i+1)) 
                   for i, disc_rate in enumerate(discount_rates)])
    
    return pv_floating / (notional * annuity)

def hedge_ratio(portfolio_value, portfolio_beta, futures_price, futures_multiplier):
    '''Calculate optimal hedge ratio'''
    return (portfolio_value * portfolio_beta) / (futures_price * futures_multiplier)

# Example usage
forward_px = forward_price(100, 0.05, 0.02, 0.25)  # $100.75
hedge_contracts = hedge_ratio(500000, 1.2, 3500, 250)  # 0.686 contracts
"""
    
    st.sidebar.download_button(
        label="💻 Download Derivatives Code",
        data=derivatives_code,
        file_name="derivatives_pricing.py",
        mime="text/python"
    )
