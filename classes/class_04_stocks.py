import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 4: Valoración de Acciones")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Modelos de Valoración", "Modelos de Dividendos", "Análisis DCF", "Valoración Comparativa"])
    
    with tab1:
        st.subheader("Fundamentos de Valoración de Acciones")
        
        st.write("""
        La valoración de acciones es el proceso de determinar el valor intrínseco de las acciones de una empresa. 
        A diferencia de los bonos con flujos de efectivo fijos, las acciones representan propiedad en una empresa con retornos variables.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Key Valuation Approaches:**
            - **Intrinsic Value**: Based on fundamentals
            - **Relative Value**: Compared to peers
            - **Absolute Value**: DCF and dividend models
            - **Market Value**: Current trading price
            """)
        
        with col2:
            st.write("""
            **Factors Affecting Stock Price:**
            - Company earnings and growth
            - Industry conditions
            - Economic environment
            - Market sentiment
            - Interest rates
            """)
        
        st.subheader("Valuation Methods Overview")
        
        methods = {
            "Dividend Discount Model (DDM)": {
                "Description": "Values stock based on present value of expected dividends",
                "Best For": "Dividend-paying mature companies",
                "Formula": "P = D₁/(r-g) for constant growth"
            },
            "Discounted Cash Flow (DCF)": {
                "Description": "Values stock based on free cash flows to equity",
                "Best For": "Companies with predictable cash flows",
                "Formula": "Sum of discounted future cash flows"
            },
            "Price Multiples": {
                "Description": "Values stock relative to financial metrics (P/E, P/B, etc.)",
                "Best For": "Quick comparison with peer companies",
                "Formula": "Price = Multiple × Metric"
            }
        }
        
        for method, details in methods.items():
            with st.expander(f"📊 {method}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Description:** {details['Description']}")
                    st.write(f"**Best For:** {details['Best For']}")
                with col2:
                    st.write(f"**Formula:** {details['Formula']}")
    
    with tab2:
        st.subheader("Dividend Discount Models")
        
        model_type = st.selectbox("Select Dividend Model:", 
                                 ["Gordon Growth Model", "Two-Stage Growth", "Zero Growth"])
        
        if model_type == "Gordon Growth Model":
            st.write("**Gordon Growth Model (Constant Growth DDM)**")
            st.latex(r"P_0 = \frac{D_1}{r - g}")
            st.write("Where: P₀ = Current price, D₁ = Next year's dividend, r = required return, g = growth rate")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                current_dividend = st.number_input("Current Dividend (D₀) ($)", value=2.00, min_value=0.0)
                growth_rate = st.number_input("Growth Rate (g) (%)", value=5.0, min_value=0.0, max_value=50.0)
            with col2:
                required_return = st.number_input("Required Return (r) (%)", value=10.0, min_value=0.1, max_value=50.0)
            with col3:
                st.write("")  # Spacing
                
            if required_return/100 > growth_rate/100:
                next_dividend = current_dividend * (1 + growth_rate/100)
                intrinsic_value = next_dividend / (required_return/100 - growth_rate/100)
                
                st.success(f"Intrinsic Value: ${intrinsic_value:.2f}")
                st.info(f"Next Year's Dividend (D₁): ${next_dividend:.2f}")
                
                # Sensitivity analysis
                if st.button("Show Sensitivity Analysis"):
                    growth_range = np.linspace(max(0, growth_rate-3), min(required_return-0.1, growth_rate+3), 20)
                    values = []
                    
                    for g in growth_range:
                        if g < required_return:
                            d1 = current_dividend * (1 + g/100)
                            value = d1 / (required_return/100 - g/100)
                            values.append(value)
                        else:
                            values.append(np.nan)
                    
                    df_sensitivity = pd.DataFrame({
                        'Growth Rate (%)': growth_range,
                        'Intrinsic Value ($)': values
                    })
                    
                    fig = px.line(df_sensitivity, x='Growth Rate (%)', y='Intrinsic Value ($)', 
                                 title='Stock Value Sensitivity to Growth Rate')
                    fig.add_vline(x=growth_rate, line_dash="dash", annotation_text="Base Case")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Growth rate must be less than required return for the model to be valid!")
        
        elif model_type == "Two-Stage Growth":
            st.write("**Two-Stage Growth Model**")
            st.write("Allows for different growth rates in two periods")
            
            col1, col2 = st.columns(2)
            with col1:
                current_dividend = st.number_input("Current Dividend ($)", value=1.50, min_value=0.0, key="2stage_d0")
                high_growth_rate = st.number_input("High Growth Rate (%)", value=15.0, min_value=0.0, key="2stage_g1")
                high_growth_years = st.number_input("High Growth Period (years)", value=5, min_value=1, max_value=20, key="2stage_years")
            with col2:
                stable_growth_rate = st.number_input("Stable Growth Rate (%)", value=3.0, min_value=0.0, key="2stage_g2")
                required_return = st.number_input("Required Return (%)", value=12.0, min_value=0.1, key="2stage_r")
            
            if required_return/100 > stable_growth_rate/100:
                # Stage 1: High growth dividends
                stage1_value = 0
                dividends_stage1 = []
                
                for year in range(1, high_growth_years + 1):
                    dividend = current_dividend * (1 + high_growth_rate/100) ** year
                    pv_dividend = dividend / (1 + required_return/100) ** year
                    stage1_value += pv_dividend
                    dividends_stage1.append({'Year': year, 'Dividend': dividend, 'PV': pv_dividend})
                
                # Stage 2: Stable growth (terminal value)
                terminal_dividend = current_dividend * (1 + high_growth_rate/100) ** high_growth_years * (1 + stable_growth_rate/100)
                terminal_value = terminal_dividend / (required_return/100 - stable_growth_rate/100)
                pv_terminal = terminal_value / (1 + required_return/100) ** high_growth_years
                
                total_value = stage1_value + pv_terminal
                
                st.success(f"Total Intrinsic Value: ${total_value:.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Stage 1 Value", f"${stage1_value:.2f}")
                    st.metric("Terminal Value", f"${terminal_value:.2f}")
                with col2:
                    st.metric("PV of Terminal Value", f"${pv_terminal:.2f}")
                    st.metric("Terminal Value %", f"{(pv_terminal/total_value)*100:.1f}%")
                
                # Show dividend projections
                df_dividends = pd.DataFrame(dividends_stage1)
                st.subheader("Dividend Projections")
                st.dataframe(df_dividends, use_container_width=True)
            else:
                st.error("Required return must be greater than stable growth rate!")
    
    with tab3:
        st.subheader("Discounted Cash Flow (DCF) Analysis")
        
        st.write("""
        DCF valuation determines intrinsic value by discounting future free cash flows to present value.
        This method is fundamental for valuing companies based on their ability to generate cash.
        """)
        
        # DCF Calculator
        st.write("**Free Cash Flow to Equity (FCFE) Model**")
        
        col1, col2 = st.columns(2)
        with col1:
            current_fcfe = st.number_input("Current FCFE ($ millions)", value=100.0, min_value=0.0)
            growth_years = st.number_input("High Growth Period (years)", value=5, min_value=1, max_value=15)
            high_growth = st.number_input("High Growth Rate (%)", value=12.0, min_value=0.0)
        with col2:
            terminal_growth = st.number_input("Terminal Growth Rate (%)", value=3.0, min_value=0.0, max_value=10.0)
            discount_rate = st.number_input("Discount Rate (%)", value=10.0, min_value=1.0)
            shares_outstanding = st.number_input("Shares Outstanding (millions)", value=50.0, min_value=0.1)
        
        if discount_rate/100 > terminal_growth/100:
            if st.button("Calculate DCF Value"):
                # High growth period cash flows
                pv_growth_cf = 0
                cash_flows = []
                
                for year in range(1, growth_years + 1):
                    cf = current_fcfe * (1 + high_growth/100) ** year
                    pv_cf = cf / (1 + discount_rate/100) ** year
                    pv_growth_cf += pv_cf
                    cash_flows.append({
                        'Year': year,
                        'FCFE': cf,
                        'Present Value': pv_cf
                    })
                
                # Terminal value
                terminal_cf = current_fcfe * (1 + high_growth/100) ** growth_years * (1 + terminal_growth/100)
                terminal_value = terminal_cf / (discount_rate/100 - terminal_growth/100)
                pv_terminal = terminal_value / (1 + discount_rate/100) ** growth_years
                
                # Total equity value
                total_equity_value = pv_growth_cf + pv_terminal
                value_per_share = total_equity_value / shares_outstanding
                
                # Results
                st.subheader("DCF Valuation Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Equity Value", f"${total_equity_value:.1f}M")
                    st.metric("Growth Period Value", f"${pv_growth_cf:.1f}M")
                with col2:
                    st.metric("Terminal Value", f"${terminal_value:.1f}M")
                    st.metric("PV of Terminal Value", f"${pv_terminal:.1f}M")
                with col3:
                    st.metric("Value Per Share", f"${value_per_share:.2f}")
                    st.metric("Terminal Value %", f"{(pv_terminal/total_equity_value)*100:.1f}%")
                
                # Cash flow table
                df_cf = pd.DataFrame(cash_flows)
                st.subheader("Projected Free Cash Flows")
                st.dataframe(df_cf, use_container_width=True)
                
                # Visualization
                years = list(range(1, growth_years + 1))
                fcfe_values = [current_fcfe * (1 + high_growth/100) ** year for year in years]
                
                fig = px.bar(x=years, y=fcfe_values, title="Projected FCFE Growth")
                fig.update_layout(xaxis_title="Year", yaxis_title="FCFE ($ millions)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Discount rate must be greater than terminal growth rate!")
    
    with tab4:
        st.subheader("Comparative Valuation (Multiples)")
        
        st.write("""
        Relative valuation uses market multiples to estimate stock value based on comparable companies.
        Common multiples include P/E, P/B, P/S, and EV/EBITDA ratios.
        """)
        
        multiple_type = st.selectbox("Select Multiple:", 
                                   ["Price-to-Earnings (P/E)", "Price-to-Book (P/B)", "Price-to-Sales (P/S)"])
        
        if multiple_type == "Price-to-Earnings (P/E)":
            st.write("**P/E Multiple Valuation**")
            st.latex(r"Target\ Price = EPS \times P/E\ Multiple")
            
            col1, col2 = st.columns(2)
            with col1:
                current_eps = st.number_input("Current EPS ($)", value=5.00, min_value=0.01)
                eps_growth = st.number_input("EPS Growth (%)", value=8.0, min_value=-50.0, max_value=100.0)
                forward_eps = current_eps * (1 + eps_growth/100)
                st.info(f"Forward EPS: ${forward_eps:.2f}")
            
            with col2:
                industry_pe = st.number_input("Industry Average P/E", value=15.0, min_value=1.0, max_value=100.0)
                company_pe = st.number_input("Company's Historical P/E", value=18.0, min_value=1.0, max_value=100.0)
            
            if st.button("Calculate P/E Valuation"):
                industry_value = forward_eps * industry_pe
                historical_value = forward_eps * company_pe
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Value using Industry P/E", f"${industry_value:.2f}")
                with col2:
                    st.metric("Value using Historical P/E", f"${historical_value:.2f}")
                
                # P/E sensitivity analysis
                pe_range = np.linspace(max(1, industry_pe - 5), industry_pe + 10, 30)
                values = [forward_eps * pe for pe in pe_range]
                
                df_pe = pd.DataFrame({
                    'P/E Multiple': pe_range,
                    'Stock Value': values
                })
                
                fig = px.line(df_pe, x='P/E Multiple', y='Stock Value', 
                             title='Stock Value vs P/E Multiple')
                fig.add_vline(x=industry_pe, line_dash="dash", annotation_text="Industry Avg")
                fig.add_vline(x=company_pe, line_dash="dot", annotation_text="Historical")
                st.plotly_chart(fig, use_container_width=True)
        
        elif multiple_type == "Price-to-Book (P/B)":
            st.write("**P/B Multiple Valuation**")
            
            col1, col2 = st.columns(2)
            with col1:
                book_value_per_share = st.number_input("Book Value per Share ($)", value=20.00, min_value=0.01)
            with col2:
                industry_pb = st.number_input("Industry Average P/B", value=2.5, min_value=0.1, max_value=20.0)
            
            pb_value = book_value_per_share * industry_pb
            st.success(f"Estimated Value using P/B: ${pb_value:.2f}")
        
        # Comparative analysis table
        st.subheader("Multiple Comparison Analysis")
        
        if st.button("Generate Comparison Table"):
            # Sample peer comparison data
            peers_data = {
                'Company': ['Target Co.', 'Peer A', 'Peer B', 'Peer C', 'Industry Avg'],
                'P/E Ratio': [0, 14.2, 16.8, 13.5, 15.0],
                'P/B Ratio': [0, 2.1, 2.8, 1.9, 2.3],
                'P/S Ratio': [0, 1.8, 2.2, 1.5, 1.9],
                'ROE (%)': [12.5, 11.2, 15.1, 9.8, 12.1],
                'Growth (%)': [8.0, 7.2, 9.5, 5.8, 7.5]
            }
            
            df_peers = pd.DataFrame(peers_data)
            st.dataframe(df_peers, use_container_width=True)
            
            st.write("**Valuation Summary using Industry Averages:**")
            if 'forward_eps' in locals():
                pe_val = forward_eps * 15.0
                st.write(f"P/E Valuation: ${pe_val:.2f}")
        
        # Practice exercise
        st.subheader("Valuation Exercise")
        
        st.write("""
        **Exercise: Value the following company**
        
        Company Data:
        - Current EPS: $4.00
        - Expected EPS Growth: 10%
        - Book Value per Share: $25.00
        - Industry P/E: 16.0x
        - Industry P/B: 2.2x
        """)
        
        exercise_pe = st.number_input("Your P/E Valuation ($):", min_value=0.0, format="%.2f", key="ex_pe")
        exercise_pb = st.number_input("Your P/B Valuation ($):", min_value=0.0, format="%.2f", key="ex_pb")
        
        if st.button("Check Exercise"):
            correct_eps = 4.00 * 1.10
            correct_pe_val = correct_eps * 16.0
            correct_pb_val = 25.00 * 2.2
            
            pe_correct = abs(exercise_pe - correct_pe_val) < 2.0
            pb_correct = abs(exercise_pb - correct_pb_val) < 2.0
            
            if pe_correct:
                st.success(f"✅ P/E Valuation Correct: ${correct_pe_val:.2f}")
            else:
                st.error(f"❌ P/E Valuation Incorrect. Correct answer: ${correct_pe_val:.2f}")
            
            if pb_correct:
                st.success(f"✅ P/B Valuation Correct: ${correct_pb_val:.2f}")
            else:
                st.error(f"❌ P/B Valuation Incorrect. Correct answer: ${correct_pb_val:.2f}")
            
            if pe_correct and pb_correct:
                progress_tracker.mark_class_completed("Class 4: Stock Valuation")
                progress_tracker.set_class_score("Class 4: Stock Valuation", 100)
                st.balloons()
                st.success("🎉 Excellent! You've mastered stock valuation!")
    
    # Download resources
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Valuation Resources")
    
    valuation_code = """
# Stock Valuation Models

def gordon_growth_model(dividend, growth_rate, required_return):
    '''Gordon Growth Model'''
    if required_return <= growth_rate:
        return None  # Invalid inputs
    
    next_dividend = dividend * (1 + growth_rate)
    return next_dividend / (required_return - growth_rate)

def two_stage_growth(current_div, high_growth, years, stable_growth, required_return):
    '''Two-stage dividend growth model'''
    
    # Stage 1: High growth
    stage1_value = 0
    for year in range(1, years + 1):
        dividend = current_div * (1 + high_growth) ** year
        pv = dividend / (1 + required_return) ** year
        stage1_value += pv
    
    # Stage 2: Stable growth (terminal value)
    terminal_dividend = current_div * (1 + high_growth) ** years * (1 + stable_growth)
    terminal_value = terminal_dividend / (required_return - stable_growth)
    pv_terminal = terminal_value / (1 + required_return) ** years
    
    return stage1_value + pv_terminal

# Example usage
value = gordon_growth_model(2.0, 0.05, 0.10)  # $42.00
two_stage_value = two_stage_growth(1.5, 0.15, 5, 0.03, 0.12)
"""
    
    st.sidebar.download_button(
        label="💻 Download Valuation Code",
        data=valuation_code,
        file_name="stock_valuation.py",
        mime="text/python"
    )
