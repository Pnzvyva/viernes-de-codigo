import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 3: Valoración de Bonos")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Fundamentos de Bonos", "Modelos de Valoración", "Precio Interactivo", "Análisis de Rendimiento"])
    
    with tab1:
        st.subheader("Entendiendo los Bonos")
        
        st.write("""
        Un bono es un título de deuda que representa un préstamo hecho por un inversionista a un prestatario. 
        Cuando compras un bono, estás prestando dinero al emisor a cambio de pagos regulares de intereses 
        y el retorno del valor nominal del bono cuando vence.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Key Bond Terms:**
            - **Face Value (Par)**: Amount paid at maturity
            - **Coupon Rate**: Annual interest rate
            - **Maturity**: When the bond expires
            - **Yield**: Return on investment
            - **Duration**: Price sensitivity to interest rates
            """)
        
        with col2:
            st.write("""
            **Types of Bonds:**
            - Government Bonds (Treasuries)
            - Corporate Bonds
            - Municipal Bonds
            - Zero-Coupon Bonds
            - Convertible Bonds
            """)
        
        st.subheader("Bond Pricing Fundamentals")
        
        st.write("""
        Bond prices are determined by discounting future cash flows (coupon payments and principal repayment) 
        to present value using the market interest rate.
        """)
        
        with st.expander("📊 Basic Bond Pricing Formula"):
            st.latex(r"P = \sum_{t=1}^{n} \frac{C}{(1+r)^t} + \frac{FV}{(1+r)^n}")
            st.write("Where: P = Price, C = Coupon payment, r = yield, n = periods, FV = Face value")
        
        with st.expander("🎯 Relationship Between Price and Yield"):
            st.write("""
            - **Inverse Relationship**: When yields ↑, prices ↓
            - **At Par**: Price = Face value when coupon rate = yield
            - **Premium**: Price > Face value when coupon rate > yield
            - **Discount**: Price < Face value when coupon rate < yield
            """)
    
    with tab2:
        st.subheader("Bond Valuation Models")
        
        model_type = st.selectbox("Select Valuation Model:", 
                                 ["Standard Bond", "Zero-Coupon Bond", "Perpetual Bond"])
        
        if model_type == "Standard Bond":
            st.write("**Standard Coupon Bond Valuation**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                face_value = st.number_input("Face Value ($)", value=1000.0, min_value=100.0)
                coupon_rate = st.number_input("Coupon Rate (%)", value=5.0, min_value=0.0, max_value=20.0)
            with col2:
                market_rate = st.number_input("Market Yield (%)", value=6.0, min_value=0.0, max_value=20.0)
                periods = st.number_input("Years to Maturity", value=10, min_value=1, max_value=50)
            with col3:
                frequency = st.selectbox("Payment Frequency", ["Annual", "Semi-annual"], key="bond_freq")
                
            if st.button("Calculate Bond Price"):
                # Adjust for payment frequency
                if frequency == "Semi-annual":
                    coupon_rate_adj = coupon_rate / 2
                    market_rate_adj = market_rate / 2
                    periods_adj = periods * 2
                else:
                    coupon_rate_adj = coupon_rate
                    market_rate_adj = market_rate
                    periods_adj = periods
                
                bond_price = FinancialCalculations.bond_price(
                    face_value, coupon_rate_adj/100, market_rate_adj/100, periods_adj
                )
                
                st.success(f"Bond Price: ${bond_price:.2f}")
                
                # Analysis
                if bond_price > face_value:
                    premium = bond_price - face_value
                    st.info(f"Trading at PREMIUM: ${premium:.2f} above par")
                elif bond_price < face_value:
                    discount = face_value - bond_price
                    st.info(f"Trading at DISCOUNT: ${discount:.2f} below par")
                else:
                    st.info("Trading at PAR")
                
                # Cash flow breakdown
                st.subheader("Cash Flow Analysis")
                
                coupon_payment = face_value * (coupon_rate_adj/100)
                periods_range = range(1, periods_adj + 1)
                
                cash_flows = []
                for period in periods_range:
                    if period == periods_adj:
                        # Final payment includes principal
                        cf = coupon_payment + face_value
                    else:
                        cf = coupon_payment
                    
                    pv_cf = cf / (1 + market_rate_adj/100) ** period
                    cash_flows.append({
                        'Period': period,
                        'Cash Flow': cf,
                        'Present Value': pv_cf
                    })
                
                df_cf = pd.DataFrame(cash_flows)
                
                # Show first few and last few periods
                if len(df_cf) > 10:
                    st.write("**First 5 periods:**")
                    st.dataframe(df_cf.head(), use_container_width=True)
                    st.write("**Last 5 periods:**")
                    st.dataframe(df_cf.tail(), use_container_width=True)
                else:
                    st.dataframe(df_cf, use_container_width=True)
        
        elif model_type == "Zero-Coupon Bond":
            st.write("**Zero-Coupon Bond Valuation**")
            st.write("A zero-coupon bond pays no interest but is sold at a discount to face value.")
            
            col1, col2 = st.columns(2)
            with col1:
                face_value = st.number_input("Face Value ($)", value=1000.0, min_value=100.0, key="zcb_fv")
                market_rate = st.number_input("Market Yield (%)", value=6.0, min_value=0.0, max_value=20.0, key="zcb_rate")
            with col2:
                periods = st.number_input("Years to Maturity", value=10, min_value=1, max_value=50, key="zcb_periods")
            
            if st.button("Calculate Zero-Coupon Bond Price"):
                zcb_price = FinancialCalculations.present_value(face_value, market_rate/100, periods)
                st.success(f"Zero-Coupon Bond Price: ${zcb_price:.2f}")
                
                discount = face_value - zcb_price
                yield_pct = (face_value / zcb_price) ** (1/periods) - 1
                
                st.info(f"Discount: ${discount:.2f}")
                st.info(f"Implied Annual Yield: {yield_pct*100:.2f}%")
    
    with tab3:
        st.subheader("Interactive Bond Pricing Tool")
        
        st.write("Explore how bond prices change with different parameters:")
        
        # Bond parameters
        col1, col2 = st.columns(2)
        with col1:
            base_face_value = st.slider("Face Value ($)", 500, 5000, 1000, step=100)
            base_coupon_rate = st.slider("Coupon Rate (%)", 0.0, 15.0, 5.0, step=0.5)
        with col2:
            base_years = st.slider("Years to Maturity", 1, 30, 10)
            yield_range = st.slider("Yield Range (±%)", 1.0, 10.0, 5.0, step=0.5)
        
        if st.button("Generate Price-Yield Curve"):
            # Create yield range
            base_yield = base_coupon_rate
            yields = np.linspace(base_yield - yield_range, base_yield + yield_range, 50)
            prices = []
            
            for y in yields:
                if y > 0:  # Avoid negative yields
                    price = FinancialCalculations.bond_price(
                        base_face_value, base_coupon_rate/100, y/100, base_years
                    )
                    prices.append(price)
                else:
                    prices.append(np.nan)
            
            # Create DataFrame
            df_curve = pd.DataFrame({
                'Yield (%)': yields,
                'Bond Price ($)': prices
            })
            
            # Plot price-yield curve
            fig = px.line(df_curve, x='Yield (%)', y='Bond Price ($)', 
                         title='Bond Price vs Yield Curve')
            
            # Add reference lines
            fig.add_hline(y=base_face_value, line_dash="dash", line_color="red", 
                         annotation_text="Par Value")
            fig.add_vline(x=base_coupon_rate, line_dash="dash", line_color="green", 
                         annotation_text="Coupon Rate")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key insights
            st.subheader("Key Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                current_price = FinancialCalculations.bond_price(
                    base_face_value, base_coupon_rate/100, base_coupon_rate/100, base_years
                )
                st.metric("Price at Coupon Rate", f"${current_price:.2f}")
                
                high_yield_price = FinancialCalculations.bond_price(
                    base_face_value, base_coupon_rate/100, (base_coupon_rate + 2)/100, base_years
                )
                st.metric("Price at +2% Yield", f"${high_yield_price:.2f}")
            
            with col2:
                low_yield_price = FinancialCalculations.bond_price(
                    base_face_value, base_coupon_rate/100, max(0.1, base_coupon_rate - 2)/100, base_years
                )
                st.metric("Price at -2% Yield", f"${low_yield_price:.2f}")
                
                price_change = ((high_yield_price - low_yield_price) / current_price) * 100
                st.metric("Price Sensitivity", f"{price_change:.1f}%")
    
    with tab4:
        st.subheader("Yield Analysis")
        
        st.write("Understanding different yield measures:")
        
        yield_type = st.selectbox("Select Yield Measure:", 
                                 ["Current Yield", "Yield to Maturity", "Yield to Call"])
        
        if yield_type == "Current Yield":
            st.write("**Current Yield = Annual Coupon Payment / Current Price**")
            
            col1, col2 = st.columns(2)
            with col1:
                annual_coupon = st.number_input("Annual Coupon ($)", value=50.0, min_value=0.0)
            with col2:
                current_price = st.number_input("Current Bond Price ($)", value=950.0, min_value=0.01)
            
            if st.button("Calculate Current Yield"):
                current_yield = (annual_coupon / current_price) * 100
                st.success(f"Current Yield: {current_yield:.2f}%")
                
                st.write("**Note:** Current yield doesn't account for capital gains/losses at maturity.")
        
        elif yield_type == "Yield to Maturity":
            st.write("**Yield to Maturity (YTM): The internal rate of return of the bond**")
            
            col1, col2 = st.columns(2)
            with col1:
                bond_price = st.number_input("Current Bond Price ($)", value=950.0, min_value=0.01, key="ytm_price")
                face_value = st.number_input("Face Value ($)", value=1000.0, min_value=0.01, key="ytm_fv")
            with col2:
                annual_coupon = st.number_input("Annual Coupon ($)", value=50.0, min_value=0.0, key="ytm_coupon")
                years = st.number_input("Years to Maturity", value=10, min_value=1, key="ytm_years")
            
            if st.button("Estimate YTM"):
                # Approximate YTM calculation
                numerator = annual_coupon + (face_value - bond_price) / years
                denominator = (face_value + bond_price) / 2
                ytm_approx = (numerator / denominator) * 100
                
                st.success(f"Approximate YTM: {ytm_approx:.2f}%")
                
                # More precise calculation using iterative method
                def ytm_function(rate):
                    price = 0
                    for t in range(1, years + 1):
                        if t == years:
                            cash_flow = annual_coupon + face_value
                        else:
                            cash_flow = annual_coupon
                        price += cash_flow / (1 + rate) ** t
                    return price - bond_price
                
                # Use bisection method for more accurate YTM
                low, high = 0.001, 0.5
                for _ in range(100):  # Max 100 iterations
                    mid = (low + high) / 2
                    if abs(ytm_function(mid)) < 0.01:
                        break
                    if ytm_function(mid) > 0:
                        low = mid
                    else:
                        high = mid
                
                precise_ytm = mid * 100
                st.info(f"Precise YTM: {precise_ytm:.3f}%")
        
        # Practice problems
        st.subheader("Practice Problems")
        
        problem = st.selectbox("Select Problem:", ["Problem 1: Bond Pricing", "Problem 2: Yield Calculation"])
        
        if problem == "Problem 1: Bond Pricing":
            st.write("""
            **Problem 1:**
            A bond has a face value of $1,000, pays a 6% annual coupon, and matures in 8 years. 
            If the current market yield is 7%, what is the bond's price?
            """)
            
            user_answer = st.number_input("Your answer ($):", min_value=0.0, format="%.2f", key="bond_p1")
            
            if st.button("Check Answer", key="check_bond_p1"):
                correct_answer = FinancialCalculations.bond_price(1000, 0.06, 0.07, 8)
                if abs(user_answer - correct_answer) < 5.0:
                    st.success(f"✅ Correct! The answer is ${correct_answer:.2f}")
                    st.write("The bond trades at a discount because the market yield (7%) > coupon rate (6%)")
                else:
                    st.error(f"❌ Incorrect. The correct answer is ${correct_answer:.2f}")
        
        # Final assessment
        st.subheader("Class Assessment")
        
        q1 = st.radio(
            "When market interest rates rise, bond prices:",
            ["Rise", "Fall", "Stay the same", "Become more volatile"],
            key="bond_q1"
        )
        
        q2 = st.radio(
            "A bond trading above par value has:",
            ["Coupon rate > Market yield", "Coupon rate < Market yield", 
             "Coupon rate = Market yield", "Negative yield"],
            key="bond_q2"
        )
        
        q3 = st.radio(
            "Duration measures:",
            ["Time to maturity", "Price sensitivity to interest rates", 
             "Credit risk", "Liquidity risk"],
            key="bond_q3"
        )
        
        if st.button("Submit Assessment"):
            score = 0
            
            if q1 == "Fall":
                st.success("Q1: Correct! ✅")
                score += 1
            else:
                st.error("Q1: Incorrect. Bond prices fall when interest rates rise.")
            
            if q2 == "Coupon rate > Market yield":
                st.success("Q2: Correct! ✅")
                score += 1
            else:
                st.error("Q2: Incorrect. Premium bonds have coupon rates above market yields.")
            
            if q3 == "Price sensitivity to interest rates":
                st.success("Q3: Correct! ✅")
                score += 1
            else:
                st.error("Q3: Incorrect. Duration measures interest rate sensitivity.")
            
            st.write(f"Your score: {score}/3")
            
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Class 3: Bond Valuation")
                progress_tracker.set_class_score("Class 3: Bond Valuation", (score/3) * 100)
                st.success("🎉 Congratulations! You've completed Class 3!")
    
    # Download materials
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Bond Resources")
    
    bond_code = """
# Bond Valuation Code

def bond_price(face_value, coupon_rate, market_rate, periods):
    '''Calculate bond price'''
    coupon = face_value * coupon_rate
    
    # Present value of coupon payments
    pv_coupons = coupon * (1 - (1 + market_rate) ** -periods) / market_rate
    
    # Present value of face value
    pv_face = face_value / (1 + market_rate) ** periods
    
    return pv_coupons + pv_face

# Example
price = bond_price(1000, 0.06, 0.07, 8)  # $944.72
"""
    
    st.sidebar.download_button(
        label="💻 Download Bond Code",
        data=bond_code,
        file_name="bond_valuation.py",
        mime="text/python"
    )
