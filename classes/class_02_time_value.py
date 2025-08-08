import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 2: Valor del Dinero en el Tiempo")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Teoría", "Cálculos", "Herramientas Interactivas", "Problemas de Práctica"])
    
    with tab1:
        st.subheader("Entendiendo el Valor del Dinero en el Tiempo")
        
        st.write("""
        El Valor del Dinero en el Tiempo (VDT) es un principio financiero fundamental que establece que el dinero 
        disponible hoy vale más que la misma cantidad en el futuro debido a su capacidad potencial de generar ganancias.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Principios Clave:**
            - El dinero tiene potencial de ganancia
            - La inflación reduce el poder adquisitivo
            - El riesgo afecta el valor futuro
            - Costo de oportunidad del capital
            """)
        
        with col2:
            st.write("""
            **Aplicaciones Principales:**
            - Decisiones de inversión
            - Cálculos de préstamos
            - Valoración de bonos
            - Valoración de proyectos
            """)
        
        st.subheader("Fórmulas Principales")
        
        with st.expander("📊 Valor Presente (VP)"):
            st.latex(r"VP = \frac{VF}{(1 + r)^n}")
            st.write("Donde: VP = Valor Presente, VF = Valor Futuro, r = tasa de interés, n = períodos")
        
        with st.expander("📈 Valor Futuro (VF)"):
            st.latex(r"VF = VP \times (1 + r)^n")
            st.write("Calcula lo que valdrá el dinero invertido hoy en el futuro")
        
        with st.expander("💰 Valor Presente de Anualidad"):
            st.latex(r"VP_{anualidad} = PMT \times \frac{1 - (1 + r)^{-n}}{r}")
            st.write("Donde: PMT = pago periódico")
        
        with st.expander("🎯 Valor Futuro de Anualidad"):
            st.latex(r"VF_{anualidad} = PMT \times \frac{(1 + r)^n - 1}{r}")
            st.write("Valor de una serie de pagos iguales en una fecha futura")
    
    with tab2:
        st.subheader("Cálculos Financieros")
        
        calc_type = st.selectbox("Selecciona el Tipo de Cálculo:", 
                                ["Valor Presente", "Valor Futuro", "VP de Anualidad", "VF de Anualidad"])
        
        if calc_type == "Valor Presente":
            st.write("**Calcular Valor Presente de Flujo de Efectivo Futuro**")
            
            col1, col2 = st.columns(2)
            with col1:
                fv = st.number_input("Valor Futuro ($)", value=1000.0, min_value=0.01)
                rate = st.number_input("Tasa de Interés (%)", value=5.0, min_value=0.0, max_value=100.0)
            with col2:
                periods = st.number_input("Number of Periods", value=10, min_value=1)
            
            if st.button("Calculate PV"):
                pv = FinancialCalculations.present_value(fv, rate/100, periods)
                st.success(f"Present Value: ${pv:.2f}")
                
                # Show calculation breakdown
                st.write("**Calculation Steps:**")
                st.write(f"PV = ${fv} / (1 + {rate/100:.3f})^{periods}")
                st.write(f"PV = ${fv} / {(1 + rate/100)**periods:.4f}")
                st.write(f"PV = ${pv:.2f}")
        
        elif calc_type == "Future Value":
            st.write("**Calculate Future Value of Present Investment**")
            
            col1, col2 = st.columns(2)
            with col1:
                pv = st.number_input("Present Value ($)", value=1000.0, min_value=0.01)
                rate = st.number_input("Interest Rate (%)", value=5.0, min_value=0.0, max_value=100.0)
            with col2:
                periods = st.number_input("Number of Periods", value=10, min_value=1)
            
            if st.button("Calculate FV"):
                fv = FinancialCalculations.future_value(pv, rate/100, periods)
                st.success(f"Future Value: ${fv:.2f}")
                
                # Visualization
                years = range(0, periods + 1)
                values = [pv * (1 + rate/100)**year for year in years]
                
                fig = px.line(x=years, y=values, title="Investment Growth Over Time")
                fig.update_layout(xaxis_title="Years", yaxis_title="Value ($)")
                st.plotly_chart(fig, use_container_width=True)
        
        elif calc_type == "Annuity PV":
            st.write("**Calculate Present Value of Annuity**")
            
            col1, col2 = st.columns(2)
            with col1:
                payment = st.number_input("Periodic Payment ($)", value=100.0, min_value=0.01)
                rate = st.number_input("Interest Rate (% per period)", value=5.0, min_value=0.0)
            with col2:
                periods = st.number_input("Number of Payments", value=10, min_value=1)
            
            if st.button("Calculate Annuity PV"):
                pv = FinancialCalculations.annuity_pv(payment, rate/100, periods)
                st.success(f"Present Value of Annuity: ${pv:.2f}")
                
                total_payments = payment * periods
                st.info(f"Total payments made: ${total_payments:.2f}")
                st.info(f"Interest savings: ${total_payments - pv:.2f}")
    
    with tab3:
        st.subheader("Interactive TVM Tools")
        
        tool_type = st.radio("Choose a tool:", ["Compound Interest Calculator", "Loan Payment Calculator", "Retirement Planner"])
        
        if tool_type == "Compound Interest Calculator":
            st.write("**See how your money grows with compound interest**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                principal = st.number_input("Initial Investment ($)", value=10000.0, min_value=1.0)
                annual_rate = st.number_input("Annual Interest Rate (%)", value=7.0, min_value=0.0)
            with col2:
                years = st.slider("Investment Period (years)", 1, 50, 20)
                compounding = st.selectbox("Compounding Frequency", ["Annually", "Semi-annually", "Quarterly", "Monthly", "Daily"])
            with col3:
                monthly_contribution = st.number_input("Monthly Contribution ($)", value=0.0, min_value=0.0)
            
            # Compounding frequency mapping
            freq_map = {"Annually": 1, "Semi-annually": 2, "Quarterly": 4, "Monthly": 12, "Daily": 365}
            n = freq_map[compounding]
            
            # Calculate compound growth
            periods_list = []
            values_list = []
            
            for year in range(0, years + 1):
                if monthly_contribution > 0:
                    # With monthly contributions
                    months = year * 12
                    fv = principal * (1 + annual_rate/100/n)**(n*year)
                    
                    if months > 0:
                        # Add future value of monthly contributions
                        monthly_rate = annual_rate/100/12
                        annuity_fv = monthly_contribution * (((1 + monthly_rate)**months - 1) / monthly_rate)
                        fv += annuity_fv
                else:
                    # Simple compound interest
                    fv = principal * (1 + annual_rate/100/n)**(n*year)
                
                periods_list.append(year)
                values_list.append(fv)
            
            # Create visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=periods_list, y=values_list, mode='lines+markers', name='Investment Value'))
            fig.update_layout(title="Investment Growth Over Time", xaxis_title="Years", yaxis_title="Value ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Final results
            final_value = values_list[-1]
            total_contributions = principal + (monthly_contribution * 12 * years)
            total_interest = final_value - total_contributions
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Value", f"${final_value:.2f}")
            with col2:
                st.metric("Total Contributions", f"${total_contributions:.2f}")
            with col3:
                st.metric("Interest Earned", f"${total_interest:.2f}")
    
    with tab4:
        st.subheader("Practice Problems")
        
        st.write("Test your understanding with these practice problems:")
        
        problem_num = st.selectbox("Select Problem:", ["Problem 1", "Problem 2", "Problem 3"])
        
        if problem_num == "Problem 1":
            st.write("""
            **Problem 1: Present Value Calculation**
            
            You expect to receive $5,000 in 3 years. If the discount rate is 6% per year, 
            what is the present value of this future cash flow?
            """)
            
            user_answer = st.number_input("Your answer ($):", min_value=0.0, format="%.2f", key="p1")
            
            if st.button("Check Answer", key="check_p1"):
                correct_answer = FinancialCalculations.present_value(5000, 0.06, 3)
                if abs(user_answer - correct_answer) < 0.50:
                    st.success(f"✅ Correct! The answer is ${correct_answer:.2f}")
                else:
                    st.error(f"❌ Incorrect. The correct answer is ${correct_answer:.2f}")
                    st.write("**Solution:** PV = $5,000 / (1.06)³ = $5,000 / 1.1910 = $4,198.10")
        
        elif problem_num == "Problem 2":
            st.write("""
            **Problem 2: Future Value with Monthly Contributions**
            
            If you invest $200 per month for 10 years at an annual interest rate of 8%, 
            what will be the future value of your investment?
            """)
            
            user_answer = st.number_input("Your answer ($):", min_value=0.0, format="%.2f", key="p2")
            
            if st.button("Check Answer", key="check_p2"):
                monthly_rate = 0.08 / 12
                months = 10 * 12
                fv_annuity = 200 * (((1 + monthly_rate)**months - 1) / monthly_rate)
                
                if abs(user_answer - fv_annuity) < 100:
                    st.success(f"✅ Correct! The answer is approximately ${fv_annuity:.2f}")
                else:
                    st.error(f"❌ Incorrect. The correct answer is ${fv_annuity:.2f}")
                    st.write("**Solution:** This is a future value of annuity calculation with monthly compounding.")
        
        # Final quiz
        st.subheader("Class Completion Quiz")
        
        col1, col2 = st.columns(2)
        
        with col1:
            q1 = st.radio(
                "What happens to present value when interest rate increases?",
                ["Increases", "Decreases", "Stays the same", "Cannot determine"],
                key="quiz_q1"
            )
        
        with col2:
            q2 = st.radio(
                "Which has higher future value: $1000 compounded annually or quarterly at 8%?",
                ["Annually", "Quarterly", "Same", "Cannot determine"],
                key="quiz_q2"
            )
        
        q3 = st.radio(
            "The present value of an annuity _______ as the number of payments increases.",
            ["Always increases", "Always decreases", "Increases then levels off", "Remains constant"],
            key="quiz_q3"
        )
        
        if st.button("Submit Quiz"):
            score = 0
            
            if q1 == "Decreases":
                st.success("Q1: Correct! ✅")
                score += 1
            else:
                st.error("Q1: Incorrect. Present value decreases when interest rate increases.")
            
            if q2 == "Quarterly":
                st.success("Q2: Correct! ✅")
                score += 1
            else:
                st.error("Q2: Incorrect. More frequent compounding results in higher future value.")
            
            if q3 == "Increases then levels off":
                st.success("Q3: Correct! ✅")
                score += 1
            else:
                st.error("Q3: Incorrect. PV increases with more payments but at a decreasing rate.")
            
            st.write(f"Your score: {score}/3")
            
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Class 2: Time Value of Money")
                progress_tracker.set_class_score("Class 2: Time Value of Money", (score/3) * 100)
                st.success("🎉 Congratulations! You've completed Class 2!")
    
    # Downloadable materials
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Class Materials")
    
    code_example = """
# Time Value of Money Calculations in Python

import numpy as np

def present_value(fv, rate, periods):
    \"\"\"Calculate present value\"\"\"
    return fv / (1 + rate) ** periods

def future_value(pv, rate, periods):
    \"\"\"Calculate future value\"\"\"
    return pv * (1 + rate) ** periods

def annuity_pv(payment, rate, periods):
    \"\"\"Present value of annuity\"\"\"
    return payment * (1 - (1 + rate) ** -periods) / rate

# Example usage
pv = present_value(1000, 0.05, 10)  # $613.91
fv = future_value(1000, 0.05, 10)   # $1628.89
annuity = annuity_pv(100, 0.05, 10) # $772.17
"""
    
    st.sidebar.download_button(
        label="💻 Download Python Code",
        data=code_example,
        file_name="tvm_calculations.py",
        mime="text/python"
    )
