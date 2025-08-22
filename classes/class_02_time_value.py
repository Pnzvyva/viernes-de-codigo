import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 2: Valor del Dinero en el Tiempo")
    
    progress_tracker: ProgressTracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Teoría", "Cálculos", "Herramientas Interactivas", "Problemas de Práctica"])
    
    # -------------------------------
    # TAB 1: TEORÍA
    # -------------------------------
    with tab1:
        st.subheader("Entendiendo el Valor del Dinero en el Tiempo")
        st.write("""
        El Valor del Dinero en el Tiempo (VDT) es un principio financiero fundamental que establece que el dinero 
        disponible hoy vale más que la misma cantidad en el futuro debido a su capacidad potencial de generar rendimientos.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            **Principios clave:**
            - El dinero tiene potencial de ganancia.
            - La inflación reduce el poder adquisitivo.
            - El riesgo afecta el valor futuro.
            - Existe un costo de oportunidad del capital.
            """)
        with col2:
            st.write("""
            **Aplicaciones principales:**
            - Decisiones de inversión.
            - Cálculos de préstamos.
            - Valoración de bonos.
            - Evaluación de proyectos.
            """)

        st.subheader("Fórmulas principales")

        with st.expander("📊 Valor Presente (VP)"):
            st.latex(r"VP = \frac{VF}{(1 + r)^n}")
            st.write("Donde: VP = Valor Presente, VF = Valor Futuro, r = tasa de interés por período, n = número de períodos.")
        
        with st.expander("📈 Valor Futuro (VF)"):
            st.latex(r"VF = VP \times (1 + r)^n")
            st.write("Calcula cuánto valdrá en el futuro una inversión hecha hoy.")
        
        with st.expander("💰 Valor Presente de una Anualidad (VP de anualidad)"):
            st.latex(r"VP_{anualidad} = PMT \times \frac{1 - (1 + r)^{-n}}{r}")
            st.write("Donde: PMT = pago periódico.")
        
        with st.expander("🎯 Valor Futuro de una Anualidad (VF de anualidad)"):
            st.latex(r"VF_{anualidad} = PMT \times \frac{(1 + r)^n - 1}{r}")
            st.write("Valor acumulado en el futuro de una serie de pagos iguales.")
    
    # -------------------------------
    # TAB 2: CÁLCULOS
    # -------------------------------
    with tab2:
        st.subheader("Cálculos financieros")
        
        calc_type = st.selectbox(
            "Selecciona el tipo de cálculo:",
            ["Valor Presente", "Valor Futuro", "VP de Anualidad", "VF de Anualidad"],
            key="t2_calc_selector"
        )
        
        # ---- Valor Presente ----
        if calc_type == "Valor Presente":
            st.write("**Calcular el Valor Presente de un flujo de efectivo futuro**")
            col1, col2 = st.columns(2)
            with col1:
                fv = st.number_input("Valor Futuro ($)", value=1000.0, min_value=0.01, key="t2_vp_fv")
                rate = st.number_input("Tasa de interés (% por período)", value=5.0, min_value=0.0, max_value=100.0, key="t2_vp_rate")
            with col2:
                periods = st.number_input("Número de períodos", value=10, min_value=1, key="t2_vp_n")
            
            if st.button("Calcular VP", key="t2_btn_vp"):
                pv = FinancialCalculations.present_value(fv, rate/100, periods)
                st.success(f"Valor Presente: ${pv:.2f}")
                
                st.write("**Desglose del cálculo:**")
                st.write(f"VP = ${fv} / (1 + {rate/100:.4f})^{periods}")
                st.write(f"VP = ${fv} / {(1 + rate/100)**periods:.6f}")
                st.write(f"VP = ${pv:.2f}")
        
        # ---- Valor Futuro ----
        elif calc_type == "Valor Futuro":
            st.write("**Calcular el Valor Futuro de una inversión presente**")
            col1, col2 = st.columns(2)
            with col1:
                pv = st.number_input("Valor Presente ($)", value=1000.0, min_value=0.01, key="t2_vf_pv")
                rate = st.number_input("Tasa de interés (% por período)", value=5.0, min_value=0.0, max_value=100.0, key="t2_vf_rate")
            with col2:
                periods = st.number_input("Número de períodos", value=10, min_value=1, key="t2_vf_n")
            
            if st.button("Calcular VF", key="t2_btn_vf"):
                fv = FinancialCalculations.future_value(pv, rate/100, periods)
                st.success(f"Valor Futuro: ${fv:.2f}")
                
                # Visualización del crecimiento
                anios = list(range(0, periods + 1))
                valores = [pv * (1 + rate/100)**t for t in anios]
                fig = px.line(x=anios, y=valores, title="Crecimiento de la inversión en el tiempo")
                fig.update_layout(xaxis_title="Períodos", yaxis_title="Valor ($)")
                st.plotly_chart(fig, use_container_width=True)
        
        # ---- VP de Anualidad ----
        elif calc_type == "VP de Anualidad":
            st.write("**Calcular el Valor Presente de una Anualidad**")
            col1, col2 = st.columns(2)
            with col1:
                payment = st.number_input("Pago periódico ($)", value=100.0, min_value=0.01, key="t2_vpa_pmt")
                rate = st.number_input("Tasa de interés (% por período)", value=5.0, min_value=0.0, key="t2_vpa_rate")
            with col2:
                periods = st.number_input("Número de pagos", value=10, min_value=1, key="t2_vpa_n")
            
            if st.button("Calcular VP de anualidad", key="t2_btn_vpa"):
                pv = FinancialCalculations.annuity_pv(payment, rate/100, periods)
                st.success(f"Valor Presente de la Anualidad: ${pv:.2f}")
                
                total_pagado = payment * periods
                st.info(f"Total pagado (suma de PMT): ${total_pagado:.2f}")
                st.info(f"Diferencia total − VP (aprox. intereses): ${total_pagado - pv:.2f}")
        
        # ---- VF de Anualidad ----
        elif calc_type == "VF de Anualidad":
            st.write("**Calcular el Valor Futuro de una Anualidad**")
            col1, col2 = st.columns(2)
            with col1:
                payment = st.number_input("Pago periódico ($)", value=100.0, min_value=0.01, key="t2_vfa_pmt")
                rate = st.number_input("Tasa de interés (% por período)", value=5.0, min_value=0.0, key="t2_vfa_rate")
            with col2:
                periods = st.number_input("Número de pagos", value=10, min_value=1, key="t2_vfa_n")
            
            if st.button("Calcular VF de anualidad", key="t2_btn_vfa"):
                r = rate / 100.0
                if r == 0:
                    fv = payment * periods
                else:
                    fv = payment * ((1 + r) ** periods - 1) / r
                st.success(f"Valor Futuro de la Anualidad: ${fv:.2f}")
                
                total_pagado = payment * periods
                intereses_aprox = fv - total_pagado
                st.info(f"Total aportado (suma de PMT): ${total_pagado:.2f}")
                st.info(f"Intereses ganados (aprox.): ${intereses_aprox:.2f}")
    
    # -------------------------------
    # TAB 3: HERRAMIENTAS INTERACTIVAS
    # -------------------------------
    with tab3:
        st.subheader("Herramientas interactivas de VDT")
        
        tool_type = st.radio(
            "Elige una herramienta:",
            ["Calculadora de interés compuesto", "Calculadora de pago de préstamo", "Planificador de retiro"],
            key="t3_tool_selector"
        )
        
        # -------------------------------------------------
        # 1) Calculadora de interés compuesto
        # -------------------------------------------------
        
        if tool_type == "Calculadora de interés compuesto":
            st.write("**Observa cómo crece tu dinero con interés compuesto**")
            col1, col2, col3 = st.columns(3)
            with col1:
                principal = st.number_input("Inversión inicial ($)", value=10000.0, min_value=1.0, key="t3_ic_principal")
                annual_rate = st.number_input("Tasa de interés anual (%)", value=7.0, min_value=0.0, key="t3_ic_rate")
            with col2:
                years = st.slider("Período de inversión (años)", 1, 50, 20, key="t3_ic_years")
                compounding = st.selectbox("Frecuencia de capitalización", ["Anual", "Semestral", "Trimestral", "Mensual", "Diaria"], key="t3_ic_freq")
            with col3:
                monthly_contribution = st.number_input("Aporte mensual ($)", value=0.0, min_value=0.0, key="t3_ic_pmt")
            
            freq_map = {"Anual": 1, "Semestral": 2, "Trimestral": 4, "Mensual": 12, "Diaria": 365}
            n = freq_map[compounding]
            
            periodos, valores = [], []
            for year in range(0, years + 1):
                if monthly_contribution > 0:
                    meses = year * 12
                    fv = principal * (1 + annual_rate/100/n) ** (n * year)
                    if meses > 0:
                        r_m = annual_rate/100/12
                        annuity_fv = monthly_contribution * meses if r_m == 0 else monthly_contribution * (((1 + r_m) ** meses - 1) / r_m)
                        fv += annuity_fv
                else:
                    fv = principal * (1 + annual_rate/100/n) ** (n * year)
                periodos.append(year)
                valores.append(fv)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=periodos, y=valores, mode='lines+markers', name='Valor de la inversión'))
            fig.update_layout(title="Crecimiento de la inversión en el tiempo", xaxis_title="Años", yaxis_title="Valor ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            valor_final = valores[-1]
            aportes_totales = principal + (monthly_contribution * 12 * years)
            intereses_totales = valor_final - aportes_totales
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor final", f"${valor_final:.2f}")
            c2.metric("Aportes totales", f"${aportes_totales:.2f}")
            c3.metric("Intereses ganados", f"${intereses_totales:.2f}")
        
        # -------------------------------------------------
        # 2) Calculadora de pago de préstamo
        # -------------------------------------------------
        elif tool_type == "Calculadora de pago de préstamo":
            st.write("**Calcula la cuota (PMT) y genera la tabla de amortización**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                principal = st.number_input("Monto del préstamo ($)", value=100000.0, min_value=0.01, step=1000.0, key="t3_lp_principal")
            with col2:
                annual_rate = st.number_input("Tasa de interés anual (%)", value=12.0, min_value=0.0, step=0.1, key="t3_lp_rate")
            with col3:
                years = st.number_input("Plazo (años)", value=5, min_value=1, step=1, key="t3_lp_years")
            
            frecuencia = st.selectbox("Frecuencia de pago", ["Mensual", "Trimestral", "Semestral", "Anual"], key="t3_lp_freq")
            freq_map = {"Mensual": 12, "Trimestral": 4, "Semestral": 2, "Anual": 1}
            m = freq_map[frecuencia]
            
            n = int(years * m)
            r = (annual_rate / 100.0) / m
            
            if r == 0:
                pmt = principal / n
            else:
                pmt = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
            
            st.success(f"Cuota periódica (PMT): ${pmt:,.2f}")
            
            saldo = principal
            filas = []
            for k in range(1, n + 1):
                interes = 0.0 if r == 0 else saldo * r
                abono_capital = pmt - interes
                # Ajuste final por redondeo
                if k == n and abs(saldo - abono_capital) < 0.01:
                    abono_capital = saldo
                    pmt_efectivo = interes + abono_capital
                else:
                    pmt_efectivo = pmt
                saldo = max(saldo - abono_capital, 0.0)
                filas.append({
                    "Período": k,
                    "Cuota (PMT)": round(pmt_efectivo, 2),
                    "Interés": round(interes, 2),
                    "Abono a capital": round(abono_capital, 2),
                    "Saldo": round(saldo, 2)
                })
            
            df_amort = pd.DataFrame(filas)
            st.dataframe(df_amort, use_container_width=True, height=360)
            
            total_pagado = df_amort["Cuota (PMT)"].sum()
            total_interes = df_amort["Interés"].sum()
            total_capital = df_amort["Abono a capital"].sum()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total pagado", f"${total_pagado:,.2f}")
            c2.metric("Total intereses", f"${total_interes:,.2f}")
            c3.metric("Total capital", f"${total_capital:,.2f}")
            
            fig_saldo = go.Figure()
            fig_saldo.add_trace(go.Scatter(x=df_amort["Período"], y=df_amort["Saldo"], mode="lines", name="Saldo"))
            fig_saldo.update_layout(title="Evolución del saldo del préstamo", xaxis_title="Período", yaxis_title="Saldo ($)")
            st.plotly_chart(fig_saldo, use_container_width=True)
        
        # -------------------------------------------------
        # 3) Planificador de retiro
        # -------------------------------------------------
        elif tool_type == "Planificador de retiro":
            st.write("**Proyecta tus ahorros hasta la edad de retiro con aportes mensuales**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                edad_actual = st.number_input("Edad actual", value=30, min_value=16, max_value=80, step=1, key="t3_ret_age_now")
                ahorro_actual = st.number_input("Ahorro actual ($)", value=10000.0, min_value=0.0, step=100.0, key="t3_ret_savings_now")
            with col2:
                edad_retiro = st.number_input("Edad de retiro", value=65, min_value=31, max_value=90, step=1, key="t3_ret_age_ret")
                aporte_mensual = st.number_input("Aporte mensual ($)", value=300.0, min_value=0.0, step=10.0, key="t3_ret_pmt")
            with col3:
                tasa_anual = st.number_input("Rentabilidad anual esperada (%)", value=7.0, min_value=0.0, step=0.1, key="t3_ret_return")
                tasa_retiro = st.number_input("Tasa de retiro segura anual (%)", value=4.0, min_value=0.0, step=0.1, key="t3_ret_swr",
                                               help="Regla aproximada del 4% para estimar ingresos anuales en retiro")
            
            anos = int(edad_retiro - edad_actual)
            meses = anos * 12
            r_m = (tasa_anual / 100.0) / 12.0
            
            fv_ahorro_actual = ahorro_actual * ((1 + r_m) ** meses)
            fv_aportes = aporte_mensual * meses if r_m == 0 else aporte_mensual * (((1 + r_m) ** meses - 1) / r_m)
            valor_final = fv_ahorro_actual + fv_aportes
            
            aportes_totales = ahorro_actual + (aporte_mensual * meses)
            intereses_ganados = valor_final - aportes_totales
            ingreso_anual_estimado = valor_final * (tasa_retiro / 100.0)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor proyectado al retiro", f"${valor_final:,.2f}")
            c2.metric("Aportes totales (acumulados)", f"${aportes_totales:,.2f}")
            c3.metric("Rendimientos estimados", f"${intereses_ganados:,.2f}")
            
            st.info(f"Ingreso anual estimado al {tasa_retiro:.1f}%: **${ingreso_anual_estimado:,.2f}**")
            
            anios = list(range(0, anos + 1))
            valores = []
            for y in anios:
                m = y * 12
                parte_ahorro = ahorro_actual * ((1 + r_m) ** m)
                parte_aportes = aporte_mensual * m if r_m == 0 else aporte_mensual * (((1 + r_m) ** m - 1) / r_m)
                valores.append(parte_ahorro + parte_aportes)
            
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Scatter(x=anios, y=valores, mode="lines+markers", name="Ahorro proyectado"))
            fig_ret.update_layout(title="Crecimiento proyectado hasta la edad de retiro",
                                  xaxis_title="Años desde hoy",
                                  yaxis_title="Valor acumulado ($)")
            st.plotly_chart(fig_ret, use_container_width=True)
    
    # -------------------------------
    # TAB 4: PROBLEMAS DE PRÁCTICA
    # -------------------------------
    with tab4:
        st.subheader("Problemas de práctica")
        st.write("Pon a prueba tu comprensión con estos ejercicios:")
        
        # Eliminado "Problema 3"
        problem_num = st.selectbox("Selecciona el problema:", ["Problema 1", "Problema 2"], key="t4_problem_selector")
        
        if problem_num == "Problema 1":
            st.write("""
            **Problema 1: Cálculo de Valor Presente**
            
            Esperas recibir $5,000 en 3 años. Si la tasa de descuento es 6% anual,
            ¿cuál es el valor presente de ese flujo futuro?
            """)
            user_answer = st.number_input("Tu respuesta ($):", min_value=0.0, format="%.2f", key="t4_p1_answer")
            if st.button("Verificar respuesta", key="t4_p1_check"):
                correct_answer = FinancialCalculations.present_value(5000, 0.06, 3)
                if abs(user_answer - correct_answer) < 0.50:
                    st.success(f"✅ ¡Correcto! La respuesta es ${correct_answer:.2f}")
                else:
                    st.error(f"❌ Incorrecto. La respuesta correcta es ${correct_answer:.2f}")
                    st.write("**Solución:** VP = $5,000 / (1.06)³ = $5,000 / 1.1910 ≈ $4,198.10")
        
        elif problem_num == "Problema 2":
            st.write("""
            **Problema 2: Valor Futuro con aportes mensuales**
            
            Si inviertes $200 al mes durante 10 años a una tasa anual del 8%,
            ¿cuál será el valor futuro de tu inversión?
            """)
            user_answer = st.number_input("Tu respuesta ($):", min_value=0.0, format="%.2f", key="t4_p2_answer")
            if st.button("Verificar respuesta", key="t4_p2_check"):
                monthly_rate = 0.08 / 12
                months = 10 * 12
                if monthly_rate == 0:
                    fv_annuity = 200 * months
                else:
                    fv_annuity = 200 * (((1 + monthly_rate) ** months - 1) / monthly_rate)
                if abs(user_answer - fv_annuity) < 100:
                    st.success(f"✅ ¡Correcto! La respuesta es aproximadamente ${fv_annuity:.2f}")
                else:
                    st.error(f"❌ Incorrecto. La respuesta correcta es ${fv_annuity:.2f}")
                    st.write("**Solución:** Es un cálculo de valor futuro de una anualidad con capitalización mensual.")
        
        # Quiz final
        st.subheader("Cuestionario de cierre de la clase")
        col1, col2 = st.columns(2)
        with col1:
            q1 = st.radio(
                "¿Qué le ocurre al valor presente cuando aumenta la tasa de interés?",
                ["Aumenta", "Disminuye", "Permanece igual", "No se puede determinar"],
                key="quiz_q1"
            )
        with col2:
            q2 = st.radio(
                "¿Cuál tiene mayor valor futuro a 8% anual: $1000 capitalizado anual o trimestralmente?",
                ["Anual", "Trimestral", "Igual", "No se puede determinar"],
                key="quiz_q2"
            )
        q3 = st.radio(
            "El valor presente de una anualidad _______ a medida que aumenta el número de pagos.",
            ["Siempre aumenta", "Siempre disminuye", "Aumenta y luego se estabiliza", "Permanece constante"],
            key="quiz_q3"
        )
        if st.button("Enviar respuestas", key="quiz_submit"):
            score = 0
            if q1 == "Disminuye":
                st.success("P1: ¡Correcto! ✅")
                score += 1
            else:
                st.error("P1: Incorrecto. El VP disminuye cuando aumenta la tasa.")
            if q2 == "Trimestral":
                st.success("P2: ¡Correcto! ✅")
                score += 1
            else:
                st.error("P2: Incorrecto. Mayor frecuencia de capitalización ⇒ mayor VF.")
            if q3 == "Aumenta y luego se estabiliza":
                st.success("P3: ¡Correcto! ✅")
                score += 1
            else:
                st.error("P3: Incorrecto. El VP aumenta con más pagos, pero a una tasa decreciente.")
            
            st.write(f"Tu puntaje: {score}/3")
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Clase 2: Valor del Dinero en el Tiempo")
                progress_tracker.set_class_score("Clase 2: Valor del Dinero en el Tiempo", (score/3) * 100)
                st.success("🎉 ¡Felicidades! Has completado la Clase 2.")
    
    # -------------------------------
    # MATERIALES DESCARGABLES (Sidebar)
    # -------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Materiales de la clase")
    
    code_example = """
# Cálculos de Valor del Dinero en el Tiempo (VDT) en Python

import numpy as np

def valor_presente(vf, tasa, periodos):
    \"\"\"Calcula el valor presente\"\"\"
    return vf / (1 + tasa) ** periodos

def valor_futuro(vp, tasa, periodos):
    \"\"\"Calcula el valor futuro\"\"\"
    return vp * (1 + tasa) ** periodos

def vp_anualidad(pmt, tasa, periodos):
    \"\"\"Valor presente de una anualidad\"\"\"
    return pmt * (1 - (1 + tasa) ** -periodos) / tasa

def vf_anualidad(pmt, tasa, periodos):
    \"\"\"Valor futuro de una anualidad\"\"\"
    if tasa == 0:
        return pmt * periodos
    return pmt * ((1 + tasa) ** periodos - 1) / tasa

# Ejemplos de uso
vp = valor_presente(1000, 0.05, 10)   # ≈ 613.91
vf = valor_futuro(1000, 0.05, 10)     # ≈ 1628.89
vp_a = vp_anualidad(100, 0.05, 10)    # ≈ 772.17
vf_a = vf_anualidad(100, 0.05, 10)    # ≈ 1257.79
"""
    st.sidebar.download_button(
        label="💻 Descargar código Python",
        data=code_example,
        file_name="calculos_vdt.py",
        mime="text/python"
    )
