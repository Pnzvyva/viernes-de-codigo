import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 3: Valoración de Bonos")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Fundamentos de Bonos",
        "Modelos de Valoración",
        "Precio Interactivo",
        "Análisis de Rendimiento"
    ])
    
    # ===========================
    # TAB 1: Fundamentos
    # ===========================
    with tab1:
        st.subheader("Entendiendo los Bonos")
        
        st.write("""
        Un bono es un título de deuda que representa un préstamo hecho por un inversionista a un emisor. 
        Cuando compras un bono, prestas dinero al emisor a cambio de pagos periódicos de intereses (cupones) 
        y la devolución del valor nominal al vencimiento.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            **Términos clave del bono:**
            - **Valor nominal (par)**: Monto devuelto al vencimiento.
            - **Tasa cupón**: Tasa de interés anual del bono.
            - **Vencimiento**: Fecha en que expira el bono.
            - **Rendimiento (yield)**: Retorno de la inversión.
            - **Duración**: Sensibilidad del precio ante cambios en tasas.
            """)
        with col2:
            st.write("""
            **Tipos de bonos:**
            - Bonos del gobierno
            - Bonos corporativos
            - Bonos municipales
            - Bonos cupón cero
            - Bonos convertibles
            """)
        
        st.subheader("Fundamentos del precio de un bono")
        st.write("""
        El precio de un bono se determina descontando a valor presente sus flujos futuros (cupones y principal) 
        usando la tasa de mercado.
        """)
        with st.expander("📊 Fórmula básica de precio de bono"):
            st.latex(r"P = \sum_{t=1}^{n} \frac{C}{(1+r)^t} + \frac{FV}{(1+r)^n}")
            st.write("Donde: P = Precio, C = Pago de cupón, r = rendimiento, n = períodos, FV = Valor nominal")
        with st.expander("🎯 Relación entre precio y rendimiento"):
            st.write("""
            - **Relación inversa**: Si suben los rendimientos, bajan los precios.
            - **A la par**: Precio = Valor nominal cuando cupón = rendimiento.
            - **Prima**: Precio > Valor nominal si cupón > rendimiento.
            - **Descuento**: Precio < Valor nominal si cupón < rendimiento.
            """)
    
    # ===========================
    # TAB 2: Modelos de Valoración
    # ===========================
    with tab2:
        st.subheader("Modelos de Valoración de Bonos")
        
        model_type = st.selectbox(
            "Selecciona el modelo:",
            ["Bono estándar", "Bono cupón cero", "Bono perpetuo"]
        )
        
        # ---- Bono estándar (con frecuencia ajustable)
        if model_type == "Bono estándar":
            st.write("**Valoración de bono con cupones estándar**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                face_value = st.number_input("Valor nominal ($)", value=1000.0, min_value=100.0)
                coupon_rate = st.number_input("Tasa cupón (%)", value=5.0, min_value=0.0, max_value=20.0)
            with col2:
                market_rate = st.number_input("Rendimiento de mercado (%)", value=6.0, min_value=0.0, max_value=20.0)
                periods = st.number_input("Años al vencimiento", value=10, min_value=1, max_value=50)
            with col3:
                frequency = st.selectbox("Frecuencia de pago", ["Anual", "Semestral"], key="bond_freq")
                
            if st.button("Calcular precio del bono"):
                # Ajuste por frecuencia
                if frequency == "Semestral":
                    coupon_rate_adj = coupon_rate / 2
                    market_rate_adj = market_rate / 2
                    periods_adj = periods * 2
                else:
                    coupon_rate_adj = coupon_rate
                    market_rate_adj = market_rate
                    periods_adj = periods
                
                bond_price = FinancialCalculations.bond_price(
                    face_value, coupon_rate_adj/100.0, market_rate_adj/100.0, periods_adj
                )
                
                st.success(f"Precio del bono: ${bond_price:.2f}")
                
                # Análisis prima/descuento
                if bond_price > face_value:
                    premium = bond_price - face_value
                    st.info(f"Negociándose con **PRIMA**: ${premium:.2f} por encima del par")
                elif bond_price < face_value:
                    discount = face_value - bond_price
                    st.info(f"Negociándose con **DESCUENTO**: ${discount:.2f} por debajo del par")
                else:
                    st.info("Negociándose **A LA PAR**")
                
                # Flujo de caja
                st.subheader("Análisis de flujos de caja")
                coupon_payment = face_value * (coupon_rate_adj/100.0)
                periods_range = range(1, periods_adj + 1)
                
                cash_flows = []
                for period in periods_range:
                    if period == periods_adj:
                        cf = coupon_payment + face_value
                    else:
                        cf = coupon_payment
                    pv_cf = cf / (1 + market_rate_adj/100.0) ** period
                    cash_flows.append({
                        'Período': period,
                        'Flujo de caja': cf,
                        'Valor presente': pv_cf
                    })
                
                df_cf = pd.DataFrame(cash_flows)
                if len(df_cf) > 10:
                    st.write("**Primeros 5 períodos:**")
                    st.dataframe(df_cf.head(), use_container_width=True)
                    st.write("**Últimos 5 períodos:**")
                    st.dataframe(df_cf.tail(), use_container_width=True)
                else:
                    st.dataframe(df_cf, use_container_width=True)
        
        # ---- Bono cupón cero
        elif model_type == "Bono cupón cero":
            st.write("**Valoración de bono cupón cero**")
            st.write("Un bono cupón cero no paga intereses; se vende con descuento y paga el valor nominal al vencimiento.")
            
            col1, col2 = st.columns(2)
            with col1:
                face_value = st.number_input("Valor nominal ($)", value=1000.0, min_value=100.0, key="zcb_fv")
                market_rate = st.number_input("Rendimiento de mercado (%)", value=6.0, min_value=0.0, max_value=20.0, key="zcb_rate")
            with col2:
                periods = st.number_input("Años al vencimiento", value=10, min_value=1, max_value=50, key="zcb_periods")
            
            if st.button("Calcular precio cupón cero"):
                zcb_price = FinancialCalculations.present_value(face_value, market_rate/100.0, periods)
                st.success(f"Precio del bono cupón cero: ${zcb_price:.2f}")
                
                discount = face_value - zcb_price
                yield_pct = (face_value / zcb_price) ** (1/periods) - 1 if zcb_price > 0 else np.nan
                st.info(f"Descuento frente al par: ${discount:.2f}")
                st.info(f"Rendimiento anual implícito: {yield_pct*100:.2f}%")
        
        # ---- Bono perpetuo (COMPLETADO)
        elif model_type == "Bono perpetuo":
            st.write("**Valoración de bono perpetuo (consol)**")
            st.write("Un bono perpetuo no tiene vencimiento: su precio es el valor presente de una renta perpetua de cupones.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                face_value = st.number_input("Valor nominal de referencia ($)", value=1000.0, min_value=100.0, key="perp_fv")
            with col2:
                coupon_rate = st.number_input("Tasa cupón (%)", value=5.0, min_value=0.0, max_value=30.0, key="perp_coupon")
            with col3:
                market_rate = st.number_input("Rendimiento de mercado (%)", value=6.0, min_value=0.01, max_value=50.0, key="perp_yield")
                frequency = st.selectbox("Frecuencia de pago", ["Anual", "Semestral"], key="perp_freq")
            
            if st.button("Calcular precio bono perpetuo"):
                # Ajuste por frecuencia: trabajamos en tasa y cupón por período
                if frequency == "Semestral":
                    coupon_rate_period = coupon_rate / 2.0
                    market_rate_period = market_rate / 2.0
                else:
                    coupon_rate_period = coupon_rate
                    market_rate_period = market_rate
                
                # Cupón por período en dólares
                coupon_payment = face_value * (coupon_rate_period / 100.0)
                r = market_rate_period / 100.0
                
                if r <= 0:
                    st.error("La tasa de mercado debe ser mayor que 0.")
                else:
                    price_perpetual = coupon_payment / r
                    st.success(f"Precio del bono perpetuo: ${price_perpetual:.2f}")
                    st.caption("Fórmula: Precio = Cupón por período / Tasa de mercado por período")
    
    # ===========================
    # TAB 3: Precio Interactivo
    # ===========================
    with tab3:
        st.subheader("Herramienta interactiva de precios")
        st.write("Explora cómo cambia el precio del bono al variar sus parámetros.")
        
        col1, col2 = st.columns(2)
        with col1:
            base_face_value = st.slider("Valor nominal ($)", 500, 5000, 1000, step=100)
            base_coupon_rate = st.slider("Tasa cupón (%)", 0.0, 15.0, 5.0, step=0.5)
        with col2:
            base_years = st.slider("Años al vencimiento", 1, 30, 10)
            yield_range = st.slider("Rango de rendimiento (±%)", 1.0, 10.0, 5.0, step=0.5)
        
        if st.button("Generar curva Precio–Rendimiento"):
            base_yield = base_coupon_rate
            yields = np.linspace(base_yield - yield_range, base_yield + yield_range, 50)
            prices = []
            for y in yields:
                if y > 0:
                    price = FinancialCalculations.bond_price(
                        base_face_value, base_coupon_rate/100.0, y/100.0, base_years
                    )
                    prices.append(price)
                else:
                    prices.append(np.nan)
            
            df_curve = pd.DataFrame({'Rendimiento (%)': yields, 'Precio del bono ($)': prices})
            fig = px.line(df_curve, x='Rendimiento (%)', y='Precio del bono ($)', title='Curva Precio vs. Rendimiento')
            fig.add_hline(y=base_face_value, line_dash="dash", line_color="red", annotation_text="Valor nominal")
            fig.add_vline(x=base_coupon_rate, line_dash="dash", line_color="green", annotation_text="Tasa cupón")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Ideas clave")
            col1, col2 = st.columns(2)
            with col1:
                current_price = FinancialCalculations.bond_price(
                    base_face_value, base_coupon_rate/100.0, base_coupon_rate/100.0, base_years
                )
                st.metric("Precio a la tasa cupón", f"${current_price:.2f}")
                
                high_yield_price = FinancialCalculations.bond_price(
                    base_face_value, base_coupon_rate/100.0, (base_coupon_rate + 2)/100.0, base_years
                )
                st.metric("Precio con +2% en rendimiento", f"${high_yield_price:.2f}")
            with col2:
                low_yield_price = FinancialCalculations.bond_price(
                    base_face_value, base_coupon_rate/100.0, max(0.1, base_coupon_rate - 2)/100.0, base_years
                )
                st.metric("Precio con -2% en rendimiento", f"${low_yield_price:.2f}")
                price_change = ((high_yield_price - low_yield_price) / current_price) * 100
                st.metric("Sensibilidad de precio", f"{price_change:.1f}%")
    
    # ===========================
    # TAB 4: Análisis de Rendimiento (incluye YTC COMPLETO)
    # ===========================
    with tab4:
        st.subheader("Análisis de Rendimiento")
        st.write("Comprende diferentes medidas de rendimiento del bono.")
        
        yield_type = st.selectbox(
            "Selecciona la medida de rendimiento:",
            ["Rendimiento actual", "Rendimiento al vencimiento (YTM)", "Rendimiento al rescate (YTC)"]
        )
        
        # ---- Rendimiento actual
        if yield_type == "Rendimiento actual":
            st.write("**Rendimiento actual = Cupón anual / Precio actual**")
            col1, col2 = st.columns(2)
            with col1:
                annual_coupon = st.number_input("Cupón anual ($)", value=50.0, min_value=0.0)
            with col2:
                current_price = st.number_input("Precio actual del bono ($)", value=950.0, min_value=0.01)
            if st.button("Calcular rendimiento actual"):
                current_yield = (annual_coupon / current_price) * 100
                st.success(f"Rendimiento actual: {current_yield:.2f}%")
                st.caption("Nota: no incorpora la ganancia/pérdida de capital al vencimiento.")
        
        # ---- YTM
        elif yield_type == "Rendimiento al vencimiento (YTM)":
            st.write("**YTM: Tasa interna de retorno de todos los flujos hasta el vencimiento**")
            col1, col2 = st.columns(2)
            with col1:
                bond_price = st.number_input("Precio actual del bono ($)", value=950.0, min_value=0.01, key="ytm_price")
                face_value = st.number_input("Valor nominal ($)", value=1000.0, min_value=0.01, key="ytm_fv")
            with col2:
                annual_coupon = st.number_input("Cupón anual ($)", value=50.0, min_value=0.0, key="ytm_coupon")
                years = st.number_input("Años al vencimiento", value=10, min_value=1, key="ytm_years")
            
            if st.button("Estimar YTM"):
                # Aproximación
                numerator = annual_coupon + (face_value - bond_price) / years
                denominator = (face_value + bond_price) / 2
                ytm_approx = (numerator / denominator) * 100
                st.success(f"YTM aproximado: {ytm_approx:.2f}%")
                
                # Preciso por bisección
                def ytm_function(rate):
                    price = 0.0
                    for t in range(1, years + 1):
                        cf = annual_coupon if t < years else annual_coupon + face_value
                        price += cf / (1 + rate) ** t
                    return price - bond_price
                
                low, high = 1e-6, 1.0
                mid = (low + high) / 2
                for _ in range(100):
                    mid = (low + high) / 2
                    fmid = ytm_function(mid)
                    if abs(fmid) < 1e-4:
                        break
                    flow = ytm_function(low)
                    if np.sign(flow) == np.sign(fmid):
                        low = mid
                    else:
                        high = mid
                precise_ytm = mid * 100
                st.info(f"YTM preciso: {precise_ytm:.3f}%")
        
        # ---- YTC (COMPLETADO)
        elif yield_type == "Rendimiento al rescate (YTC)":
            st.write("**YTC: Tasa interna de retorno hasta la primera fecha de rescate (call)**")
            st.caption("Descuenta cupones hasta la fecha de call y el **precio de rescate** en dicha fecha.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                bond_price = st.number_input("Precio actual del bono ($)", value=1020.0, min_value=0.01, key="ytc_price")
                face_value = st.number_input("Valor nominal ($)", value=1000.0, min_value=0.01, key="ytc_fv")
            with col2:
                annual_coupon = st.number_input("Cupón anual ($)", value=60.0, min_value=0.0, key="ytc_coupon")
                years_to_call = st.number_input("Años hasta el call", value=5, min_value=1, max_value=50, key="ytc_years")
            with col3:
                call_price = st.number_input("Precio de rescate (call) ($)", value=1000.0, min_value=0.01, key="ytc_call_price")
                frequency = st.selectbox("Frecuencia de cupón", ["Anual", "Semestral"], key="ytc_freq")
            
            if st.button("Calcular YTC"):
                # Ajuste por frecuencia
                if frequency == "Semestral":
                    periods = int(2 * years_to_call)
                    coupon_per_period = annual_coupon / 2.0
                else:
                    periods = int(years_to_call)
                    coupon_per_period = annual_coupon
                
                # Función objetivo: PV de cupones + call_price al período 'periods' - precio actual
                def ytc_function(rate_period):
                    pv = 0.0
                    for t in range(1, periods + 1):
                        cf = coupon_per_period if t < periods else (coupon_per_period + call_price)
                        pv += cf / ((1 + rate_period) ** t)
                    return pv - bond_price
                
                # Búsqueda por bisección en tasa por período
                low, high = 1e-6, 1.0  # 0.000001 a 100% por período
                mid = (low + high) / 2.0
                for _ in range(120):
                    mid = (low + high) / 2.0
                    fmid = ytc_function(mid)
                    if abs(fmid) < 1e-4:
                        break
                    flow = ytc_function(low)
                    if np.sign(flow) == np.sign(fmid):
                        low = mid
                    else:
                        high = mid
                
                # Convertimos a tasa efectiva anual según la frecuencia
                if frequency == "Semestral":
                    annualized_ytc = (1 + mid) ** 2 - 1
                else:
                    annualized_ytc = mid  # ya es anual
                st.success(f"YTC (anualizado): {annualized_ytc*100:.3f}%")
                
                # Nota y comprobación
                st.caption("Comprobación: el YTC suele ser relevante si el bono se negocia con prima y es rescatable antes del vencimiento.")
        
        # ===========================
        # Problemas de práctica
        # ===========================
        st.subheader("Problema de práctica")
        problem = st.selectbox("Elige el problema:", ["Problema 1: Precio de bono"])
        
        if problem == "Problema 1: Precio de bono":
            st.write("""
            **Problema 1:**
            Un bono con valor nominal de $1.000 paga cupón anual del 6% y vence en 8 años.
            Si el rendimiento de mercado es 7%, ¿cuál es su precio?
            """)
            user_answer = st.number_input("Tu respuesta ($):", min_value=0.0, format="%.2f", key="bond_p1")
            if st.button("Revisar respuesta", key="check_bond_p1"):
                correct_answer = FinancialCalculations.bond_price(1000, 0.06, 0.07, 8)
                if abs(user_answer - correct_answer) < 5.0:
                    st.success(f"✅ ¡Correcto! La respuesta es ${correct_answer:.2f}")
                    st.write("El bono se negocia con **descuento** porque el rendimiento (7%) > cupón (6%).")
                else:
                    st.error(f"❌ Incorrecto. La respuesta correcta es ${correct_answer:.2f}")
        
        # ===========================
        # Evaluación final
        # ===========================
        st.subheader("Evaluación de la clase")
        q1 = st.radio(
            "Cuando suben las tasas de interés de mercado, los precios de los bonos:",
            ["Suben", "Bajan", "Permanecen iguales", "Se vuelven más volátiles"],
            key="bond_q1"
        )
        q2 = st.radio(
            "Un bono que cotiza sobre la par tiene:",
            ["Tasa cupón > Rendimiento de mercado", "Tasa cupón < Rendimiento de mercado", 
             "Tasa cupón = Rendimiento de mercado", "Rendimiento negativo"],
            key="bond_q2"
        )
        q3 = st.radio(
            "La duración mide:",
            ["Tiempo al vencimiento", "Sensibilidad del precio a tasas de interés", 
             "Riesgo de crédito", "Riesgo de liquidez"],
            key="bond_q3"
        )
        if st.button("Enviar evaluación"):
            score = 0
            if q1 == "Bajan":
                st.success("P1: ¡Correcto! ✅"); score += 1
            else:
                st.error("P1: Incorrecto. Los precios bajan cuando suben las tasas.")
            if q2 == "Tasa cupón > Rendimiento de mercado":
                st.success("P2: ¡Correcto! ✅"); score += 1
            else:
                st.error("P2: Incorrecto. Un bono con prima tiene cupón mayor al rendimiento.")
            if q3 == "Sensibilidad del precio a tasas de interés":
                st.success("P3: ¡Correcto! ✅"); score += 1
            else:
                st.error("P3: Incorrecto. La duración mide sensibilidad a tasas.")
            
            st.write(f"Tu puntaje: {score}/3")
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Class 3: Bond Valuation")
                progress_tracker.set_class_score("Class 3: Bond Valuation", (score/3) * 100)
                st.success("🎉 ¡Felicitaciones! ¡Has completado la Clase 3!")
    
    # ===========================
    # Material de descarga
    # ===========================
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Recursos sobre bonos")
    
    bond_code = """
# Valoración básica de bonos (ejemplo didáctico)

def bond_price(face_value, coupon_rate, market_rate, periods):
    \"\"\"Calcula el precio de un bono con cupones nivelados (todas las tasas en decimales).\"\"\"
    coupon = face_value * coupon_rate
    if market_rate == 0:
        # Si r = 0, el precio es la suma de cupones + principal sin descuento
        return coupon * periods + face_value
    pv_coupons = coupon * (1 - (1 + market_rate) ** -periods) / market_rate
    pv_face = face_value / (1 + market_rate) ** periods
    return pv_coupons + pv_face

# Ejemplo:
# price = bond_price(1000, 0.06, 0.07, 8)  # ≈ 944.72
"""
    st.sidebar.download_button(
        label="💻 Descargar código de valoración",
        data=bond_code,
        file_name="valoracion_bonos.py",
        mime="text/python"
    )
