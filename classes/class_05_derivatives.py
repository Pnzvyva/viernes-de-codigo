import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 5: Introducción a los Derivados")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Fundamentos de Derivados", "Forwards y Futuros", "Swaps", "Aplicaciones"])
    
    # --- TAB 1: Fundamentos ---
    with tab1:
        st.subheader("Entendiendo los Derivados")
        st.write("""
        Los derivados son instrumentos financieros cuyo valor se deriva de un activo subyacente.
        Se utilizan para la cobertura de riesgos, la especulación y las oportunidades de arbitraje.
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            **Principales Tipos de Derivados:**
            - **Forwards**: Contratos personalizados fuera de bolsa (OTC)
            - **Futuros**: Contratos estandarizados negociados en bolsa
            - **Opciones**: Derechos, pero no obligaciones
            - **Swaps**: Intercambio de flujos de efectivo
            """)
        with col2:
            st.write("""
            **Características Clave:**
            - Efecto de apalancamiento
            - Bajo costo inicial
            - Basados en activos subyacentes
            - Usados para cobertura o especulación
            """)

        st.subheader("Mercados de Derivados")

        market_comparison = pd.DataFrame({
            'Característica': ['Estandarización', 'Lugar de Negociación', 'Riesgo de Contraparte', 'Personalización', 'Liquidez'],
            'Negociados en Bolsa': ['Alta', 'Bolsa Organizada', 'Bajo (Cámara de Compensación)', 'Limitada', 'Alta'],
            'OTC (Over-the-Counter)': ['Baja', 'Bilateral', 'Alta', 'Alta', 'Variable']
        })
        st.dataframe(market_comparison, use_container_width=True)

        st.subheader("Usos de los Derivados")
        usos = {
            "Cobertura (Hedging)": {
                "Descripción": "Reducir o eliminar la exposición al riesgo",
                "Ejemplo": "Una aerolínea se cubre del riesgo del precio del petróleo con futuros de crudo",
                "Beneficio": "Reducción de riesgo y estabilidad de flujos de efectivo"
            },
            "Especulación": {
                "Descripción": "Buscar ganancias aprovechando movimientos de precios mediante apalancamiento",
                "Ejemplo": "Un trader compra opciones call esperando una subida en el precio de la acción",
                "Beneficio": "Retornos amplificados (con mayor riesgo)"
            },
            "Arbitraje": {
                "Descripción": "Aprovechar diferencias de precios entre mercados",
                "Ejemplo": "Comprar futuros subvalorados y vender acciones sobrevaloradas",
                "Beneficio": "Ganancias sin riesgo"
            }
        }

        for uso, detalle in usos.items():
            with st.expander(f"🎯 {uso}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Descripción:** {detalle['Descripción']}")
                    st.write(f"**Ejemplo:** {detalle['Ejemplo']}")
                with col2:
                    st.write(f"**Beneficio:** {detalle['Beneficio']}")

    # --- TAB 2: Forwards y Futuros ---
    with tab2:
        st.subheader("Forwards y Futuros")
        st.write("""
        Los contratos forward y futuros son acuerdos para comprar o vender un activo a un precio determinado 
        en una fecha futura específica. La diferencia principal es que los futuros son estandarizados y se negocian en bolsa.
        """)

        tipo_derivado = st.selectbox("Selecciona el tipo de contrato:", ["Contrato Forward", "Contrato de Futuros"])

        if tipo_derivado == "Contrato Forward":
            st.write("**Valoración de un Contrato Forward**")
            st.latex(r"F_0 = S_0 \times e^{(r-q) \times T}")
            st.write("Donde: F₀ = precio forward, S₀ = precio spot, r = tasa libre de riesgo, q = rendimiento por dividendos, T = tiempo al vencimiento")

            col1, col2, col3 = st.columns(3)
            with col1:
                spot_price = st.number_input("Precio Spot Actual ($)", value=100.0, min_value=0.01)
                risk_free_rate = st.number_input("Tasa Libre de Riesgo (%)", value=5.0, min_value=0.0)
            with col2:
                dividend_yield = st.number_input("Rendimiento por Dividendos (%)", value=2.0, min_value=0.0)
                time_to_maturity = st.number_input("Tiempo al Vencimiento (años)", value=0.25, min_value=0.01, max_value=10.0)
            with col3:
                st.write("")  # Espaciado

            if st.button("Calcular Precio Forward"):
                forward_price = spot_price * np.exp((risk_free_rate/100 - dividend_yield/100) * time_to_maturity)
                st.success(f"Precio Forward Teórico: ${forward_price:.2f}")

                st.write("**Pasos de Cálculo:**")
                st.write(f"F₀ = ${spot_price} × e^(({risk_free_rate/100:.3f} - {dividend_yield/100:.3f}) × {time_to_maturity})")
                st.write(f"F₀ = ${spot_price} × {np.exp((risk_free_rate/100 - dividend_yield/100) * time_to_maturity):.4f}")
                st.write(f"F₀ = ${forward_price:.2f}")

                # Gráfico de Payoff
                st.subheader("Diagramas de Ganancia/Pérdida (Payoff)")
                spot_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
                long_payoff = spot_range - forward_price
                short_payoff = forward_price - spot_range

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spot_range, y=long_payoff, mode='lines', name='Posición Larga'))
                fig.add_trace(go.Scatter(x=spot_range, y=short_payoff, mode='lines', name='Posición Corta'))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.add_vline(x=forward_price, line_dash="dash", line_color="red", annotation_text="Precio Forward")

                fig.update_layout(
                    title="Diagrama de Payoff de un Contrato Forward",
                    xaxis_title="Precio Spot al Vencimiento ($)",
                    yaxis_title="Ganancia/Pérdida ($)"
                )
                st.plotly_chart(fig, use_container_width=True)

        elif tipo_derivado == "Contrato de Futuros":
            st.write("**Análisis de Contratos de Futuros**")
            st.write("""
            Los contratos de futuros son similares a los forwards, pero con diferencias importantes:
            - Liquidación diaria (mark-to-market)
            - Requisitos de margen
            - Términos estandarizados
            - Garantía de cámara de compensación
            """)

            st.subheader("Requerimientos de Margen")
            col1, col2 = st.columns(2)
            with col1:
                contract_size = st.number_input("Tamaño del Contrato", value=100, min_value=1)
                futures_price = st.number_input("Precio del Futuro ($)", value=105.0, min_value=0.01)
                initial_margin = st.number_input("Margen Inicial (%)", value=10.0, min_value=1.0, max_value=50.0)
            with col2:
                maintenance_margin = st.number_input("Margen de Mantenimiento (%)", value=7.5, min_value=1.0, max_value=50.0)
                price_change = st.number_input("Cambio en el Precio ($)", value=-2.0, min_value=-50.0, max_value=50.0)

            if st.button("Calcular Llamada de Margen"):
                contract_value = contract_size * futures_price
                initial_margin_amount = contract_value * (initial_margin / 100)
                maintenance_margin_amount = contract_value * (maintenance_margin / 100)

                new_price = futures_price + price_change
                new_contract_value = contract_size * new_price
                profit_loss = (new_contract_value - contract_value)
                margin_account = initial_margin_amount + profit_loss

                st.subheader("Análisis de Cuenta de Margen")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Valor del Contrato", f"${contract_value:,.2f}")
                    st.metric("Margen Inicial", f"${initial_margin_amount:,.2f}")
                    st.metric("Margen de Mantenimiento", f"${maintenance_margin_amount:,.2f}")
                with col2:
                    st.metric("Cambio de Precio", f"${price_change:.2f}")
                    st.metric("Ganancia/Pérdida", f"${profit_loss:,.2f}")
                    st.metric("Saldo de Cuenta de Margen", f"${margin_account:,.2f}")

                if margin_account < maintenance_margin_amount:
                    margin_call = initial_margin_amount - margin_account
                    st.error(f"⚠️ LLAMADA DE MARGEN: ¡Debes depositar ${margin_call:,.2f} adicionales!")
                else:
                    st.success("✅ No se requiere llamada de margen")

    # --- TAB 3: Swaps ---
    with tab3:
        st.subheader("Swaps de Tasas de Interés")
        st.write("""
        Un swap de tasas de interés es un acuerdo entre dos partes para intercambiar pagos de intereses 
        sobre un monto nominal. El tipo más común es el swap fijo contra flotante.
        """)

        st.subheader("Swap de Tipo Fijo-Vanilla")
        col1, col2 = st.columns(2)
        with col1:
            notional_amount = st.number_input("Monto Nominal ($)", value=1000000.0, min_value=1000.0, format="%.0f")
            fixed_rate = st.number_input("Tasa Fija (%)", value=4.5, min_value=0.0, max_value=20.0)
            swap_tenor = st.number_input("Plazo del Swap (años)", value=5, min_value=1, max_value=30)
        with col2:
            floating_rate_current = st.number_input("Tasa Flotante Actual (%)", value=3.8, min_value=0.0, max_value=20.0)
            payment_frequency = st.selectbox("Frecuencia de Pago", ["Anual", "Semestral", "Trimestral"])

        freq_map = {"Anual": 1, "Semestral": 2, "Trimestral": 4}
        payments_per_year = freq_map[payment_frequency]

        if st.button("Analizar Swap"):
            fixed_payment = (notional_amount * fixed_rate / 100) / payments_per_year
            floating_payment = (notional_amount * floating_rate_current / 100) / payments_per_year
            net_payment = fixed_payment - floating_payment

            st.subheader("Análisis de Pagos del Swap")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pago Fijo", f"${fixed_payment:,.2f}")
            with col2:
                st.metric("Pago Flotante", f"${floating_payment:,.2f}")
            with col3:
                if net_payment > 0:
                    st.metric("Pago Neto (Pagador Fijo)", f"${net_payment:,.2f}", delta="Pagar")
                else:
                    st.metric("Pago Neto (Pagador Fijo)", f"${abs(net_payment):,.2f}", delta="Recibir")

            # Sensibilidad
            st.subheader("Sensibilidad a las Tasas de Interés")
            rate_scenarios = np.linspace(1.0, 8.0, 50)
            net_payments = [(fixed_payment - (notional_amount * rate / 100) / payments_per_year) for rate in rate_scenarios]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rate_scenarios, y=net_payments, mode='lines', name='Pago Neto'))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.add_vline(x=floating_rate_current, line_dash="dot", annotation_text="Tasa Actual")

            fig.update_layout(
                title="Pago Neto del Swap vs Tasa Flotante",
                xaxis_title="Tasa Flotante (%)",
                yaxis_title="Pago Neto ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 4: Aplicaciones ---
    with tab4:
        st.subheader("Aplicaciones y Gestión del Riesgo con Derivados")

        tipo_aplicacion = st.selectbox(
            "Selecciona la aplicación:",
            ["Cobertura de Portafolio", "Cobertura Cambiaria", "Gestión del Riesgo de Tasas de Interés"]
        )

        # --- Cobertura de Portafolio ---
        if tipo_aplicacion == "Cobertura de Portafolio":
            st.write("**Cobertura de un Portafolio de Acciones con Futuros sobre Índices**")

            col1, col2 = st.columns(2)
            with col1:
                valor_portafolio = st.number_input("Valor del Portafolio ($)", value=500000.0, min_value=1000.0)
                beta_portafolio = st.number_input("Beta del Portafolio", value=1.2, min_value=0.1, max_value=3.0)
            with col2:
                precio_futuro_indice = st.number_input("Precio del Futuro sobre Índice", value=3500.0, min_value=100.0)
                multiplicador_futuro = st.number_input("Multiplicador del Futuro", value=250.0, min_value=1.0)

            if st.button("Calcular Ratio de Cobertura"):
                ratio_cobertura = (valor_portafolio * beta_portafolio) / (precio_futuro_indice * multiplicador_futuro)
                contratos_necesarios = round(ratio_cobertura)

                st.success(f"Ratio Óptimo de Cobertura: {ratio_cobertura:.3f}")
                st.success(f"Número de Contratos Futuros Necesarios: {contratos_necesarios}")

                # Simulación de escenarios
                st.subheader("Análisis de Efectividad de la Cobertura")
                cambios_mercado = np.linspace(-20, 20, 41)
                pnl_sin_cobertura = []
                pnl_cobertura = []

                for cambio in cambios_mercado:
                    cambio_portafolio = valor_portafolio * (cambio / 100) * beta_portafolio
                    cambio_precio_futuro = precio_futuro_indice * (cambio / 100)
                    pnl_futuro = -contratos_necesarios * cambio_precio_futuro * multiplicador_futuro  # Posición corta
                    pnl_sin_cobertura.append(cambio_portafolio)
                    pnl_cobertura.append(cambio_portafolio + pnl_futuro)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cambios_mercado, y=pnl_sin_cobertura, mode='lines', name='Portafolio sin Cobertura'))
                fig.add_trace(go.Scatter(x=cambios_mercado, y=pnl_cobertura, mode='lines', name='Portafolio Cubierto'))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(
                    title="Efectividad de la Cobertura del Portafolio",
                    xaxis_title="Cambio en el Mercado (%)",
                    yaxis_title="Ganancia/Pérdida del Portafolio ($)"
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- Cobertura Cambiaria ---
        elif tipo_aplicacion == "Cobertura Cambiaria":
            st.write("**Cobertura con Contratos Forward de Divisas**")

            col1, col2 = st.columns(2)
            with col1:
                monto_exposicion = st.number_input("Exposición en Moneda Extranjera", value=1000000.0, min_value=1000.0)
                tasa_cambio_actual = st.number_input("Tasa de Cambio Actual", value=1.20, min_value=0.01)
                tasa_domestica = st.number_input("Tasa de Interés Doméstica (%)", value=3.0, min_value=0.0)
            with col2:
                tasa_extranjera = st.number_input("Tasa de Interés Extranjera (%)", value=1.5, min_value=0.0)
                horizonte_cobertura = st.number_input("Horizonte de Cobertura (meses)", value=6, min_value=1, max_value=60)

            if st.button("Calcular Cobertura Cambiaria"):
                fraccion_tiempo = horizonte_cobertura / 12
                tasa_forward = tasa_cambio_actual * np.exp((tasa_domestica/100 - tasa_extranjera/100) * fraccion_tiempo)
                st.success(f"Tasa de Cambio Forward: {tasa_forward:.4f}")

                tasas_futuras = np.linspace(tasa_cambio_actual * 0.8, tasa_cambio_actual * 1.2, 50)
                valores_sin_cobertura = [monto_exposicion * tasa for tasa in tasas_futuras]
                valor_cubierto = monto_exposicion * tasa_forward
                valores_cubiertos = [valor_cubierto] * len(tasas_futuras)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=tasas_futuras, y=valores_sin_cobertura, mode='lines', name='Posición sin Cobertura'))
                fig.add_trace(go.Scatter(x=tasas_futuras, y=valores_cubiertos, mode='lines', name='Posición Cubierta'))
                fig.add_vline(x=tasa_cambio_actual, line_dash="dot", annotation_text="Tasa Actual")
                fig.add_vline(x=tasa_forward, line_dash="dash", annotation_text="Tasa Forward")
                fig.update_layout(
                    title="Comparación de Cobertura Cambiaria",
                    xaxis_title="Tasa de Cambio Futura",
                    yaxis_title="Valor en Moneda Local ($)"
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- Gestión de Riesgo de Tasas de Interés ---
        elif tipo_aplicacion == "Gestión del Riesgo de Tasas de Interés":
            st.write("""
            **Uso de Swaps de Tasas de Interés para Gestionar el Riesgo**
            
            Las empresas con préstamos a tasa variable pueden intercambiar sus pagos flotantes por pagos fijos
            mediante un swap, asegurando estabilidad en sus flujos financieros.
            """)
            st.info("Este tipo de cobertura es común entre instituciones financieras y corporaciones con alta exposición a deuda variable.")

        # --- Evaluación de Conocimientos ---
        st.subheader("Evaluación de la Clase")

        q1 = st.radio(
            "¿Cuál derivado tiene mayor riesgo de contraparte?",
            ["Futuros en bolsa", "Contratos Forward", "Opciones listadas", "Swaps estandarizados"],
            key="deriv_q1"
        )
        q2 = st.radio(
            "En un swap de tasas, si las tasas flotantes suben, el pagador de tasa fija:",
            ["Se beneficia", "Pierde", "No se ve afectado", "Debe aumentar su margen"],
            key="deriv_q2"
        )
        q3 = st.radio(
            "Los precios forward son mayores que los spot cuando:",
            ["La tasa de interés > rendimiento por dividendos", "La tasa de interés < rendimiento por dividendos",
             "El activo es muy volátil", "El contrato vence pronto"],
            key="deriv_q3"
        )

        if st.button("Enviar Evaluación"):
            puntaje = 0
            if q1 == "Contratos Forward":
                st.success("✅ Pregunta 1 Correcta: Los forwards son OTC y tienen alto riesgo de contraparte.")
                puntaje += 1
            else:
                st.error("❌ Pregunta 1 Incorrecta: los contratos forward son los de mayor riesgo.")

            if q2 == "Se beneficia":
                st.success("✅ Pregunta 2 Correcta: el pagador fijo se beneficia cuando suben las tasas flotantes.")
                puntaje += 1
            else:
                st.error("❌ Pregunta 2 Incorrecta: el pagador fijo gana si las tasas flotantes aumentan.")

            if q3 == "La tasa de interés > rendimiento por dividendos":
                st.success("✅ Pregunta 3 Correcta: el precio forward aumenta con tasas más altas.")
                puntaje += 1
            else:
                st.error("❌ Pregunta 3 Incorrecta: el forward sube cuando la tasa de interés supera al dividendo.")

            st.write(f"**Tu puntaje: {puntaje}/3**")

            if puntaje >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Clase 5: Introducción a los Derivados")
                progress_tracker.set_class_score("Clase 5: Introducción a los Derivados", (puntaje/3)*100)
                st.success("🎉 ¡Excelente! Has comprendido los fundamentos de los derivados.")

    # --- Recursos y Descarga ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Recursos sobre Derivados")

    codigo_derivados = """
# Modelos de Valoración de Derivados y Gestión del Riesgo

import numpy as np

def precio_forward(precio_spot, tasa_libre, rendimiento_div, tiempo):
    '''Calcula el precio forward teórico'''
    return precio_spot * np.exp((tasa_libre - rendimiento_div) * tiempo)

def tasa_swap_fija(nominal, plazo, tasas_flotantes, tasas_descuento):
    '''Calcula la tasa fija justa en un swap de tasas de interés'''
    vp_flotante = sum([nominal * tasa * np.exp(-t_desc * (i+1)) 
                      for i, (tasa, t_desc) in enumerate(zip(tasas_flotantes, tasas_descuento))])
    anualidad = sum([np.exp(-t_desc * (i+1)) for i, t_desc in enumerate(tasas_descuento)])
    return vp_flotante / (nominal * anualidad)

def ratio_cobertura(valor_portafolio, beta_portafolio, precio_futuro, multiplicador):
    '''Calcula el ratio de cobertura óptimo con futuros'''
    return (valor_portafolio * beta_portafolio) / (precio_futuro * multiplicador)

# Ejemplo
forward_px = precio_forward(100, 0.05, 0.02, 0.25)  # ≈ $100.75
contratos = ratio_cobertura(500000, 1.2, 3500, 250)  # ≈ 0.686 contratos
"""

    st.sidebar.download_button(
        label="💻 Descargar Código de Derivados",
        data=codigo_derivados,
        file_name="valoracion_derivados.py",
        mime="text/python"
    )
