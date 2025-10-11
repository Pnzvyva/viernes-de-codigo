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
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Fundamentos de Opciones",
        "Modelo Black-Scholes",
        "Griegas de Opciones",
        "Estrategias"
    ])
    
    # -----------------------------------------------------------
    # TAB 1: Fundamentos
    # -----------------------------------------------------------
    with tab1:
        st.subheader("Comprendiendo las Opciones")
        
        st.write("""
        Las opciones son instrumentos financieros derivados que otorgan al comprador el derecho, 
        pero no la obligación, de **comprar (call)** o **vender (put)** un activo subyacente 
        a un precio determinado (precio de ejercicio o strike) antes o en la fecha de vencimiento.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Opciones Call:**
            - Derecho a **comprar** al precio strike
            - Beneficio cuando S > K
            - Pérdida limitada (prima)
            - Potencial de ganancia ilimitado
            - Estrategia alcista
            """)
        
        with col2:
            st.write("""
            **Opciones Put:**
            - Derecho a **vender** al precio strike
            - Beneficio cuando S < K
            - Pérdida limitada (prima)
            - Ganancia máxima limitada (K - prima)
            - Estrategia bajista
            """)
        
        st.subheader("Diagramas de Pago de Opciones")
        
        option_type = st.selectbox("Selecciona el tipo de opción:", ["Call Option", "Put Option"])
        
        col1, col2 = st.columns(2)
        with col1:
            strike_price = st.slider("Precio de Ejercicio ($)", 80, 120, 100)
            premium = st.slider("Prima de la Opción ($)", 1, 20, 5)
        with col2:
            position = st.radio("Posición:", ["Long (Buy)", "Short (Sell)"])
        
        # Generar diagrama
        spot_range = np.linspace(60, 140, 100)
        
        if option_type == "Call Option":
            if position == "Long (Buy)":
                payoffs = np.maximum(spot_range - strike_price, 0) - premium
                title = f"Call Larga (Strike: ${strike_price}, Prima: ${premium})"
            else:
                payoffs = premium - np.maximum(spot_range - strike_price, 0)
                title = f"Call Corta (Strike: ${strike_price}, Prima: ${premium})"
        else:  # Put Option
            if position == "Long (Buy)":
                payoffs = np.maximum(strike_price - spot_range, 0) - premium
                title = f"Put Larga (Strike: ${strike_price}, Prima: ${premium})"
            else:
                payoffs = premium - np.maximum(strike_price - spot_range, 0)
                title = f"Put Corta (Strike: ${strike_price}, Prima: ${premium})"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=payoffs, mode='lines', name='Pago'))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.add_vline(x=strike_price, line_dash="dot", annotation_text="Precio Strike")
        
        profit_mask = payoffs > 0
        loss_mask = payoffs < 0
        
        fig.add_trace(go.Scatter(x=spot_range[profit_mask], y=payoffs[profit_mask],
                                 fill='tonexty', fillcolor='rgba(0,255,0,0.2)',
                                 mode='none', name='Ganancia', showlegend=False))
        fig.add_trace(go.Scatter(x=spot_range[loss_mask], y=payoffs[loss_mask],
                                 fill='tonexty', fillcolor='rgba(255,0,0,0.2)',
                                 mode='none', name='Pérdida', showlegend=False))
        
        fig.update_layout(title=title, xaxis_title="Precio del Activo al Vencimiento ($)",
                          yaxis_title="Ganancia/Pérdida ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        breakeven = strike_price + premium if "Call" in option_type else strike_price - premium
        if position == "Short (Sell)":
            breakeven = strike_price - premium if "Call" in option_type else strike_price + premium
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precio Strike", f"${strike_price}")
        with col2:
            st.metric("Prima", f"${premium}")
        with col3:
            st.metric("Punto de Equilibrio", f"${breakeven:.2f}")
    
    # -----------------------------------------------------------
    # TAB 2: Black-Scholes
    # -----------------------------------------------------------
    with tab2:
        st.subheader("Modelo de Valoración de Opciones Black-Scholes")
        
        st.write("""
        El modelo de **Black-Scholes** proporciona un marco matemático para valorar opciones europeas.  
        Asume volatilidad constante, tasa libre de riesgo y ausencia de dividendos.
        """)
        
        with st.expander("📊 Fórmulas del Modelo Black-Scholes"):
            st.latex(r"C = S_0 N(d_1) - K e^{-rT} N(d_2)")
            st.latex(r"P = K e^{-rT} N(-d_2) - S_0 N(-d_1)")
            st.write("Donde:")
            st.latex(r"d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}")
            st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")
            st.write("C = Precio del Call, P = Precio del Put, N() = Distribución normal acumulada")
        
        st.subheader("Calculadora Black-Scholes")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            S = st.number_input("Precio actual del activo (S₀) ($)", value=100.0, min_value=0.01)
            K = st.number_input("Precio de ejercicio (K) ($)", value=100.0, min_value=0.01)
        with col2:
            T = st.number_input("Tiempo al vencimiento (T) en años", value=0.25, min_value=0.001, max_value=10.0)
            r = st.number_input("Tasa libre de riesgo (r) %", value=5.0, min_value=0.0, max_value=50.0) / 100
        with col3:
            sigma = st.number_input("Volatilidad (σ) %", value=20.0, min_value=0.1, max_value=200.0) / 100
        
        if st.button("Calcular precios de las opciones"):
            call_price = FinancialCalculations.black_scholes_call(S, K, T, r, sigma)
            put_price = FinancialCalculations.black_scholes_put(S, K, T, r, sigma)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"Precio del Call: ${call_price:.2f}")
                st.info(f"N(d₁) = {norm.cdf(d1):.4f}")
                st.info(f"N(d₂) = {norm.cdf(d2):.4f}")
            with col2:
                st.success(f"Precio del Put: ${put_price:.2f}")
                st.info(f"d₁ = {d1:.4f}")
                st.info(f"d₂ = {d2:.4f}")
            
            parity_check = call_price - put_price - (S - K * np.exp(-r * T))
            st.write(f"**Revisión de Paridad Put-Call:** {parity_check:.6f} (debería ser ≈ 0)")
        
        st.subheader("Análisis de Sensibilidad")
        
        analysis_param = st.selectbox("Selecciona el parámetro a analizar:", 
                                     ["Precio del Activo", "Volatilidad", "Tiempo al Vencimiento"])
        
        if analysis_param == "Precio del Activo":
            spot_range = np.linspace(S * 0.7, S * 1.3, 50)
            call_prices = [FinancialCalculations.black_scholes_call(s, K, T, r, sigma) for s in spot_range]
            put_prices = [FinancialCalculations.black_scholes_put(s, K, T, r, sigma) for s in spot_range]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spot_range, y=call_prices, mode='lines', name='Precio del Call'))
            fig.add_trace(go.Scatter(x=spot_range, y=put_prices, mode='lines', name='Precio del Put'))
            fig.add_vline(x=S, line_dash="dot", annotation_text="Precio Actual")
            fig.add_vline(x=K, line_dash="dash", annotation_text="Precio Strike")
            fig.update_layout(title="Precios de Opciones vs Precio del Activo", 
                              xaxis_title="Precio del Activo ($)", yaxis_title="Precio de la Opción ($)")
            st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------
    # TAB 3: Griegas de Opciones
    # -----------------------------------------------------------
    with tab3:
        st.subheader("Griegas de las Opciones")
        
        st.write("""
        Las **Griegas** miden la sensibilidad del precio de una opción ante cambios en distintas variables.  
        Son herramientas fundamentales para la **gestión del riesgo** y el diseño de **estrategias de cobertura**.
        """)
        
        def calcular_griegas(S, K, T, r, sigma, tipo='call'):
            """Calcular las Griegas de una opción"""
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if tipo == 'call':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta (por día)
            if tipo == 'call':
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) -
                        r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) +
                        r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega (por 1% cambio en volatilidad)
            vega = S * np.sqrt(T) * norm.pdf(d1) / 100
            
            # Rho (por 1% cambio en tasa)
            if tipo == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}
        
        st.subheader("Calculadora de Griegas")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            S_g = st.number_input("Precio del Activo ($)", value=100.0, min_value=0.01)
            K_g = st.number_input("Precio Strike ($)", value=100.0, min_value=0.01)
        with col2:
            T_g = st.number_input("Tiempo al Vencimiento (años)", value=0.25, min_value=0.001)
            r_g = st.number_input("Tasa Libre de Riesgo (%)", value=5.0, min_value=0.0) / 100
        with col3:
            sigma_g = st.number_input("Volatilidad (%)", value=20.0, min_value=0.1) / 100
        
        tipo_opcion = st.radio("Tipo de Opción:", ["Call", "Put"])
        
        if st.button("Calcular Griegas"):
            g_call = calcular_griegas(S_g, K_g, T_g, r_g, sigma_g, 'call')
            g_put = calcular_griegas(S_g, K_g, T_g, r_g, sigma_g, 'put')
            
            st.subheader("Resultados de las Griegas")
            datos = {
                'Griega': ['Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ)', 'Vega (ν)', 'Rho (ρ)'],
                'Call Option': [f"{g_call['Delta']:.4f}", f"{g_call['Gamma']:.4f}", 
                                f"{g_call['Theta']:.4f}", f"{g_call['Vega']:.4f}", f"{g_call['Rho']:.4f}"],
                'Put Option': [f"{g_put['Delta']:.4f}", f"{g_put['Gamma']:.4f}", 
                               f"{g_put['Theta']:.4f}", f"{g_put['Vega']:.4f}", f"{g_put['Rho']:.4f}"],
                'Interpretación': [
                    'Cambio del precio ante una variación de $1 en el activo',
                    'Cambio del Delta ante una variación de $1 en el activo',
                    'Cambio del precio por día (decadencia temporal)',
                    'Cambio del precio ante 1% de cambio en la volatilidad',
                    'Cambio del precio ante 1% de cambio en la tasa'
                ]
            }
            
            df_g = pd.DataFrame(datos)
            st.dataframe(df_g, use_container_width=True)
    
    # -----------------------------------------------------------
    # TAB 4: Estrategias y Ejercicios
    # -----------------------------------------------------------
    with tab4:
        st.subheader("Estrategias con Opciones")
        
        st.write("""
        Las estrategias con opciones combinan múltiples posiciones para crear perfiles específicos de riesgo y retorno.  
        A continuación exploraremos algunas de las más comunes.
        """)
        
        strategy = st.selectbox("Selecciona una estrategia:", 
                               ["Covered Call", "Protective Put", "Bull Call Spread", "Iron Condor"])
        
        # ---------------- Covered Call ----------------
        if strategy == "Covered Call":
            st.write("**Covered Call Strategy**")
            st.write("Posición larga en acciones + venta de opción call. Genera ingresos pero limita la ganancia máxima.")
            
            col1, col2 = st.columns(2)
            with col1:
                stock_price = st.number_input("Precio de la acción ($)", value=100.0, key="cc_stock")
                shares = st.number_input("Número de acciones", value=100, min_value=1)
            with col2:
                call_strike = st.number_input("Precio Strike del Call ($)", value=105.0)
                call_premium = st.number_input("Prima del Call ($)", value=3.0)
            
            spot_range = np.linspace(stock_price * 0.8, stock_price * 1.3, 100)
            stock_payoffs = (spot_range - stock_price) * shares
            call_payoffs = (call_premium - np.maximum(spot_range - call_strike, 0)) * shares
            total_payoffs = stock_payoffs + call_payoffs
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spot_range, y=stock_payoffs, mode='lines', name='Acción'))
            fig.add_trace(go.Scatter(x=spot_range, y=call_payoffs, mode='lines', name='Short Call'))
            fig.add_trace(go.Scatter(x=spot_range, y=total_payoffs, mode='lines', name='Posición Combinada', line=dict(width=3)))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.update_layout(title="Payoff Estrategia Covered Call", 
                              xaxis_title="Precio al Vencimiento ($)", yaxis_title="Ganancia/Pérdida ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            max_profit = call_premium * shares + max(0, call_strike - stock_price) * shares
            breakeven = stock_price - call_premium

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ganancia Máxima", f"${max_profit:.2f}")
            with col2:
                st.metric("Pérdida Máxima", "Ilimitada")
            with col3:
                st.metric("Punto de Equilibrio", f"${breakeven:.2f}")

        # ---------------- Protective Put ----------------
        elif strategy == "Protective Put":
            st.write("**Protective Put Strategy**")
            st.write("Posición larga en acciones + compra de opción put. Protege ante caídas manteniendo el potencial alcista.")
            
            col1, col2 = st.columns(2)
            with col1:
                stock_price = st.number_input("Precio de la acción ($)", value=100.0, key="pp_stock")
                shares = st.number_input("Número de acciones", value=100, min_value=1)
            with col2:
                put_strike = st.number_input("Precio Strike del Put ($)", value=95.0)
                put_premium = st.number_input("Prima del Put ($)", value=4.0)
            
            spot_range = np.linspace(stock_price * 0.7, stock_price * 1.3, 100)
            stock_payoffs = (spot_range - stock_price) * shares
            put_payoffs = (np.maximum(put_strike - spot_range, 0) - put_premium) * shares
            total_payoffs = stock_payoffs + put_payoffs
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spot_range, y=stock_payoffs, mode='lines', name='Acción'))
            fig.add_trace(go.Scatter(x=spot_range, y=put_payoffs, mode='lines', name='Long Put'))
            fig.add_trace(go.Scatter(x=spot_range, y=total_payoffs, mode='lines', name='Posición Combinada', line=dict(width=3)))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.update_layout(title="Payoff Estrategia Protective Put",
                              xaxis_title="Precio al Vencimiento ($)", yaxis_title="Ganancia/Pérdida ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            max_loss = (stock_price - put_strike + put_premium) * shares
            breakeven = stock_price + put_premium

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ganancia Máxima", "Ilimitada")
            with col2:
                st.metric("Pérdida Máxima", f"${max_loss:.2f}")
            with col3:
                st.metric("Punto de Equilibrio", f"${breakeven:.2f}")

        # ---------------- Bull Call Spread ----------------
        elif strategy == "Bull Call Spread":
            st.write("**Bull Call Spread Strategy**")
            st.write("Compra de call con strike bajo + venta de call con strike alto. Riesgo y ganancia limitados.")
            
            col1, col2 = st.columns(2)
            with col1:
                long_strike = st.number_input("Strike Call Largo ($)", value=95.0)
                long_premium = st.number_input("Prima Call Largo ($)", value=6.0)
            with col2:
                short_strike = st.number_input("Strike Call Corto ($)", value=105.0)
                short_premium = st.number_input("Prima Call Corto ($)", value=2.0)
            
            net_debit = long_premium - short_premium
            st.info(f"Débito Neto: ${net_debit:.2f}")
            
            spot_range = np.linspace(80, 120, 100)
            long_call_payoffs = np.maximum(spot_range - long_strike, 0) - long_premium
            short_call_payoffs = short_premium - np.maximum(spot_range - short_strike, 0)
            spread_payoffs = long_call_payoffs + short_call_payoffs
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spot_range, y=spread_payoffs, mode='lines', name='Bull Call Spread', line=dict(width=3)))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.update_layout(title="Payoff Estrategia Bull Call Spread", 
                              xaxis_title="Precio al Vencimiento ($)", yaxis_title="Ganancia/Pérdida ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            max_profit = short_strike - long_strike - net_debit
            max_loss = net_debit
            breakeven = long_strike + net_debit

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ganancia Máxima", f"${max_profit:.2f}")
            with col2:
                st.metric("Pérdida Máxima", f"${max_loss:.2f}")
            with col3:
                st.metric("Punto de Equilibrio", f"${breakeven:.2f}")

        # ---------------- Iron Condor ----------------
        elif strategy == "Iron Condor":
            st.write("**Iron Condor Strategy**")
            st.write("""
            Combinación de un Bull Put Spread y un Bear Call Spread.  
            Gana cuando el precio se mantiene dentro de un rango.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                short_put_strike = st.number_input("Strike Put Corto ($)", value=95.0)
                long_put_strike = st.number_input("Strike Put Largo ($)", value=90.0)
                put_credit = st.number_input("Crédito Neto Put ($)", value=1.5)
            with col2:
                short_call_strike = st.number_input("Strike Call Corto ($)", value=105.0)
                long_call_strike = st.number_input("Strike Call Largo ($)", value=110.0)
                call_credit = st.number_input("Crédito Neto Call ($)", value=1.5)
            
            total_credit = put_credit + call_credit
            st.info(f"Crédito Neto Total: ${total_credit:.2f}")
            
            spot_range = np.linspace(80, 120, 200)
            short_put_payoff = np.minimum(0, -(short_put_strike - spot_range))
            long_put_payoff = np.maximum(long_put_strike - spot_range, 0)
            short_call_payoff = np.minimum(0, -(spot_range - short_call_strike))
            long_call_payoff = np.maximum(spot_range - long_call_strike, 0)
            
            total_payoff = (total_credit 
                            + short_put_payoff - long_put_payoff 
                            + short_call_payoff - long_call_payoff)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spot_range, y=total_payoff, mode='lines', name='Iron Condor Payoff', line=dict(width=3)))
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.add_vrect(x0=short_put_strike, x1=short_call_strike, fillcolor="green", opacity=0.1,
                          annotation_text="Zona de Ganancia")
            
            fig.update_layout(title="Payoff Estrategia Iron Condor",
                              xaxis_title="Precio al Vencimiento ($)", yaxis_title="Ganancia/Pérdida ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            max_profit = total_credit
            max_loss = (short_call_strike - long_call_strike) - total_credit
            breakeven_low = short_put_strike - total_credit
            breakeven_high = short_call_strike + total_credit

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Ganancia Máxima", f"${max_profit:.2f}")
            with col2:
                st.metric("Pérdida Máxima", f"${max_loss:.2f}")
            with col3:
                st.metric("Equilibrio Inferior", f"${breakeven_low:.2f}")
            with col4:
                st.metric("Equilibrio Superior", f"${breakeven_high:.2f}")
    
    # -----------------------------------------------------------
    # Recursos Descargables
    # -----------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Recursos sobre Opciones")
    
    options_code = """
# Cálculo de Opciones y Griegas - Ejemplo en Python

import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def option_delta(S, K, T, r, sigma, tipo='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if tipo == 'call' else norm.cdf(d1) - 1
"""
    
    st.sidebar.download_button(
        label="💻 Descargar código de opciones",
        data=options_code,
        file_name="valoracion_opciones.py",
        mime="text/python"
    )
