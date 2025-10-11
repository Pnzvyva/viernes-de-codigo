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
        st.subheader("Teoría Moderna de Portafolios (MPT)")
        
        st.write("""
        La Teoría Moderna de Portafolios, desarrollada por Harry Markowitz en 1952, proporciona un marco matemático
        para construir portafolios óptimos que maximizan el retorno esperado para un nivel dado de riesgo.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Supuestos Clave de la MPT:**
            - Los inversionistas son adversos al riesgo
            - Los retornos están normalmente distribuidos
            - Las correlaciones son estables en el tiempo
            - Los mercados son eficientes
            - No existen costos de transacción
            """)
        
        with col2:
            st.write("""
            **Conceptos Fundamentales:**
            - **Diversificación**: reduce el riesgo sin disminuir el retorno esperado
            - **Frontera Eficiente**: conjunto de portafolios óptimos
            - **Relación Riesgo-Retorno**: un mayor retorno requiere asumir mayor riesgo
            - **Correlación**: factor clave de los beneficios de diversificación
            """)
        
        st.subheader("Riesgo y Retorno del Portafolio")
        
        with st.expander("📊 Matemática del Portafolio"):
            st.write("**Retorno Esperado del Portafolio:**")
            st.latex(r"E(R_p) = \sum_{i=1}^{n} w_i E(R_i)")
            
            st.write("**Varianza del Portafolio (Dos Assets):**")
            st.latex(r"\sigma_p^2 = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1w_2\sigma_1\sigma_2\rho_{1,2}")
            
            st.write("**Desviación Estándar del Portafolio:**")
            st.latex(r"\sigma_p = \sqrt{\sigma_p^2}")
            
            st.write("Donde: w = ponderaciones, σ = desviación estándar, ρ = coeficiente de correlación")
        
        # Ejemplo de portafolio de dos activos
        st.subheader("Ejemplo de Portafolio con Dos Activos")
        
        col1, col2 = st.columns(2)
        with col1:
            return1 = st.slider("Retorno Esperado del Activo 1 (%)", 0.0, 20.0, 10.0, 0.5)
            risk1 = st.slider("Riesgo del Activo 1 (Desv. Est.) (%)", 1.0, 50.0, 15.0, 1.0)
            weight1 = st.slider("Ponderación del Activo 1 (%)", 0.0, 100.0, 60.0, 5.0)
        with col2:
            return2 = st.slider("Retorno Esperado del Activo 2 (%)", 0.0, 20.0, 8.0, 0.5)
            risk2 = st.slider("Riesgo del Activo 2 (Desv. Est.) (%)", 1.0, 50.0, 20.0, 1.0)
            correlation = st.slider("Correlación", -1.0, 1.0, 0.3, 0.1)
        
        weight2 = 100.0 - weight1
        
        # Cálculo de métricas del portafolio
        portfolio_return = (weight1/100) * return1 + (weight2/100) * return2
        portfolio_variance = ((weight1/100)**2 * (risk1/100)**2 + 
                            (weight2/100)**2 * (risk2/100)**2 + 
                            2 * (weight1/100) * (weight2/100) * (risk1/100) * (risk2/100) * correlation)
        portfolio_risk = np.sqrt(portfolio_variance) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Retorno del Portafolio", f"{portfolio_return:.2f}%")
        with col2:
            st.metric("Riesgo del Portafolio", f"{portfolio_risk:.2f}%")
        with col3:
            sharpe_ratio = (portfolio_return - 3.0) / portfolio_risk  # tasa libre de riesgo 3%
            st.metric("Ratio de Sharpe", f"{sharpe_ratio:.3f}")
        
        # Mostrar beneficio de diversificación
        weighted_avg_risk = (weight1/100) * risk1 + (weight2/100) * risk2
        diversification_benefit = weighted_avg_risk - portfolio_risk
        
        if diversification_benefit > 0:
            st.success(f"✅ Beneficio de diversificación: reducción de riesgo de {diversification_benefit:.2f}%")
        else:
            st.warning("⚠️ No hay beneficio de diversificación (correlación demasiado alta)")
    
    with tab2:
        st.subheader("Frontera Eficiente")
        
        st.write("""
        La frontera eficiente representa el conjunto de portafolios que ofrecen el mayor retorno esperado
        para cada nivel de riesgo, o el menor riesgo posible para un nivel de retorno determinado.
        """)
        
        if st.button("Generar Frontera Eficiente"):
            assets = ['Acción A', 'Acción B', 'Acción C', 'Bono']
            expected_returns = np.array([0.12, 0.10, 0.08, 0.04])
            
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
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - target_return}
                ]
                
                bounds = tuple((0, 1) for _ in range(n_assets))
                initial_guess = np.array([1/n_assets] * n_assets)
                
                result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
                return result
            
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
                efficient_df = pd.DataFrame(efficient_portfolios, columns=['Riesgo (%)', 'Retorno (%)'])
                
                fig = px.scatter(efficient_df, x='Riesgo (%)', y='Retorno (%)', title='Frontera Eficiente')
                
                individual_risks = [np.sqrt(cov_matrix[i, i]) * 100 for i in range(len(assets))]
                individual_returns = expected_returns * 100
                
                fig.add_trace(go.Scatter(
                    x=individual_risks, 
                    y=individual_returns, 
                    mode='markers+text', 
                    text=assets, 
                    textposition="top center",
                    marker=dict(size=10, color='red'), 
                    name='Activos Individuales'
                ))
                
                risk_free_rate = 4.0
                sharpe_ratios = [(ret - risk_free_rate) / risk for risk, ret in efficient_portfolios if risk > 0]
                if sharpe_ratios:
                    max_sharpe_idx = np.argmax(sharpe_ratios)
                    market_risk = efficient_portfolios[max_sharpe_idx][0]
                    market_return = efficient_portfolios[max_sharpe_idx][1]
                    
                    cml_x = np.linspace(0, max(efficient_df['Riesgo (%)']) * 1.2, 100)
                    cml_y = risk_free_rate + (market_return - risk_free_rate) * cml_x / market_risk
                    
                    fig.add_trace(go.Scatter(
                        x=cml_x, 
                        y=cml_y, 
                        mode='lines',
                        name='Línea del Mercado de Capitales', 
                        line=dict(dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[market_risk], 
                        y=[market_return],
                        mode='markers', 
                        marker=dict(size=15, color='gold'),
                        name='Portafolio de Mercado'
                    ))
                
                fig.update_layout(
                    xaxis_title="Riesgo (Desviación Estándar %)", 
                    yaxis_title="Retorno Esperado (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Información de los Activos / Assets")
                asset_df = pd.DataFrame({
                    'Activo/Asset': assets,
                    'Retorno Esperado (%)': expected_returns * 100,
                    'Riesgo (%)': [np.sqrt(cov_matrix[i, i]) * 100 for i in range(len(assets))]
                })
                st.dataframe(asset_df, use_container_width=True)
                
                st.subheader("Matriz de Correlación")
                std_devs = np.sqrt(np.diag(cov_matrix))
                corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
                corr_df = pd.DataFrame(corr_matrix, index=assets, columns=assets)
                st.dataframe(corr_df.round(3), use_container_width=True)

    with tab3:
        st.subheader("Modelo de Valoración de Activos de Capital (CAPM)")
        
        st.write("""
        El modelo CAPM describe la relación entre el riesgo sistemático y el retorno esperado de los activos,
        especialmente de las acciones. Se utiliza para estimar el costo del capital y evaluar el rendimiento
        de una inversión en función de su riesgo.
        """)
        
        with st.expander("📊 Fórmula del CAPM"):
            st.latex(r"E(R_i) = R_f + \beta_i [E(R_m) - R_f]")
            st.write("Donde:")
            st.write("- E(Rᵢ) = Retorno esperado del activo i")
            st.write("- Rₓ = Tasa libre de riesgo")
            st.write("- βᵢ = Beta del activo i")
            st.write("- E(Rₘ) = Retorno esperado del mercado")
            st.write("- [E(Rₘ) - Rₓ] = Prima de riesgo del mercado")
        
        st.subheader("Calculadora CAPM")
        
        col1, col2 = st.columns(2)
        with col1:
            risk_free_rate = st.number_input("Tasa Libre de Riesgo (%)", value=3.0, min_value=0.0, max_value=20.0)
            market_return = st.number_input("Retorno Esperado del Mercado (%)", value=10.0, min_value=0.0, max_value=30.0)
        with col2:
            asset_beta = st.number_input("Beta del Activo", value=1.2, min_value=-3.0, max_value=5.0)
        
        if market_return > risk_free_rate:
            market_premium = market_return - risk_free_rate
            required_return = risk_free_rate + asset_beta * market_premium
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prima de Riesgo del Mercado", f"{market_premium:.2f}%")
            with col2:
                st.metric("Retorno Requerido", f"{required_return:.2f}%")
            with col3:
                risk_premium = asset_beta * market_premium
                st.metric("Prima de Riesgo del Activo", f"{risk_premium:.2f}%")
        
        st.subheader("Línea del Mercado de Valores (SML)")
        
        if st.button("Graficar Línea del Mercado de Valores"):
            beta_range = np.linspace(-0.5, 3.0, 100)
            sml_returns = risk_free_rate + beta_range * (market_return - risk_free_rate)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=beta_range, y=sml_returns, mode='lines', 
                                   name='Security Market Line'))
            
            sample_securities = {
                'Bonos del Tesoro': (0.0, risk_free_rate),
                'Portafolio de Mercado': (1.0, market_return),
                'Tu Activo': (asset_beta, required_return),
                'Acción Defensiva': (0.6, risk_free_rate + 0.6 * market_premium),
                'Acción de Crecimiento': (1.8, risk_free_rate + 1.8 * market_premium)
            }
            
            for security, (beta, ret) in sample_securities.items():
                color = 'red' if security == 'Tu Activo' else 'blue'
                fig.add_trace(go.Scatter(x=[beta], y=[ret], mode='markers+text', 
                                       text=[security], textposition="top center",
                                       marker=dict(size=10, color=color), name=security,
                                       showlegend=False))
            
            fig.update_layout(title="Línea del Mercado de Valores (SML)", 
                            xaxis_title="Beta", yaxis_title="Retorno Esperado (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Cálculo de Beta")
        
        st.write("El beta mide el riesgo sistemático: cuánto se mueve el retorno de un activo con respecto al mercado.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Interpretación del Beta:**")
            st.write("- β = 1.0: se mueve igual que el mercado")
            st.write("- β > 1.0: más volátil que el mercado")
            st.write("- β < 1.0: menos volátil que el mercado")
            st.write("- β < 0: se mueve en dirección opuesta al mercado")
        
        with col2:
            st.write("**Fórmula del Beta:**")
            st.latex(r"\beta_i = \rho_{i,m} \times \frac{\sigma_i}{\sigma_m}")
            
            correlation = st.slider("Correlación con el Mercado", -1.0, 1.0, 0.75, 0.05)
            asset_volatility = st.slider("Volatilidad del Activo (%)", 1.0, 50.0, 20.0, 1.0)
            market_volatility = st.slider("Volatilidad del Mercado (%)", 1.0, 50.0, 15.0, 1.0)
            
            calculated_beta = correlation * (asset_volatility / market_volatility)
            st.metric("Beta Calculado", f"{calculated_beta:.3f}")
    
    with tab4:
        st.subheader("Optimización de Portafolios")
        
        st.write("""
        La optimización de portafolios consiste en encontrar los pesos óptimos de los activos para lograr
        objetivos específicos como maximizar el retorno, minimizar el riesgo o maximizar el ratio de Sharpe.
        """)
        
        optimization_type = st.selectbox("Selecciona el objetivo de optimización:", 
                                         ["Maximize Sharpe Ratio", "Minimize Risk", "Maximize Return"])
        
        # Datos de ejemplo del portafolio
        st.subheader("Activos del Portafolio")
        
        # Siempre guardamos internamente con claves en inglés
        if 'portfolio_assets' not in st.session_state:
            st.session_state.portfolio_assets = pd.DataFrame({
                'Asset': ['US Stocks', 'International Stocks', 'Bonds', 'REITs'],
                'Expected Return (%)': [10.0, 8.5, 4.0, 7.0],
                'Risk (%)': [16.0, 18.0, 4.0, 20.0]
            })
        
        # Mostrar editor en español pero normalizar a inglés para el cálculo
        display_df = st.session_state.portfolio_assets.copy()
        display_df = display_df.rename(columns={
            'Asset': 'Activo',
            'Expected Return (%)': 'Retorno Esperado (%)',
            'Risk (%)': 'Riesgo (%)'
        })
        
        edited_display = st.data_editor(
            display_df,
            use_container_width=True,
            num_rows="dynamic"
        )
        
        # Normalización: aceptar tanto español como inglés de regreso desde el editor
        back_map = {
            'Activo': 'Asset',
            'Retorno Esperado (%)': 'Expected Return (%)',
            'Riesgo (%)': 'Risk (%)'
        }
        edited_assets = edited_display.rename(columns={c: back_map.get(c, c) for c in edited_display.columns})
        
        if len(edited_assets) >= 2:
            # Entrada de la matriz de correlaciones
            st.subheader("Matriz de Correlaciones")
            n_assets = len(edited_assets)
            
            # Inicializar correlaciones si cambia el número de activos
            if 'corr_matrix' not in st.session_state or len(st.session_state.corr_matrix) != n_assets:
                st.session_state.corr_matrix = np.eye(n_assets)
                if n_assets >= 2:
                    st.session_state.corr_matrix[0, 1] = st.session_state.corr_matrix[1, 0] = 0.7  # US-Intl stocks
                if n_assets >= 3:
                    st.session_state.corr_matrix[0, 2] = st.session_state.corr_matrix[2, 0] = -0.1  # Stocks-Bonds
                    st.session_state.corr_matrix[1, 2] = st.session_state.corr_matrix[2, 1] = -0.2
                if n_assets >= 4:
                    st.session_state.corr_matrix[0, 3] = st.session_state.corr_matrix[3, 0] = 0.6  # Stocks-REITs
            
            # Mostrar pares para edición
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
            
            if st.button("Optimizar Portafolio"):
                # Validar columnas (por si alguien renombró manualmente)
                required_cols = {'Asset', 'Expected Return (%)', 'Risk (%)'}
                if not required_cols.issubset(set(edited_assets.columns)):
                    st.error("Las columnas deben contener: 'Asset', 'Expected Return (%)', 'Risk (%)'.")
                    st.stop()
                
                # Preparar datos
                returns = edited_assets['Expected Return (%)'].values / 100.0
                risks = edited_assets['Risk (%)'].values / 100.0
                cov_matrix = np.outer(risks, risks) * st.session_state.corr_matrix
                
                rf_rate = st.number_input("Tasa libre de riesgo (%)", value=3.0, min_value=0.0) / 100.0
                
                def portfolio_performance(weights):
                    port_return = np.sum(returns * weights)
                    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    sharpe = (port_return - rf_rate) / port_risk if port_risk > 0 else 0.0
                    return port_return, port_risk, sharpe
                
                # Restricciones y límites
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                bounds = tuple((0.0, 1.0) for _ in range(n_assets))
                initial_guess = np.array([1.0 / n_assets] * n_assets)
                
                # Objetivo según selección
                if optimization_type == "Maximize Sharpe Ratio":
                    objective = lambda w: -portfolio_performance(w)[2]
                elif optimization_type == "Minimize Risk":
                    objective = lambda w: portfolio_performance(w)[1]
                else:  # Maximize Return
                    objective = lambda w: -np.sum(returns * w)
                
                # Resolver
                result = minimize(objective, initial_guess, method='SLSQP',
                                  bounds=bounds, constraints=constraints,
                                  options={'maxiter': 1000, 'disp': False})
                
                if result.success:
                    optimal_weights = result.x
                    opt_return, opt_risk, opt_sharpe = portfolio_performance(optimal_weights)
                    
                    st.success("✅ ¡Optimización exitosa!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Retorno del Portafolio", f"{opt_return*100:.2f}%")
                    with col2:
                        st.metric("Riesgo del Portafolio", f"{opt_risk*100:.2f}%")
                    with col3:
                        st.metric("Ratio de Sharpe", f"{opt_sharpe:.3f}")
                    
                    # Pesos óptimos (mostrar en español)
                    st.subheader("Asignación Óptima por Activo")
                    weights_df = pd.DataFrame({
                        'Activo': edited_assets['Asset'],
                        'Peso (%)': optimal_weights * 100.0
                    })
                    st.dataframe(weights_df, use_container_width=True)
                    
                    # Gráfico de torta
                    fig = px.pie(weights_df, values='Peso (%)', names='Activo',
                                 title='Asignación Óptima del Portafolio')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Comparación vs. equiponderado
                    equal_weights = np.array([1.0 / n_assets] * n_assets)
                    eq_return, eq_risk, eq_sharpe = portfolio_performance(equal_weights)
                    
                    st.subheader("Comparación con Portafolio Equiponderado")
                    comparison_df = pd.DataFrame({
                        'Portafolio': ['Optimizado', 'Equiponderado'],
                        'Retorno (%)': [opt_return*100.0, eq_return*100.0],
                        'Riesgo (%)': [opt_risk*100.0, eq_risk*100.0],
                        'Ratio de Sharpe': [opt_sharpe, eq_sharpe]
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                else:
                    st.error("❌ La optimización falló. Ajusta correlaciones o retornos esperados e inténtalo de nuevo.")

        
        st.subheader("Evaluación de la Teoría de Portafolios")
        
        q1 = st.radio(
            "¿Qué representa la frontera eficiente?",
            ["Todos los portafolios posibles", "Portafolios óptimos", "Portafolios de alto riesgo", "Portafolios de bajo riesgo"],
            key="port_q1"
        )
        
        q2 = st.radio(
            "Según el CAPM, ¿qué determina el retorno esperado?",
            ["Riesgo total", "Riesgo sistemático (beta)", "Riesgo no sistemático", "Correlación"],
            key="port_q2"
        )
        
        q3 = st.radio(
            "¿Qué tipo de riesgo reduce la diversificación?",
            ["Riesgo sistemático", "Riesgo de mercado", "Riesgo no sistemático", "Riesgo de tasa de interés"],
            key="port_q3"
        )
        
        if st.button("Enviar Evaluación"):
            score = 0
            
            if q1 == "Portafolios óptimos":
                st.success("P1: ¡Correcta! ✅")
                score += 1
            else:
                st.error("P1: Incorrecta. La frontera eficiente muestra combinaciones óptimas de riesgo y retorno.")
            
            if q2 == "Riesgo sistemático (beta)":
                st.success("P2: ¡Correcta! ✅")
                score += 1
            else:
                st.error("P2: Incorrecta. Según el CAPM, el retorno esperado depende del riesgo sistemático (beta).")
            
            if q3 == "Riesgo no sistemático":
                st.success("P3: ¡Correcta! ✅")
                score += 1
            else:
                st.error("P3: Incorrecta. La diversificación reduce el riesgo no sistemático o específico.")
            
            st.write(f"Tu puntaje: {score}/3")
            
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Clase 7: Teoría de Portafolios")
                progress_tracker.set_class_score("Clase 7: Teoría de Portafolios", (score/3) * 100)
                st.success("🎉 ¡Excelente! ¡Dominas la teoría de portafolios!")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Recursos sobre Portafolios")
    
    portfolio_code = """
# Teoría y Optimización de Portafolios

import numpy as np
from scipy.optimize import minimize

def portfolio_performance(weights, returns, cov_matrix):
    '''Calcular retorno y riesgo del portafolio'''
    portfolio_return = np.sum(returns * weights)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    return portfolio_return, portfolio_std

def optimize_sharpe_ratio(returns, cov_matrix, risk_free_rate):
    '''Optimizar portafolio para máximo ratio de Sharpe'''
    n_assets = len(returns)
    
    def objective(weights):
        port_return, port_std = portfolio_performance(weights, returns, cov_matrix)
        sharpe_ratio = (port_return - risk_free_rate) / port_std
        return -sharpe_ratio
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    
    result = minimize(objective, initial_guess, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    return result.x if result.success else None

def calculate_capm_return(risk_free_rate, beta, market_return):
    '''Calcular retorno esperado según CAPM'''
    return risk_free_rate + beta * (market_return - risk_free_rate)

# Ejemplo
returns = np.array([0.10, 0.08, 0.04])
cov_matrix = np.array([[0.0225, 0.0108, 0.0015],
                       [0.0108, 0.0196, 0.0012],
                       [0.0015, 0.0012, 0.0025]])

optimal_weights = optimize_sharpe_ratio(returns, cov_matrix, 0.03)
"""
    
    st.sidebar.download_button(
        label="💻 Descargar Código del Portafolio",
        data=portfolio_code,
        file_name="teoria_portafolios.py",
        mime="text/python"
    )
