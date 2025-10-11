import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, t
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 8: Gestión de Riesgos")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Tipos de Riesgo", 
        "Valor en Riesgo (VaR)", 
        "Pruebas de Estrés", 
        "Métricas de Riesgo"
    ])
    
    # === TAB 1: TIPOS DE RIESGO ===
    with tab1:
        st.subheader("Comprendiendo el Riesgo Financiero")
        
        st.write("""
        La gestión del riesgo es el proceso de identificar, analizar y controlar las amenazas
        al capital y las ganancias de una organización. En finanzas, el riesgo toma muchas formas
        y debe ser cuidadosamente medido y administrado.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Riesgo de Mercado:**
            - Riesgo de tasa de interés  
            - Riesgo de precio de acciones  
            - Riesgo cambiario  
            - Riesgo de materias primas  
            - Riesgo de volatilidad  
            """)
        
        with col2:
            st.write("""
            **Riesgo de Crédito:**
            - Riesgo de incumplimiento  
            - Riesgo de diferencial crediticio  
            - Riesgo de contraparte  
            - Riesgo de liquidación  
            """)
        
        st.write("""
        **Otros Tipos de Riesgo:**
        - **Riesgo Operacional:** errores humanos o de procesos internos  
        - **Riesgo de Liquidez:** incapacidad de cumplir obligaciones de corto plazo  
        - **Riesgo Legal:** pérdidas derivadas de sanciones o litigios  
        - **Riesgo Reputacional:** pérdida de confianza del mercado o clientes  
        """)
        
        st.subheader("Marco de Medición del Riesgo")
        
        risk_measures = {
            "Desviación Estándar": {
                "Descripción": "Mide la dispersión de los retornos alrededor de la media.",
                "Fórmula": "σ = √(Σ(r - μ)² / (n-1))",
                "Uso": "Medida básica de volatilidad."
            },
            "Beta": {
                "Descripción": "Mide el riesgo sistemático relativo al mercado.",
                "Fórmula": "β = Cov(r_activo, r_mercado) / Var(r_mercado)",
                "Uso": "Evaluación de riesgo de mercado."
            },
            "Valor en Riesgo": {
                "Descripción": "Pérdida máxima esperada durante un horizonte de tiempo dado.",
                "Fórmula": "VaR = -Percentil(retornos, α)",
                "Uso": "Límites de riesgo y asignación de capital."
            },
            "Pérdida Esperada (ES)": {
                "Descripción": "Pérdida promedio más allá del umbral de VaR.",
                "Fórmula": "ES = E[Pérdida | Pérdida > VaR]",
                "Uso": "Medición del riesgo en la cola de la distribución."
            }
        }
        
        for medida, detalles in risk_measures.items():
            with st.expander(f"📊 {medida}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Descripción:** {detalles['Descripción']}")
                    st.write(f"**Uso:** {detalles['Uso']}")
                with col2:
                    st.write(f"**Fórmula:** {detalles['Fórmula']}")
        
        # === Visualización Riesgo-Retorno ===
        st.subheader("Análisis de Riesgo-Retorno")
        
        if st.button("Generar Análisis Riesgo-Retorno"):
            np.random.seed(42)
            n_portfolios = 1000
            n_assets = 4
            
            # Pesos aleatorios
            weights = np.random.random((n_portfolios, n_assets))
            weights = weights / weights.sum(axis=1)[:, np.newaxis]
            
            # Parámetros de activos
            expected_returns = np.array([0.08, 0.12, 0.15, 0.06])
            volatilities = np.array([0.12, 0.18, 0.25, 0.08])
            
            correlation_matrix = np.array([
                [1.0, 0.3, 0.2, -0.1],
                [0.3, 1.0, 0.5, 0.0],
                [0.2, 0.5, 1.0, 0.1],
                [-0.1, 0.0, 0.1, 1.0]
            ])
            
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            
            portfolio_returns, portfolio_risks = [], []
            
            for w in weights:
                port_return = np.sum(w * expected_returns)
                port_risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                portfolio_returns.append(port_return * 100)
                portfolio_risks.append(port_risk * 100)
            
            fig = go.Figure()
            
            risk_free_rate = 3.0
            sharpe_ratios = [(ret - risk_free_rate) / risk for ret, risk in zip(portfolio_returns, portfolio_risks)]
            
            fig.add_trace(go.Scatter(
                x=portfolio_risks,
                y=portfolio_returns,
                mode='markers',
                marker=dict(
                    color=sharpe_ratios,
                    colorscale='Viridis',
                    size=6,
                    colorbar=dict(title="Ratio de Sharpe")
                ),
                text=[f"Retorno: {ret:.1f}%<br>Riesgo: {risk:.1f}%<br>Sharpe: {sharpe:.2f}" 
                      for ret, risk, sharpe in zip(portfolio_returns, portfolio_risks, sharpe_ratios)],
                name='Portafolios'
            ))
            
            asset_names = ['Bonos', 'Acciones Grandes', 'Acciones Pequeñas', 'Efectivo']
            fig.add_trace(go.Scatter(
                x=volatilities * 100,
                y=expected_returns * 100,
                mode='markers+text',
                text=asset_names,
                textposition="top center",
                marker=dict(size=12, color='red'),
                name='Activos Individuales'
            ))
            
            fig.update_layout(
                title="Análisis Riesgo-Retorno del Portafolio",
                xaxis_title="Riesgo (Desviación Estándar %)",
                yaxis_title="Retorno Esperado (%)",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: VALOR EN RIESGO (VaR) ===
    with tab2:
        st.subheader("Valor en Riesgo (VaR)")
        
        st.write("""
        El Valor en Riesgo (VaR) es una medida estadística que cuantifica el nivel de riesgo financiero
        de una empresa, portafolio o posición durante un período de tiempo específico.
        """)
        
        with st.expander("📊 Métodos para Calcular VaR"):
            st.write("""
            **1. Método Histórico:** utiliza los retornos históricos reales.  
            **2. Método Paramétrico:** asume una distribución normal de los retornos.  
            **3. Método Monte Carlo:** genera escenarios simulados aleatoriamente.  
            """)
        
        var_method = st.selectbox("Selecciona el método de VaR:", 
                                 ["VaR Histórico", "VaR Paramétrico", "VaR Monte Carlo"])
        
        # === MÉTODO HISTÓRICO ===
        if var_method == "VaR Histórico":
            st.write("**Valor en Riesgo Histórico**")
            st.write("Utiliza los retornos reales del pasado para estimar posibles pérdidas futuras.")
            
            if st.button("Generar Datos Históricos de Retornos"):
                np.random.seed(42)
                n_days = 252  # Un año de datos diarios
                
                daily_returns = np.random.normal(0.0008, 0.02, n_days)
                
                extreme_events = np.random.choice(n_days, size=10, replace=False)
                daily_returns[extreme_events] = np.random.normal(-0.03, 0.01, 10)
                
                df_returns = pd.DataFrame({
                    'Fecha': pd.date_range(start='2023-01-01', periods=n_days, freq='B'),
                    'Retorno Diario': daily_returns,
                    'Retorno Acumulado': np.cumsum(daily_returns)
                })
                
                fig = px.line(df_returns, x='Fecha', y='Retorno Acumulado', 
                             title='Retornos Acumulados del Portafolio')
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    confidence_level = st.selectbox("Nivel de Confianza (%)", [90, 95, 99], index=1)
                    portfolio_value = st.number_input("Valor del Portafolio ($)", value=1000000, min_value=1000)
                with col2:
                    time_horizon = st.selectbox("Horizonte de Tiempo (días)", [1, 5, 10], index=0)
                
                alpha = (100 - confidence_level) / 100
                scaled_returns = daily_returns * np.sqrt(time_horizon)
                
                var_historical = np.percentile(scaled_returns, alpha * 100) * portfolio_value
                
                tail_losses = scaled_returns[scaled_returns <= np.percentile(scaled_returns, alpha * 100)]
                expected_shortfall = np.mean(tail_losses) * portfolio_value
                
                st.subheader("Resultados del VaR Histórico")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("VaR Histórico", f"${abs(var_historical):,.0f}")
                with col2:
                    st.metric("Pérdida Esperada (ES)", f"${abs(expected_shortfall):,.0f}")
                with col3:
                    st.metric("VaR como % del Portafolio", f"{abs(var_historical)/portfolio_value*100:.2f}%")
                
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(x=scaled_returns*100, nbinsx=50, 
                                         name='Distribución de Retornos', opacity=0.7))
                
                fig.add_vline(x=np.percentile(scaled_returns, alpha * 100)*100, 
                             line_dash="dash", line_color="red",
                             annotation_text=f"{confidence_level}% VaR")
                
                tail_threshold = np.percentile(scaled_returns, alpha * 100)
                tail_returns = scaled_returns[scaled_returns <= tail_threshold]
                
                fig.add_trace(go.Histogram(x=tail_returns*100, nbinsx=20, 
                                         name='Pérdidas Extremas', opacity=0.8, 
                                         marker_color='red'))
                
                fig.update_layout(
                    title=f"Distribución de Retornos y {confidence_level}% VaR",
                    xaxis_title="Retorno Diario (%)",
                    yaxis_title="Frecuencia",
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # === MÉTODO PARAMÉTRICO ===
        elif var_method == "VaR Paramétrico":
            st.write("**Valor en Riesgo Paramétrico**")
            st.write("Asume que los retornos siguen una distribución normal.")
            
            col1, col2 = st.columns(2)
            with col1:
                portfolio_value = st.number_input("Valor del Portafolio ($)", value=1000000, min_value=1000, key="param_pv")
                expected_return = st.number_input("Retorno Diario Esperado (%)", value=0.05, min_value=-5.0, max_value=5.0) / 100
            with col2:
                volatility = st.number_input("Volatilidad Diaria (%)", value=1.5, min_value=0.1, max_value=10.0) / 100
                confidence_level = st.selectbox("Nivel de Confianza (%)", [90, 95, 99], index=1, key="param_conf")
            
            if st.button("Calcular VaR Paramétrico"):
                alpha = (100 - confidence_level) / 100
                z_score = norm.ppf(alpha)
                
                var_parametric = -(expected_return + z_score * volatility) * portfolio_value
                var_zero_mean = -z_score * volatility * portfolio_value
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("VaR Paramétrico", f"${var_parametric:,.0f}")
                    st.metric("Valor Z", f"{z_score:.3f}")
                with col2:
                    st.metric("VaR (Media Cero)", f"${var_zero_mean:,.0f}")
                    st.metric("VaR como % del Portafolio", f"{var_parametric/portfolio_value*100:.2f}%")
                
                x_range = np.linspace(-4*volatility, 4*volatility, 1000)
                pdf = norm.pdf(x_range, expected_return, volatility)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_range*100, y=pdf, mode='lines', 
                                       name='Distribución de Retornos'))
                
                var_threshold = expected_return + z_score * volatility
                fig.add_vline(x=var_threshold*100, line_dash="dash", line_color="red",
                             annotation_text=f"{confidence_level}% VaR")
                
                tail_x = x_range[x_range <= var_threshold]
                tail_y = norm.pdf(tail_x, expected_return, volatility)
                
                fig.add_trace(go.Scatter(x=tail_x*100, y=tail_y, fill='tonexty', 
                                       fillcolor='rgba(255,0,0,0.3)', mode='none',
                                       name=f'Riesgo de Cola ({alpha*100:.0f}%)'))
                
                fig.update_layout(
                    title="Distribución Normal y VaR Paramétrico",
                    xaxis_title="Retorno Diario (%)",
                    yaxis_title="Densidad de Probabilidad"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # === MÉTODO MONTE CARLO ===
        elif var_method == "VaR Monte Carlo":
            st.write("**Valor en Riesgo por Simulación Monte Carlo**")
            st.write("Utiliza simulaciones aleatorias para generar escenarios posibles de pérdidas futuras.")
            
            col1, col2 = st.columns(2)
            with col1:
                portfolio_value = st.number_input("Valor del Portafolio ($)", value=1000000, min_value=1000, key="mc_pv")
                expected_return = st.number_input("Retorno Diario Esperado (%)", value=0.05, min_value=-5.0, max_value=5.0, key="mc_ret") / 100
                volatility = st.number_input("Volatilidad Diaria (%)", value=1.5, min_value=0.1, max_value=10.0, key="mc_vol") / 100
            with col2:
                n_simulations = st.number_input("Número de Simulaciones", value=10000, min_value=1000, max_value=100000)
                confidence_level = st.selectbox("Nivel de Confianza (%)", [90, 95, 99], index=1, key="mc_conf")
                time_horizon = st.number_input("Horizonte de Tiempo (días)", value=1, min_value=1, max_value=30)
            
            if st.button("Ejecutar Simulación Monte Carlo"):
                np.random.seed(42)
                
                random_returns = np.random.normal(
                    expected_return * time_horizon,
                    volatility * np.sqrt(time_horizon),
                    n_simulations
                )
                
                portfolio_values = portfolio_value * (1 + random_returns)
                portfolio_changes = portfolio_values - portfolio_value
                
                alpha = (100 - confidence_level) / 100
                var_monte_carlo = -np.percentile(portfolio_changes, alpha * 100)
                
                tail_losses = portfolio_changes[portfolio_changes <= -var_monte_carlo]
                expected_shortfall = -np.mean(tail_losses)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("VaR Monte Carlo", f"${var_monte_carlo:,.0f}")
                with col2:
                    st.metric("Pérdida Esperada (ES)", f"${expected_shortfall:,.0f}")
                with col3:
                    st.metric("Pérdida Máxima Simulada", f"${-min(portfolio_changes):,.0f}")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=portfolio_changes/1000, nbinsx=50, 
                                         name='Resultados Simulados ($000)', opacity=0.7))
                
                fig.add_vline(x=-var_monte_carlo/1000, line_dash="dash", line_color="red",
                             annotation_text=f"{confidence_level}% VaR")
                fig.add_vline(x=-expected_shortfall/1000, line_dash="dot", line_color="orange",
                             annotation_text="Pérdida Esperada")
                
                fig.update_layout(
                    title=f"Resultados de la Simulación Monte Carlo ({n_simulations:,} simulaciones)",
                    xaxis_title="Cambio del Portafolio ($000)",
                    yaxis_title="Frecuencia"
                )
                
                st.plotly_chart(fig, use_container_width=True)

    # === TAB 3: PRUEBAS DE ESTRÉS ===
    with tab3:
        st.subheader("Pruebas de Estrés")
        
        st.write("""
        Las pruebas de estrés evalúan cómo se comporta un portafolio bajo escenarios extremos pero plausibles.
        Complementan el VaR al analizar el riesgo en la cola y las limitaciones de los modelos.
        """)
        
        st.subheader("Análisis de Escenarios")
        
        escenarios = {
            "Caso Base": {"Acciones": 0.0, "Bonos": 0.0, "Divisas": 0.0},
            "Colapso del Mercado": {"Acciones": -30.0, "Bonos": 5.0, "Divisas": -10.0},
            "Shock de Tasas de Interés": {"Acciones": -10.0, "Bonos": -15.0, "Divisas": 0.0},
            "Crisis Cambiaria": {"Acciones": -15.0, "Bonos": -5.0, "Divisas": -25.0},
            "Estanflación": {"Acciones": -20.0, "Bonos": -10.0, "Divisas": -15.0},
            "Recuperación": {"Acciones": 25.0, "Bonos": -3.0, "Divisas": 5.0}
        }
        
        st.subheader("Asignación del Portafolio")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            stock_allocation = st.slider("Asignación en Acciones (%)", 0, 100, 60)
            portfolio_value = st.number_input("Valor del Portafolio ($)", value=1000000, min_value=1000, key="stress_pv")
        with col2:
            bond_allocation = st.slider("Asignación en Bonos (%)", 0, 100, 30)
        with col3:
            currency_allocation = 100 - stock_allocation - bond_allocation
            st.metric("Asignación en Divisas (%)", currency_allocation)
        
        if stock_allocation + bond_allocation <= 100:
            if st.button("Ejecutar Prueba de Estrés"):
                resultados = []
                
                for nombre, shocks in escenarios.items():
                    impacto = (
                        (stock_allocation/100) * (shocks["Acciones"]/100) * portfolio_value +
                        (bond_allocation/100) * (shocks["Bonos"]/100) * portfolio_value +
                        (currency_allocation/100) * (shocks["Divisas"]/100) * portfolio_value
                    )
                    
                    nuevo_valor = portfolio_value + impacto
                    
                    resultados.append({
                        'Escenario': nombre,
                        'Impacto ($)': impacto,
                        'Impacto (%)': (impacto / portfolio_value) * 100,
                        'Nuevo Valor ($)': nuevo_valor
                    })
                
                df_stress = pd.DataFrame(resultados)
                st.dataframe(df_stress, use_container_width=True)
                
                fig = px.bar(df_stress, x='Escenario', y='Impacto (%)', 
                           title='Resultados de la Prueba de Estrés',
                           color='Impacto (%)',
                           color_continuous_scale='RdYlGn_r')
                
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Evaluación del Riesgo")
                
                peor = min(resultados, key=lambda x: x['Impacto ($)'])
                mejor = max(resultados, key=lambda x: x['Impacto ($)'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pérdida Máxima", 
                             f"${abs(peor['Impacto ($)']):,.0f}",
                             f"{peor['Impacto (%)']:.1f}%")
                with col2:
                    st.metric("Ganancia Máxima", 
                             f"${mejor['Impacto ($)']:,.0f}",
                             f"{mejor['Impacto (%)']:.1f}%")
                with col3:
                    rango = mejor['Impacto ($)'] - peor['Impacto ($)']
                    st.metric("Rango de Impacto", f"${rango:,.0f}")
        else:
            st.error("⚠️ Las asignaciones deben sumar 100% o menos.")
        
        # === PRUEBA DE ESTRÉS HISTÓRICA ===
        st.subheader("Prueba de Estrés Histórica")
        
        st.write("Evalúa el rendimiento del portafolio durante crisis históricas conocidas:")
        
        crisis_históricas = {
            "Lunes Negro (1987)": {"Fecha": "Oct 1987", "Acciones": -22.6, "Bonos": 2.1},
            "Burbuja Puntocom (2000)": {"Fecha": "Mar 2000", "Acciones": -39.3, "Bonos": 12.6},
            "Atentados 9/11 (2001)": {"Fecha": "Sep 2001", "Acciones": -11.6, "Bonos": 4.5},
            "Crisis Financiera (2008)": {"Fecha": "Oct 2008", "Acciones": -36.8, "Bonos": 20.1},
            "Flash Crash (2010)": {"Fecha": "May 2010", "Acciones": -6.1, "Bonos": 1.2},
            "COVID-19 (2020)": {"Fecha": "Mar 2020", "Acciones": -33.8, "Bonos": 8.0}
        }
        
        seleccion = st.selectbox("Selecciona una crisis histórica:", list(crisis_históricas.keys()))
        
        if st.button("Analizar Impacto Histórico"):
            crisis = crisis_históricas[seleccion]
            
            total = stock_allocation + bond_allocation
            if total > 0:
                peso_acciones = stock_allocation / total
                peso_bonos = bond_allocation / total
                
                impacto_histórico = (
                    peso_acciones * (crisis["Acciones"]/100) * portfolio_value +
                    peso_bonos * (crisis["Bonos"]/100) * portfolio_value
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Impacto Histórico", 
                             f"${impacto_histórico:,.0f}",
                             f"{(impacto_histórico/portfolio_value)*100:.2f}%")
                with col2:
                    st.metric("Periodo de Crisis", crisis["Fecha"])
                
                if impacto_histórico < 0:
                    st.warning(f"⚠️ Tu portafolio habría perdido ${abs(impacto_histórico):,.0f} durante {seleccion}")
                else:
                    st.success(f"✅ Tu portafolio habría ganado ${impacto_histórico:,.0f} durante {seleccion}")
    
    # === TAB 4: MÉTRICAS DE RIESGO ===
    with tab4:
        st.subheader("Métricas de Riesgo Avanzadas")
        
        st.write("""
        Más allá de la volatilidad y el VaR, existen métricas avanzadas que ofrecen una visión más profunda
        del comportamiento del portafolio y su perfil de riesgo.
        """)
        
        if st.button("Generar Datos de Ejemplo del Portafolio"):
            np.random.seed(42)
            n_days = 252
            dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
            
            correlation_matrix = np.array([
                [1.0, 0.7, -0.2, 0.4],
                [0.7, 1.0, -0.1, 0.5],
                [-0.2, -0.1, 1.0, 0.1],
                [0.4, 0.5, 0.1, 1.0]
            ])
            
            means = np.array([0.0008, 0.0006, 0.0002, 0.0004])
            volatilities = np.array([0.016, 0.020, 0.008, 0.025])
            
            random_normal = np.random.multivariate_normal(means, 
                                                         np.outer(volatilities, volatilities) * correlation_matrix, 
                                                         n_days)
            
            df_returns = pd.DataFrame(random_normal, 
                                    columns=['Acciones_USA', 'Acciones_Intl', 'Bonos', 'MateriasPrimas'],
                                    index=dates)
            
            weights = np.array([0.4, 0.3, 0.2, 0.1])
            portfolio_returns = (df_returns * weights).sum(axis=1)
            
            st.subheader("Resumen de Métricas de Riesgo")
            
            annual_return = portfolio_returns.mean() * 252
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.03) / annual_volatility
            
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annual_return - 0.03) / downside_deviation if len(downside_returns) > 0 else np.nan
            
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            var_95 = np.percentile(portfolio_returns, 5)
            expected_shortfall = portfolio_returns[portfolio_returns <= var_95].mean()
            
            from scipy.stats import skew, kurtosis
            portfolio_skewness = skew(portfolio_returns)
            portfolio_kurtosis = kurtosis(portfolio_returns)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Retorno Anual", f"{annual_return*100:.2f}%")
                st.metric("Volatilidad", f"{annual_volatility*100:.2f}%")
            with col2:
                st.metric("Ratio de Sharpe", f"{sharpe_ratio:.3f}")
                st.metric("Ratio de Sortino", f"{sortino_ratio:.3f}")
            with col3:
                st.metric("Máxima Pérdida (Drawdown)", f"{max_drawdown*100:.2f}%")
                st.metric("VaR (95%) Diario", f"{var_95*100:.2f}%")
            with col4:
                st.metric("Asimetría (Skewness)", f"{portfolio_skewness:.3f}")
                st.metric("Curtosis", f"{portfolio_kurtosis:.3f}")
            
            st.subheader("Análisis Detallado de Riesgo")
            
            risk_metrics = {
                'Métrica': [
                    'Valor en Riesgo (95%)', 'Pérdida Esperada (95%)', 'Desviación a la Baja',
                    'Máximo Drawdown', 'Ratio de Calmar', 'Asimetría', 'Exceso de Curtosis'
                ],
                'Valor': [
                    f"{var_95*100:.2f}%",
                    f"{expected_shortfall*100:.2f}%",
                    f"{downside_deviation*100:.2f}%",
                    f"{max_drawdown*100:.2f}%",
                    f"{(annual_return / abs(max_drawdown)):.3f}" if max_drawdown != 0 else "N/A",
                    f"{portfolio_skewness:.3f}",
                    f"{portfolio_kurtosis:.3f}"
                ],
                'Interpretación': [
                    'Pérdida diaria esperada superada el 5% del tiempo',
                    'Promedio de pérdida cuando se supera el VaR',
                    'Volatilidad de los retornos negativos',
                    'Mayor caída desde un máximo histórico',
                    'Retorno por unidad de pérdida máxima',
                    'Grado de asimetría en la distribución',
                    'Comparación de colas con la distribución normal'
                ]
            }
            
            df_risk_metrics = pd.DataFrame(risk_metrics)
            st.dataframe(df_risk_metrics, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                rolling_vol = portfolio_returns.rolling(window=30).std() * np.sqrt(252)
                fig1 = px.line(x=rolling_vol.index, y=rolling_vol.values, 
                              title='Volatilidad Móvil de 30 Días')
                fig1.update_layout(xaxis_title="Fecha", yaxis_title="Volatilidad Anualizada")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.line(x=drawdowns.index, y=drawdowns.values * 100, 
                              title='Evolución del Drawdown del Portafolio')
                fig2.update_layout(xaxis_title="Fecha", yaxis_title="Drawdown (%)")
                st.plotly_chart(fig2, use_container_width=True)
        
        # === EVALUACIÓN FINAL ===
        st.subheader("Evaluación de Gestión de Riesgos")
        
        q1 = st.radio(
            "¿Qué medida captura mejor el riesgo en la cola que el VaR?",
            ["Desviación estándar", "Pérdida esperada", "Beta", "Correlación"],
            key="risk_q1"
        )
        
        q2 = st.radio(
            "El máximo drawdown mide:",
            ["Pérdida promedio", "Caída desde el máximo al mínimo", "Volatilidad diaria", "Asimetría"],
            key="risk_q2"
        )
        
        q3 = st.radio(
            "Un portafolio con asimetría negativa tiene:",
            ["Más potencial al alza", "Mayor riesgo a la baja", "Mayores retornos", "Menor volatilidad"],
            key="risk_q3"
        )
        
        if st.button("Enviar Evaluación"):
            score = 0
            
            if q1 == "Pérdida esperada":
                st.success("Pregunta 1: ✅ Correcto. La pérdida esperada mide la gravedad de las pérdidas extremas.")
                score += 1
            else:
                st.error("Pregunta 1: ❌ Incorrecto. La pérdida esperada captura mejor las pérdidas más allá del VaR.")
            
            if q2 == "Caída desde el máximo al mínimo":
                st.success("Pregunta 2: ✅ Correcto. El drawdown mide la peor caída entre un pico y un valle.")
                score += 1
            else:
                st.error("Pregunta 2: ❌ Incorrecto. El máximo drawdown representa la caída más profunda del valor del portafolio.")
            
            if q3 == "Mayor riesgo a la baja":
                st.success("Pregunta 3: ✅ Correcto. La asimetría negativa implica más probabilidad de grandes pérdidas.")
                score += 1
            else:
                st.error("Pregunta 3: ❌ Incorrecto. Una asimetría negativa significa más pérdidas extremas que ganancias.")
            
            st.write(f"Puntaje obtenido: {score}/3")
            
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Clase 8: Gestión de Riesgos")
                progress_tracker.set_class_score("Clase 8: Gestión de Riesgos", (score/3) * 100)
                st.success("🎉 ¡Excelente! Has comprendido los fundamentos de la gestión de riesgos.")
    
    # === DESCARGA DE MATERIALES ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Recursos de Gestión de Riesgos")
    
    risk_code = """
# Risk Management Tools (versión original en inglés)
import numpy as np
from scipy.stats import norm

def historical_var(returns, confidence_level=0.05):
    return np.percentile(returns, confidence_level * 100)

def parametric_var(expected_return, volatility, confidence_level=0.05):
    z_score = norm.ppf(confidence_level)
    return expected_return + z_score * volatility

def expected_shortfall(returns, confidence_level=0.05):
    var = historical_var(returns, confidence_level)
    return np.mean(returns[returns <= var])

def maximum_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    return drawdowns.min()

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

def sortino_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    return excess_returns.mean() / downside_std
"""
    
    st.sidebar.download_button(
        label="💻 Descargar Código de Riesgo",
        data=risk_code,
        file_name="gestion_riesgos.py",
        mime="text/python"
    )
