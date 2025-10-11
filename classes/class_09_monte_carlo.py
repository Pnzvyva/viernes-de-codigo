import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 9: Métodos de Monte Carlo")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Fundamentos de Monte Carlo", "Simulación de Precios", "Valoración de Opciones", "Aplicaciones de Riesgo"])
    
    with tab1:
        st.subheader("Introducción a los Métodos de Monte Carlo")
        
        st.write("""
        Los métodos de Monte Carlo son algoritmos computacionales que dependen de muestreo aleatorio repetido 
        para obtener resultados numéricos. En finanzas, se utilizan para modelar sistemas complejos y 
        calcular el valor de instrumentos con múltiples fuentes de incertidumbre.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Características clave:**
            - Utiliza generación de números aleatorios
            - Aproxima soluciones mediante muestreo
            - Particularmente útil para problemas de alta dimensión
            - Proporciona intervalos de confianza
            - Maneja estructuras de pago complejas
            """)
        
        with col2:
            st.write("""
            **Aplicaciones financieras:**
            - Valoración de opciones (especialmente opciones exóticas)
            - Gestión de riesgos (VaR, pruebas de estrés)
            - Optimización de portafolios
            - Modelado de riesgo crediticio
            - Gestión de activos y pasivos
            """)
        
        st.subheader("Generación de Números Aleatorios")
        
        with st.expander("🎲 Entendiendo los Números Aleatorios"):
            st.write("""
            Las simulaciones de Monte Carlo requieren números aleatorios de alta calidad. En la práctica, utilizamos:
            - **Números pseudo-aleatorios**: Algoritmos deterministas que producen secuencias que parecen aleatorias
            - **Números cuasi-aleatorios**: Secuencias de baja discrepancia para una mejor convergencia
            - **Variantes antitéticas**: Técnica de reducción de varianza utilizando muestras negativamente correlacionadas
            """)
        
        # Ejemplo simple de Monte Carlo: Estimación de π
        st.subheader("Ejemplo Simple: Estimación de π")
        
        st.write("Podemos estimar π tirando aleatoriamente dardos a un cuadrado que contiene un círculo.")
        
        n_samples = st.slider("Número de muestras:", 1000, 50000, 10000, step=1000)
        
        if st.button("Ejecutar Estimación de π"):
            np.random.seed(42)
            
            # Generar puntos aleatorios en [-1,1] x [-1,1]
            x = np.random.uniform(-1, 1, n_samples)
            y = np.random.uniform(-1, 1, n_samples)
            
            # Comprobar si los puntos están dentro del círculo unitario
            inside_circle = (x**2 + y**2) <= 1
            
            # Estimar π
            pi_estimate = 4 * np.sum(inside_circle) / n_samples
            
            # Crear visualización
            sample_size = min(2000, n_samples)  # Limitar puntos para visualización
            idx = np.random.choice(n_samples, sample_size, replace=False)
            
            colors = ['red' if inside_circle[i] else 'blue' for i in idx]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x[idx], y=y[idx],
                mode='markers',
                marker=dict(color=colors, size=3),
                name='Puntos Aleatorios'
            ))
            
            # Añadir círculo
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = np.cos(theta)
            circle_y = np.sin(theta)
            fig.add_trace(go.Scatter(
                x=circle_x, y=circle_y,
                mode='lines',
                line=dict(color='black', width=2),
                name='Círculo Unitario'
            ))
            
            fig.update_layout(
                title=f"Estimación de π usando Monte Carlo (n={n_samples:,})",
                xaxis_title="X",
                yaxis_title="Y",
                width=600, height=600,
                xaxis=dict(range=[-1.1, 1.1]),
                yaxis=dict(range=[-1.1, 1.1], scaleanchor="x", scaleratio=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Resultados
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimación de π", f"{pi_estimate:.4f}")
            with col2:
                st.metric("π Real", f"{np.pi:.4f}")
            with col3:
                error = abs(pi_estimate - np.pi)
                st.metric("Error", f"{error:.4f}")
        
        st.subheader("Proceso de Monte Carlo")
        
        with st.expander("📊 Algoritmo General de Monte Carlo"):
            st.write("""
            **Paso 1:** Definir el problema e identificar las variables aleatorias
            **Paso 2:** Generar muestras aleatorias de las distribuciones apropiadas
            **Paso 3:** Calcular el resultado para cada muestra
            **Paso 4:** Agregar los resultados (media, percentiles, etc.)
            **Paso 5:** Estimar la precisión y los intervalos de confianza
            """)
        
        # Demostración de la Ley de los Grandes Números
        st.subheader("Propiedades de la Convergencia")
        
        if st.button("Demostrar Convergencia"):
            np.random.seed(42)
            
            # Simular la convergencia de la media muestral
            true_mean = 0.05  # 5% de retorno diario
            true_std = 0.02   # 2% de volatilidad diaria
            
            max_samples = 10000
            sample_sizes = np.logspace(1, np.log10(max_samples), 100, dtype=int)
            
            # Generar todas las muestras de una vez
            all_samples = np.random.normal(true_mean, true_std, max_samples)
            
            # Calcular medias y desviaciones estándar en curso
            running_means = []
            running_stds = []
            
            for n in sample_sizes:
                sample_mean = np.mean(all_samples[:n])
                sample_std = np.std(all_samples[:n], ddof=1)
                running_means.append(sample_mean)
                running_stds.append(sample_std)
            
            # Graficar convergencia
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sample_sizes, y=running_means,
                mode='lines', name='Media Muestral',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=sample_sizes, y=running_stds,
                mode='lines', name='Desviación Estándar Muestral',
                line=dict(color='red'), yaxis='y2'
            ))
            
            # Añadir valores verdaderos como líneas horizontales
            fig.add_hline(y=true_mean, line_dash="dash", line_color="blue", 
                         annotation_text="Media Real")
            fig.add_hline(y=true_std, line_dash="dash", line_color="red", 
                         annotation_text="Desviación Estándar Real")
            
            fig.update_layout(
                title="Convergencia de Monte Carlo",
                xaxis_title="Tamaño de Muestra",
                yaxis_title="Media Muestral",
                yaxis2=dict(title="Desviación Estándar Muestral", overlaying='y', side='right')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Simulación de Precios de Acciones")
        
        st.write("""
        Los precios de las acciones son comúnmente modelados utilizando el Movimiento Browniano Geométrico (GBM), 
        que asume que los cambios de precio siguen una distribución log-normal.
        """)
        
        with st.expander("📈 Movimiento Browniano Geométrico"):
            st.latex(r"dS_t = \mu S_t dt + \sigma S_t dW_t")
            st.write("Forma discreta:")
            st.latex(r"S_{t+1} = S_t \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}Z\right)")
            st.write("Donde Z ~ N(0,1) es una variable aleatoria normal estándar")
        
        st.subheader("Simulación de un Solo Activo")
        
        # Parámetros para la simulación
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.number_input("Precio Inicial de la Acción ($)", value=100.0, min_value=0.01)
            mu = st.number_input("Deriva (μ) - % Anual", value=8.0, min_value=-50.0, max_value=50.0) / 100
        with col2:
            sigma = st.number_input("Volatilidad (σ) - % Anual", value=20.0, min_value=1.0, max_value=100.0) / 100
            T = st.number_input("Horizonte Temporal (años)", value=1.0, min_value=0.1, max_value=10.0)
        with col3:
            n_paths = st.number_input("Número de Caminos", value=1000, min_value=10, max_value=10000)
            n_steps = st.number_input("Número de Pasos", value=252, min_value=10, max_value=1000)
        
        if st.button("Simular Caminos de Acciones"):
            # Parámetros de tiempo
            dt = T / n_steps
            t = np.linspace(0, T, n_steps + 1)
            
            # Generar caminos aleatorios
            np.random.seed(42)
            Z = np.random.standard_normal((n_paths, n_steps))
            
            # Inicializar el array de precios
            S = np.zeros((n_paths, n_steps + 1))
            S[:, 0] = S0
            
            # Generar los caminos usando la fórmula del GBM
            for i in range(n_steps):
                S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
            
            # Crear DataFrame para visualización
            sample_paths = min(50, n_paths)  # Mostrar caminos limitados para mayor claridad
            path_indices = np.random.choice(n_paths, sample_paths, replace=False)
            
            fig = go.Figure()
            
            # Graficar los caminos muestrales
            for i in path_indices:
                fig.add_trace(go.Scatter(
                    x=t, y=S[i, :],
                    mode='lines',
                    opacity=0.3,
                    line=dict(width=1),
                    showlegend=False,
                    hovertemplate='Tiempo: %{x:.2f}<br>Precio: $%{y:.2f}<extra></extra>'
                ))
            
            # Añadir la ruta media
            mean_path = np.mean(S, axis=0)
            fig.add_trace(go.Scatter(
                x=t, y=mean_path,
                mode='lines',
                line=dict(color='red', width=3),
                name='Ruta Media'
            ))
            
            # Añadir la media teórica
            theoretical_mean = S0 * np.exp(mu * t)
            fig.add_trace(go.Scatter(
                x=t, y=theoretical_mean,
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='Media Teórica'
            ))
            
            fig.update_layout(
                title=f"Simulación de Precios de Acciones de Monte Carlo ({n_paths:,} caminos)",
                xaxis_title="Tiempo (años)",
                yaxis_title="Precio de la Acción ($)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Estadísticas
            final_prices = S[:, -1]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precio Final Promedio", f"${np.mean(final_prices):.2f}")
            with col2:
                st.metric("Precio Final Mediano", f"${np.median(final_prices):.2f}")
            with col3:
                st.metric("Desviación Estándar", f"${np.std(final_prices):.2f}")
            with col4:
                theoretical_final = S0 * np.exp(mu * T)
                st.metric("Media Teórica Final", f"${theoretical_final:.2f}")
            
            # Distribución de los precios finales
            fig_hist = px.histogram(final_prices, nbins=50, title="Distribución de los Precios Finales de las Acciones")
            fig_hist.add_vline(x=np.mean(final_prices), line_dash="dash", 
                              annotation_text="Media de la Muestra")
            fig_hist.add_vline(x=theoretical_final, line_dash="dot", 
                              annotation_text="Media Teórica")
            fig_hist.update_layout(xaxis_title="Precio Final de la Acción ($)", yaxis_title="Frecuencia")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Simulación de múltiples activos
        st.subheader("Simulación de Múltiples Activos")
        
        st.write("Simula precios de acciones correlacionadas usando la descomposición de Cholesky.")
        
        # Parámetros para múltiples activos
        n_assets = st.selectbox("Número de Activos:", [2, 3, 4], index=0)
        
        if n_assets >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Parámetros de los Activos:**")
                asset_params = []
                for i in range(n_assets):
                    st.write(f"Activo {i+1}:")
                    price = st.number_input(f"Precio Inicial {i+1}", value=100.0, key=f"price_{i}")
                    drift = st.number_input(f"Deriva {i+1} (%)", value=8.0 + i*2, key=f"drift_{i}") / 100
                    vol = st.number_input(f"Volatilidad {i+1} (%)", value=20.0 + i*5, key=f"vol_{i}") / 100
                    asset_params.append([price, drift, vol])
            
            with col2:
                st.write("**Matriz de Correlación:**")
                # Inicializar la matriz de correlación
                corr_matrix = np.eye(n_assets)
                
                # Ingresar los coeficientes de correlación
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        corr = st.number_input(
                            f"Correlación {i+1}-{j+1}:",
                            value=0.3, min_value=-0.99, max_value=0.99,
                            step=0.1, key=f"corr_{i}_{j}"
                        )
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
                
                # Mostrar la matriz de correlación
                st.write("Matriz de Correlación:")
                st.dataframe(pd.DataFrame(corr_matrix, 
                           columns=[f"Activo {i+1}" for i in range(n_assets)],
                           index=[f"Activo {i+1}" for i in range(n_assets)]))
            
            if st.button("Simular Caminos de Múltiples Activos"):
                # Parámetros de simulación
                T_multi = 1.0
                n_steps_multi = 252
                n_paths_multi = 1000
                dt = T_multi / n_steps_multi
                
                # Extraer parámetros
                S0_multi = np.array([param[0] for param in asset_params])
                mu_multi = np.array([param[1] for param in asset_params])
                sigma_multi = np.array([param[2] for param in asset_params])
                
                # Descomposición de Cholesky para correlación
                L = np.linalg.cholesky(corr_matrix)
                
                # Generar números aleatorios correlacionados
                np.random.seed(42)
                Z_independent = np.random.standard_normal((n_paths_multi, n_steps_multi, n_assets))
                Z_correlated = np.zeros_like(Z_independent)
                
                for path in range(n_paths_multi):
                    for step in range(n_steps_multi):
                        Z_correlated[path, step, :] = L @ Z_independent[path, step, :]
                
                # Inicializar los arrays de precios
                S_multi = np.zeros((n_paths_multi, n_steps_multi + 1, n_assets))
                S_multi[:, 0, :] = S0_multi
                
                # Generar caminos correlacionados
                for path in range(n_paths_multi):
                    for step in range(n_steps_multi):
                        for asset in range(n_assets):
                            S_multi[path, step + 1, asset] = S_multi[path, step, asset] * np.exp(
                                (mu_multi[asset] - 0.5 * sigma_multi[asset]**2) * dt + 
                                sigma_multi[asset] * np.sqrt(dt) * Z_correlated[path, step, asset]
                            )
                
                # Graficar resultados
                fig_multi = go.Figure()
                t = np.linspace(0, T_multi, n_steps_multi + 1)
                colors = ['blue', 'red', 'green', 'purple']
                
                for asset in range(n_assets):
                    # Graficar la ruta media de cada activo
                    mean_path = np.mean(S_multi[:, :, asset], axis=0)
                    fig_multi.add_trace(go.Scatter(
                        x=t, y=mean_path,
                        mode='lines',
                        line=dict(color=colors[asset % len(colors)], width=3),
                        name=f'Activo {asset+1} Media'
                    ))
                
                fig_multi.update_layout(
                    title="Simulación de Monte Carlo de Múltiples Activos (Rutas Medias)",
                    xaxis_title="Tiempo (años)",
                    yaxis_title="Precio de la Acción ($)"
                )
                
                st.plotly_chart(fig_multi, use_container_width=True)
                
                # Análisis de correlación de los precios finales
                final_prices_multi = S_multi[:, -1, :]
                realized_corr = np.corrcoef(final_prices_multi.T)
                
                st.subheader("Correlaciones Realizadas vs Objetivo")
                
                comparison_data = []
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        comparison_data.append({
                            'Par de Activos': f"{i+1}-{j+1}",
                            'Correlación Objetivo': corr_matrix[i, j],
                            'Correlación Realizada': realized_corr[i, j],
                            'Diferencia': realized_corr[i, j] - corr_matrix[i, j]
                        })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    with tab3:
        st.subheader("Valoración de Opciones con Monte Carlo")
        
        st.write("""
        Los métodos de Monte Carlo son particularmente poderosos para valorar opciones complejas donde 
        no existen soluciones analíticas, como las opciones dependientes del camino.
        """)
        
        option_type = st.selectbox("Selecciona el tipo de opción:", 
                                  ["Opciones Europeas", "Opciones Asiáticas", "Opciones de Barrera", "Opciones Lookback"])
        
        if option_type == "Opciones Europeas":
            st.write("**Valoración de Opciones Europeas mediante Monte Carlo**")
            st.write("Compara los resultados de Monte Carlo con los precios analíticos de Black-Scholes.")
            
            col1, col2 = st.columns(2)
            with col1:
                S0_opt = st.number_input("Precio Actual de la Acción", value=100.0, key="eur_S0")
                K_opt = st.number_input("Precio de Ejercicio", value=100.0, key="eur_K")
                T_opt = st.number_input("Tiempo hasta la Expiración (años)", value=0.25, key="eur_T")
            with col2:
                r_opt = st.number_input("Tasa Libre de Riesgo (%)", value=5.0, key="eur_r") / 100
                sigma_opt = st.number_input("Volatilidad (%)", value=20.0, key="eur_sigma") / 100
                n_sims = st.number_input("Número de Simulaciones", value=100000, min_value=1000, key="eur_sims")
            
            call_or_put = st.radio("Tipo de Opción:", ["Call", "Put"], key="eur_type")
            
            if st.button("Valorizar Opción Europea"):
                np.random.seed(42)
                
                # Generar precios de acciones al vencimiento
                Z = np.random.standard_normal(n_sims)
                ST = S0_opt * np.exp((r_opt - 0.5 * sigma_opt**2) * T_opt + sigma_opt * np.sqrt(T_opt) * Z)
                
                # Calcular los payoffs
                if call_or_put == "Call":
                    payoffs = np.maximum(ST - K_opt, 0)
                    bs_price = FinancialCalculations.black_scholes_call(S0_opt, K_opt, T_opt, r_opt, sigma_opt)
                else:
                    payoffs = np.maximum(K_opt - ST, 0)
                    bs_price = FinancialCalculations.black_scholes_put(S0_opt, K_opt, T_opt, r_opt, sigma_opt)
                
                # Descontar al valor presente
                mc_price = np.exp(-r_opt * T_opt) * np.mean(payoffs)
                
                # Calcular el error estándar
                standard_error = np.exp(-r_opt * T_opt) * np.std(payoffs) / np.sqrt(n_sims)
                
                # Resultados
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precio de Monte Carlo", f"${mc_price:.4f}")
                with col2:
                    st.metric("Precio de Black-Scholes", f"${bs_price:.4f}")
                with col3:
                    error = abs(mc_price - bs_price)
                    st.metric("Error de Valoración", f"${error:.4f}")
                
                st.info(f"Error estándar: ±${1.96 * standard_error:.4f} (intervalo de confianza del 95%)")
                
                # Análisis de convergencia
                sample_sizes = np.logspace(2, np.log10(n_sims), 50, dtype=int)
                running_prices = []
                
                for n in sample_sizes:
                    running_price = np.exp(-r_opt * T_opt) * np.mean(payoffs[:n])
                    running_prices.append(running_price)
                
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=sample_sizes, y=running_prices,
                    mode='lines', name='Precio de Monte Carlo'
                ))
                fig_conv.add_hline(y=bs_price, line_dash="dash", 
                                  annotation_text="Precio de Black-Scholes")
                
                fig_conv.update_layout(
                    title="Convergencia de Monte Carlo",
                    xaxis_title="Número de Simulaciones",
                    yaxis_title="Precio de la Opción ($)",
                    xaxis_type="log"
                )
                
                st.plotly_chart(fig_conv, use_container_width=True)
        
        elif option_type == "Opciones Asiáticas":
            st.write("**Valoración de Opciones Asiáticas (Promedio de Precios)**")
            st.write("El payoff de la opción depende del precio promedio de la acción durante la vida de la opción.")
            
            col1, col2 = st.columns(2)
            with col1:
                S0_asian = st.number_input("Precio Actual de la Acción", value=100.0, key="asian_S0")
                K_asian = st.number_input("Precio de Ejercicio", value=100.0, key="asian_K")
                T_asian = st.number_input("Tiempo hasta la Expiración (años)", value=0.25, key="asian_T")
                n_steps_asian = st.number_input("Pasos de Monitoreo", value=50, min_value=10, key="asian_steps")
            with col2:
                r_asian = st.number_input("Tasa Libre de Riesgo (%)", value=5.0, key="asian_r") / 100
                sigma_asian = st.number_input("Volatilidad (%)", value=20.0, key="asian_sigma") / 100
                n_sims_asian = st.number_input("Número de Simulaciones", value=50000, min_value=1000, key="asian_sims")
                
            asian_type = st.radio("Tipo de Opción Asiática:", ["Promedio Aritmético", "Promedio Geométrico"])
            call_put_asian = st.radio("Call o Put:", ["Call", "Put"], key="asian_cp")
            
            if st.button("Valorizar Opción Asiática"):
                dt = T_asian / n_steps_asian
                np.random.seed(42)
                
                # Generar caminos de precios de la acción
                Z = np.random.standard_normal((n_sims_asian, n_steps_asian))
                S = np.zeros((n_sims_asian, n_steps_asian + 1))
                S[:, 0] = S0_asian
                
                for i in range(n_steps_asian):
                    S[:, i + 1] = S[:, i] * np.exp(
                        (r_asian - 0.5 * sigma_asian**2) * dt + sigma_asian * np.sqrt(dt) * Z[:, i]
                    )
                
                # Calcular promedios
                if asian_type == "Promedio Aritmético":
                    averages = np.mean(S, axis=1)
                else:  # Promedio Geométrico
                    averages = np.exp(np.mean(np.log(S), axis=1))
                
                # Calcular los payoffs
                if call_put_asian == "Call":
                    payoffs = np.maximum(averages - K_asian, 0)
                else:
                    payoffs = np.maximum(K_asian - averages, 0)
                
                # Valor de la opción
                asian_price = np.exp(-r_asian * T_asian) * np.mean(payoffs)
                standard_error = np.exp(-r_asian * T_asian) * np.std(payoffs) / np.sqrt(n_sims_asian)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Precio de la Opción Asiática", f"${asian_price:.4f}")
                    st.metric("Error Estándar", f"±${1.96 * standard_error:.4f}")
                with col2:
                    st.metric("Promedio de los Promedios", f"${np.mean(averages):.2f}")
                    st.metric("Tasa de Payoff", f"{np.mean(payoffs > 0)*100:.1f}%")
                
                # Distribución de promedios
                fig_asian = px.histogram(averages, nbins=50, title=f"Distribución de los Precios de {asian_type}")
                fig_asian.add_vline(x=K_asian, line_dash="dash", annotation_text="Precio de Ejercicio")
                st.plotly_chart(fig_asian, use_container_width=True)
        
        elif option_type == "Opciones de Barrera":
            st.write("**Valoración de Opciones de Barrera**")
            st.write("Opciones que se activan o desactivan cuando el precio de la acción cruza un nivel de barrera.")
            
            col1, col2 = st.columns(2)
            with col1:
                S0_barrier = st.number_input("Precio Actual de la Acción", value=100.0, key="barrier_S0")
                K_barrier = st.number_input("Precio de Ejercicio", value=100.0, key="barrier_K")
                barrier_level = st.number_input("Nivel de Barrera", value=120.0, key="barrier_level")
                T_barrier = st.number_input("Tiempo hasta la Expiración (años)", value=0.25, key="barrier_T")
            with col2:
                r_barrier = st.number_input("Tasa Libre de Riesgo (%)", value=5.0, key="barrier_r") / 100
                sigma_barrier = st.number_input("Volatilidad (%)", value=25.0, key="barrier_sigma") / 100
                n_sims_barrier = st.number_input("Número de Simulaciones", value=50000, min_value=1000, key="barrier_sims")
                n_steps_barrier = st.number_input("Pasos de Monitoreo", value=100, min_value=50, key="barrier_steps")
            
            barrier_type = st.selectbox("Tipo de Barrera:", ["Up-and-Out Call", "Up-and-In Call", "Down-and-Out Put"])
            
            if st.button("Valorizar Opción de Barrera"):
                dt = T_barrier / n_steps_barrier
                np.random.seed(42)
                
                # Generar caminos de precios de la acción
                Z = np.random.standard_normal((n_sims_barrier, n_steps_barrier))
                S = np.zeros((n_sims_barrier, n_steps_barrier + 1))
                S[:, 0] = S0_barrier
                
                for i in range(n_steps_barrier):
                    S[:, i + 1] = S[:, i] * np.exp(
                        (r_barrier - 0.5 * sigma_barrier**2) * dt + sigma_barrier * np.sqrt(dt) * Z[:, i]
                    )
                
                # Comprobar condiciones de barrera
                if barrier_type == "Up-and-Out Call":
                    barrier_crossed = np.any(S >= barrier_level, axis=1)
                    payoffs = np.where(~barrier_crossed, np.maximum(S[:, -1] - K_barrier, 0), 0)
                elif barrier_type == "Up-and-In Call":
                    barrier_crossed = np.any(S >= barrier_level, axis=1)
                    payoffs = np.where(barrier_crossed, np.maximum(S[:, -1] - K_barrier, 0), 0)
                else:  # Down-and-Out Put
                    barrier_crossed = np.any(S <= barrier_level, axis=1)
                    payoffs = np.where(~barrier_crossed, np.maximum(K_barrier - S[:, -1], 0), 0)
                
                # Valor de la opción
                barrier_price = np.exp(-r_barrier * T_barrier) * np.mean(payoffs)
                
                # Comparar con opción vanilla
                if "Call" in barrier_type:
                    vanilla_price = FinancialCalculations.black_scholes_call(S0_barrier, K_barrier, T_barrier, r_barrier, sigma_barrier)
                else:
                    vanilla_price = FinancialCalculations.black_scholes_put(S0_barrier, K_barrier, T_barrier, r_barrier, sigma_barrier)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precio de la Opción de Barrera", f"${barrier_price:.4f}")
                with col2:
                    st.metric("Precio de la Opción Vanilla", f"${vanilla_price:.4f}")
                with col3:
                    knockout_rate = np.mean(barrier_crossed) * 100
                    st.metric("Tasa de Golpeo de Barrera", f"{knockout_rate:.1f}%")
                
                # Mostrar caminos muestrales
                sample_paths = min(100, n_sims_barrier)
                path_indices = np.random.choice(n_sims_barrier, sample_paths, replace=False)
                
                fig_barrier = go.Figure()
                
                t = np.linspace(0, T_barrier, n_steps_barrier + 1)
                for i in path_indices[:10]:  # Mostrar solo los primeros 10 por claridad
                    color = 'red' if barrier_crossed[i] else 'blue'
                    fig_barrier.add_trace(go.Scatter(
                        x=t, y=S[i, :],
                        mode='lines',
                        line=dict(color=color, width=1),
                        opacity=0.6,
                        showlegend=False
                    ))
                
                fig_barrier.add_hline(y=barrier_level, line_dash="dash", 
                                     annotation_text="Nivel de Barrera")
                fig_barrier.add_hline(y=S0_barrier, line_dash="dot", 
                                     annotation_text="Precio Inicial")
                
                fig_barrier.update_layout(
                    title=f"{barrier_type} - Caminos Muestrales (Rojo = Barrera Golpeada)",
                    xaxis_title="Tiempo (años)",
                    yaxis_title="Precio de la Acción ($)"
                )
                
                st.plotly_chart(fig_barrier, use_container_width=True)

        elif option_type == "Opciones Lookback":
            st.write("**Valoración de Opciones Lookback**")
            
            # Parámetros de la opción Lookback
            col1, col2 = st.columns(2)
            with col1:
                S0_lookback = st.number_input("Precio Actual de la Acción", value=100.0, key="lookback_S0")
                T_lookback = st.number_input("Tiempo hasta la Expiración (años)", value=0.25, key="lookback_T")
                n_steps_lookback = st.number_input("Pasos por Simulación", value=252, min_value=10, key="lookback_steps")
            with col2:
                r_lookback = st.number_input("Tasa Libre de Riesgo (%)", value=5.0, key="lookback_r") / 100
                sigma_lookback = st.number_input("Volatilidad (%)", value=20.0, key="lookback_sigma") / 100
                n_sims_lookback = st.number_input("Número de Simulaciones", value=50000, min_value=1000, key="lookback_sims")
            
            option_type_lookback = st.radio("Tipo de Opción:", ["Call", "Put"], key="lookback_option_type")
            
            if st.button("Valorizar Opción Lookback"):
                # Precio de la Opción Lookback usando Monte Carlo
                lookback_price, lookback_se, payoffs = FinancialCalculations.lookback_option_mc(
                    S0_lookback, T_lookback, r_lookback, sigma_lookback, n_sims_lookback, option_type=option_type_lookback, n_steps=n_steps_lookback
                )
                
                # Depuración de valores
                st.write(f"Lookback Option Price: {lookback_price}")
                st.write(f"Standard Error: {lookback_se}")
                
                # Mostrar los resultados
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{option_type_lookback} Lookback Option Price", f"${lookback_price:.4f}")
                    st.metric("Standard Error", f"±${1.96 * lookback_se:.4f}")
                
                # Análisis de convergencia
                sample_sizes = np.logspace(2, np.log10(n_sims_lookback), 50, dtype=int)
                running_prices = []
                
                # Aquí usamos los payoffs calculados previamente
                for n in sample_sizes:
                    running_price = np.exp(-r_lookback * T_lookback) * np.mean(payoffs[:n])
                    running_prices.append(running_price)
                
                # Gráfico de la convergencia
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=sample_sizes, y=running_prices,
                    mode='lines', name='Monte Carlo Price'
                ))
                
                fig_conv.update_layout(
                    title="Monte Carlo Convergence",
                    xaxis_title="Number of Simulations",
                    yaxis_title="Option Price ($)",
                    xaxis_type="log"
                )
                
                st.plotly_chart(fig_conv, use_container_width=True)

    with tab4:
        st.subheader("Aplicaciones en la Gestión de Riesgos")
        
        st.write("""
        Los métodos de Monte Carlo se utilizan ampliamente en la gestión de riesgos para el análisis de escenarios, 
        pruebas de estrés y el cálculo de métricas de riesgo como el VaR (Valor en Riesgo) y la Pérdida Esperada (Expected Shortfall).
        """)
        
        risk_app = st.selectbox("Selecciona la Aplicación de Riesgo:", 
                               ["VaR de Portafolio", "Pruebas de Estrés", "Modelado de Riesgo Crediticio"])
        
        if risk_app == "VaR de Portafolio":
            st.write("**Valor en Riesgo de Portafolio utilizando Monte Carlo**")
            
            # Configuración del portafolio
            st.subheader("Configuración del Portafolio")
            
            n_assets_var = st.selectbox("Número de Activos:", [2, 3, 4, 5], index=2)
            
            # Parámetros de los activos
            asset_weights = []
            asset_returns = []
            asset_vols = []
            
            for i in range(n_assets_var):
                col1, col2, col3 = st.columns(3)
                with col1:
                    weight = st.number_input(f"Peso del Activo {i+1} (%)", value=100/n_assets_var, 
                                           min_value=0.0, max_value=100.0, key=f"var_weight_{i}")
                    asset_weights.append(weight/100)
                with col2:
                    ret = st.number_input(f"Retorno Esperado {i+1} (%)", value=8.0 + i*2, key=f"var_ret_{i}")
                    asset_returns.append(ret/100)
                with col3:
                    vol = st.number_input(f"Volatilidad {i+1} (%)", value=15.0 + i*5, key=f"var_vol_{i}")
                    asset_vols.append(vol/100)
            
            # Matriz de correlación
            st.subheader("Correlaciones entre los Activos")
            corr_var = np.eye(n_assets_var)
            
            for i in range(n_assets_var):
                for j in range(i+1, n_assets_var):
                    corr_val = st.number_input(f"Correlación {i+1}-{j+1}:", 
                                             value=0.3, min_value=-0.99, max_value=0.99,
                                             key=f"var_corr_{i}_{j}")
                    corr_var[i, j] = corr_val
                    corr_var[j, i] = corr_val
            
            portfolio_value = st.number_input("Valor del Portafolio ($)", value=1000000.0, min_value=1000.0)
            time_horizon = st.selectbox("Horizonte Temporal:", ["1 Día", "1 Semana", "1 Mes"])
            confidence_level = st.selectbox("Nivel de Confianza:", ["95%", "99%", "99.9%"])
            
            if st.button("Calcular VaR de Portafolio"):
                # Convertir el horizonte temporal a fracción de año
                horizon_map = {"1 Día": 1/252, "1 Semana": 5/252, "1 Mes": 21/252}
                dt = horizon_map[time_horizon]
                
                # Convertir el nivel de confianza
                conf_map = {"95%": 0.05, "99%": 0.01, "99.9%": 0.001}
                alpha = conf_map[confidence_level]
                
                # Parámetros del portafolio
                weights = np.array(asset_weights)
                mu = np.array(asset_returns)
                sigma = np.array(asset_vols)
                
                # Estadísticas del portafolio
                portfolio_return = np.sum(weights * mu)
                cov_matrix = np.outer(sigma, sigma) * corr_var
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Simulación de Monte Carlo
                n_sims_var = 100000
                np.random.seed(42)
                
                # Generar retornos correlacionados
                L = np.linalg.cholesky(corr_var)
                Z = np.random.standard_normal((n_sims_var, n_assets_var))
                corr_returns = Z @ L.T
                
                # Escalar según los parámetros individuales de los activos
                asset_returns_sim = np.zeros_like(corr_returns)
                for i in range(n_assets_var):
                    asset_returns_sim[:, i] = mu[i] * dt + sigma[i] * np.sqrt(dt) * corr_returns[:, i]
                
                # Calcular los retornos del portafolio
                portfolio_returns = asset_returns_sim @ weights
                portfolio_values = portfolio_value * (1 + portfolio_returns)
                portfolio_pnl = portfolio_values - portfolio_value
                
                # Calcular el VaR y Expected Shortfall
                var_mc = -np.percentile(portfolio_pnl, alpha * 100)
                tail_losses = portfolio_pnl[portfolio_pnl <= -var_mc]
                es_mc = -np.mean(tail_losses)
                
                # VaR paramétrico para comparación
                var_param = -portfolio_value * (portfolio_return * dt - 1.96 * portfolio_vol * np.sqrt(dt))
                
                # Resultados
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("VaR de Monte Carlo", f"${var_mc:,.0f}")
                    st.metric("Expected Shortfall", f"${es_mc:,.0f}")
                with col2:
                    st.metric("VaR Paramétrico", f"${var_param:,.0f}")
                    st.metric("Diferencia de VaR", f"${abs(var_mc - var_param):,.0f}")
                with col3:
                    st.metric("Retorno del Portafolio", f"{portfolio_return*100:.2f}%")
                    st.metric("Volatilidad del Portafolio", f"{portfolio_vol*100:.2f}%")
                
                # Distribución del P&L
                fig_var = px.histogram(portfolio_pnl/1000, nbins=50, 
                                      title=f"Distribución de P&L del Portafolio ({time_horizon})")
                fig_var.add_vline(x=-var_mc/1000, line_dash="dash", line_color="red",
                                 annotation_text=f"VaR {confidence_level}")
                fig_var.add_vline(x=-es_mc/1000, line_dash="dot", line_color="orange",
                                 annotation_text="Expected Shortfall")
                fig_var.update_layout(xaxis_title="P&L ($000)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_var, use_container_width=True)
        
        elif risk_app == "Pruebas de Estrés":
            st.write("**Pruebas de Estrés con Monte Carlo**")
            st.write("Genera escenarios de mercado extremos y evalúa el impacto en el portafolio.")
            
            # Parámetros de las pruebas de estrés
            stress_scenarios = {
                "Caída del Mercado": {"equity": -0.30, "bond": 0.05, "volatility_multiplier": 3.0},
                "Shock de Tasa de Interés": {"equity": -0.15, "bond": -0.20, "volatility_multiplier": 2.0},
                "Pico de Inflación": {"equity": -0.20, "bond": -0.15, "volatility_multiplier": 2.5},
                "Crisis de Liquidez": {"equity": -0.25, "bond": -0.10, "volatility_multiplier": 4.0}
            }
            
            selected_stress = st.selectbox("Selecciona el Escenario de Estrés:", list(stress_scenarios.keys()))
            portfolio_allocation = st.slider("Asignación a Acciones (%)", 0, 100, 60)
            bond_allocation = 100 - portfolio_allocation
            
            if st.button("Ejecutar Prueba de Estrés"):
                scenario = stress_scenarios[selected_stress]
                n_sims_stress = 10000
                
                # Parámetros base
                base_equity_return = 0.08
                base_bond_return = 0.03
                equity_vol = 0.20
                bond_vol = 0.05
                correlation = 0.1
                
                # Parámetros estresados
                stressed_equity_return = base_equity_return + scenario["equity"]
                stressed_bond_return = base_bond_return + scenario["bond"]
                stressed_equity_vol = equity_vol * scenario["volatility_multiplier"]
                stressed_bond_vol = bond_vol * scenario["volatility_multiplier"]
                
                np.random.seed(42)
                
                # Generar retornos correlacionados
                cov_matrix = np.array([
                    [stressed_equity_vol**2, correlation * stressed_equity_vol * stressed_bond_vol],
                    [correlation * stressed_equity_vol * stressed_bond_vol, stressed_bond_vol**2]
                ])
                
                mean_returns = np.array([stressed_equity_return, stressed_bond_return])
                
                # Simulación de Monte Carlo
                returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_sims_stress)
                
                # Pesos del portafolio
                weights = np.array([portfolio_allocation/100, bond_allocation/100])
                portfolio_returns_stress = returns @ weights
                
                # Calcular estadísticas del portafolio
                mean_return = np.mean(portfolio_returns_stress)
                portfolio_vol_stress = np.std(portfolio_returns_stress)
                var_95_stress = np.percentile(portfolio_returns_stress, 5)
                
                # Resultados
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retorno Esperado", f"{mean_return*100:.2f}%")
                    st.metric("Volatilidad", f"{portfolio_vol_stress*100:.2f}%")
                with col2:
                    st.metric("VaR al 95%", f"{var_95_stress*100:.2f}%")
                    prob_loss = np.mean(portfolio_returns_stress < 0) * 100
                    st.metric("Probabilidad de Pérdida", f"{prob_loss:.1f}%")
                with col3:
                    worst_case = np.min(portfolio_returns_stress)
                    best_case = np.max(portfolio_returns_stress)
                    st.metric("Peor Caso", f"{worst_case*100:.2f}%")
                    st.metric("Mejor Caso", f"{best_case*100:.2f}%")
                
                # Gráfico de distribución
                fig_stress = px.histogram(portfolio_returns_stress*100, nbins=50, 
                                         title=f"Retornos del Portafolio Bajo Estrés - {selected_stress}")
                fig_stress.add_vline(x=0, line_dash="dash", annotation_text="Punto de Equilibrio")
                fig_stress.add_vline(x=var_95_stress*100, line_dash="dot", line_color="red",
                                    annotation_text="VaR al 95%")
                fig_stress.update_layout(xaxis_title="Retorno del Portafolio (%)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_stress, use_container_width=True)
        
        # Evaluación
        st.subheader("Evaluación de los Métodos de Monte Carlo")
        
        q1 = st.radio(
            "¿Cuál es la principal ventaja de los métodos de Monte Carlo en finanzas?",
            ["Velocidad de cálculo", "Manejo de payoffs complejos", "Precisión perfecta", "Sin aleatoriedad"],
            key="mc_q1"
        )
        
        q2 = st.radio(
            "El error estándar de las estimaciones de Monte Carlo disminuye a medida que:",
            ["n (tamaño de la muestra)", "√n", "1/√n", "1/n"],
            key="mc_q2"
        )
        
        q3 = st.radio(
            "¿Qué tipo de opción es más adecuada para la valoración mediante Monte Carlo?",
            ["Opciones vanillas europeas", "Opciones dependientes del camino", "Opciones binarias simples", "Todas igualmente adecuadas"],
            key="mc_q3"
        )
        
        if st.button("Enviar Evaluación"):
            score = 0
            
            if q1 == "Manejo de payoffs complejos":
                st.success("Q1: ¡Correcto! ✅")
                score += 1
            else:
                st.error("Q1: Incorrecto. Monte Carlo es ideal para manejar payoffs complejos y dependientes del camino.")
            
            if q2 == "1/√n":
                st.success("Q2: ¡Correcto! ✅")
                score += 1
            else:
                st.error("Q2: Incorrecto. El error estándar de Monte Carlo disminuye a medida que 1/√n.")
            
            if q3 == "Opciones dependientes del camino":
                st.success("Q3: ¡Correcto! ✅")
                score += 1
            else:
                st.error("Q3: Incorrecto. Monte Carlo es más valioso para opciones dependientes del camino.")
            
            st.write(f"Tu puntuación: {score}/3")
            
            if score >= 2:
                st.balloons()
                progress_tracker.mark_class_completed("Clase 9: Métodos de Monte Carlo")
                progress_tracker.set_class_score("Clase 9: Métodos de Monte Carlo", (score/3) * 100)
                st.success("🎉 ¡Excelente! ¡Has dominado los métodos de Monte Carlo!")
    
    # Materiales de descarga
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Recursos de Monte Carlo")
    
    monte_carlo_code = """
# Métodos de Monte Carlo en Finanzas

import numpy as np
from scipy.stats import norm

def geometric_brownian_motion(S0, mu, sigma, T, n_steps, n_paths):
    '''Simula los caminos de precios de la acción usando GBM'''
    dt = T / n_steps
    
    # Genera choques aleatorios
    Z = np.random.standard_normal((n_paths, n_steps))
    
    # Inicializa el arreglo de precios
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    
    # Genera los caminos
    for i in range(n_steps):
        S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
    
    return S

def monte_carlo_option_price(S0, K, T, r, sigma, n_sims, option_type='call'):
    '''Valora una opción europea usando Monte Carlo'''
    np.random.seed(42)
    
    # Genera los precios finales de las acciones
    Z = np.random.standard_normal(n_sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Calcula los payoffs
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Descuento al valor presente
    option_price = np.exp(-r * T) * np.mean(payoffs)
    standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    
    return option_price, standard_error

def asian_option_mc(S0, K, T, r, sigma, n_steps, n_sims, option_type='call', avg_type='arithmetic'):
    '''Valora una opción asiática usando Monte Carlo'''
    dt = T / n_steps
    
    # Genera los caminos de precios de la acción
    paths = geometric_brownian_motion(S0, r, sigma, T, n_steps, n_sims)
    
    # Calcula los promedios
    if avg_type == 'arithmetic':
        averages = np.mean(paths, axis=1)
    else:  # geométrico
        averages = np.exp(np.mean(np.log(paths), axis=1))
    
    # Calcula los payoffs
    if option_type == 'call':
        payoffs = np.maximum(averages - K, 0)
    else:
        payoffs = np.maximum(K - averages, 0)
    
    # Valor de la opción
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def portfolio_var_mc(weights, returns, cov_matrix, portfolio_value, confidence_level=0.05, n_sims=100000):
    '''Calcula el VaR del portafolio usando Monte Carlo'''
    np.random.seed(42)
    
    # Genera retornos correlacionados
    portfolio_returns = np.random.multivariate_normal(returns, cov_matrix, n_sims)
    
    # Calcula los retornos del portafolio
    port_returns = portfolio_returns @ weights
    
    # Calcula el P&L
    portfolio_pnl = portfolio_value * port_returns
    
    # Calcula el VaR y Expected Shortfall
    var = -np.percentile(portfolio_pnl, confidence_level * 100)
    es = -np.mean(portfolio_pnl[portfolio_pnl <= -var])
    
    return var, es

# Ejemplo de uso
S0, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
call_price, se = monte_carlo_option_price(S0, K, T, r, sigma, 100000)
print(f"Precio de la opción Call: ${call_price:.4f} ± ${1.96*se:.4f}")
"""
    
    st.sidebar.download_button(
        label="💻 Descargar Código de Monte Carlo",
        data=monte_carlo_code,
        file_name="metodos_monte_carlo.py",
        mime="text/python"
    )

