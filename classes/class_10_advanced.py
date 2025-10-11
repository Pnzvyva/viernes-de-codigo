import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 10: Temas Avanzados en Finanzas Cuantitativas")
    
    progress_tracker = st.session_state.progress_tracker
    
    tab1, tab2, tab3, tab4 = st.tabs(["Derivados Exóticos", "Modelos de Riesgo Crediticio", "Trading Algorítmico", "Síntesis del Curso"])
    
    with tab1:
        st.subheader("Derivados Exóticos y Productos Estructurados")
        st.write("""
        Los derivados exóticos son instrumentos financieros no estándar con estructuras de pago complejas. 
        Se utilizan para necesidades de cobertura específicas o para expresar visiones sofisticadas del mercado.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            **Opciones Dependientes del Camino:**
            - Opciones Asiáticas (precio promedio)
            - Opciones Barrera (knock-in/out)
            - Opciones Lookback (extremos)
            - Opciones Cliquet (ratchet)
            """)
        with col2:
            st.write("""
            **Derivados Multi-Activo:**
            - Opciones Rainbow
            - Opciones Basket
            - Opciones de Intercambio
            - Swaps de Correlación
            """)
        
        exotic_type = st.selectbox("Seleccione Derivado Exótico:", 
                                  ["Opción Rainbow", "Opción Cliquet", "Variance Swap"])
        
        # --------------------- OPCIÓN RAINBOW ---------------------
        if exotic_type == "Opción Rainbow":
            st.write("**Opción Rainbow (Mejor/Peor de N Activos)**")
            st.write("El pago depende del activo con mejor o peor desempeño entre varios subyacentes.")
            
            n_assets_rainbow = st.selectbox("Número de Activos:", [2, 3, 4], index=1)
            
            col1, col2 = st.columns(2)
            with col1:
                K_rainbow = st.number_input("Precio de Ejercicio", value=100.0, key="rainbow_K")
                T_rainbow = st.number_input("Tiempo a Vencimiento (años)", value=1.0, key="rainbow_T")
                r_rainbow = st.number_input("Tasa Libre de Riesgo (%)", value=5.0, key="rainbow_r") / 100
                rainbow_type = st.radio("Tipo de Rainbow:", ["Mejor de", "Peor de"])
            with col2:
                payoff_type = st.radio("Tipo de Pago:", ["Call", "Put"])
                n_sims_rainbow = st.number_input("Simulaciones Monte Carlo", value=50000, key="rainbow_sims")
            
            st.subheader("Parámetros de los Activos")
            asset_params_rainbow = []
            for i in range(n_assets_rainbow):
                col1, col2, col3 = st.columns(3)
                with col1:
                    S0 = st.number_input(f"Precio Activo {i+1}", value=100.0, key=f"rainbow_S0_{i}")
                with col2:
                    vol = st.number_input(f"Volatilidad {i+1} (%)", value=20.0 + i*5, key=f"rainbow_vol_{i}") / 100
                with col3:
                    div = st.number_input(f"Rendimiento de Dividendos {i+1} (%)", value=2.0, key=f"rainbow_div_{i}") / 100
                asset_params_rainbow.append([S0, vol, div])
            
            st.subheader("Estructura de Correlación")
            avg_correlation = st.slider("Correlación Promedio", -0.5, 0.9, 0.3, 0.1)
            
            if st.button("Valorar Opción Rainbow"):
                corr_rainbow = np.full((n_assets_rainbow, n_assets_rainbow), avg_correlation)
                np.fill_diagonal(corr_rainbow, 1.0)
                S0_rainbow = np.array([p[0] for p in asset_params_rainbow])
                vols_rainbow = np.array([p[1] for p in asset_params_rainbow])
                divs_rainbow = np.array([p[2] for p in asset_params_rainbow])
                
                np.random.seed(42)
                L = np.linalg.cholesky(corr_rainbow)
                Z = np.random.standard_normal((n_sims_rainbow, n_assets_rainbow))
                Z_corr = Z @ L.T
                ST_rainbow = np.zeros((n_sims_rainbow, n_assets_rainbow))
                
                for i in range(n_assets_rainbow):
                    ST_rainbow[:, i] = S0_rainbow[i] * np.exp(
                        (r_rainbow - divs_rainbow[i] - 0.5*vols_rainbow[i]**2)*T_rainbow +
                        vols_rainbow[i]*np.sqrt(T_rainbow)*Z_corr[:, i]
                    )
                
                if rainbow_type == "Mejor de":
                    selected_prices = np.max(ST_rainbow, axis=1)
                else:
                    selected_prices = np.min(ST_rainbow, axis=1)
                
                if payoff_type == "Call":
                    payoffs = np.maximum(selected_prices - K_rainbow, 0)
                else:
                    payoffs = np.maximum(K_rainbow - selected_prices, 0)
                
                rainbow_price = np.exp(-r_rainbow*T_rainbow)*np.mean(payoffs)
                standard_error = np.exp(-r_rainbow*T_rainbow)*np.std(payoffs)/np.sqrt(n_sims_rainbow)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precio Opción Rainbow", f"${rainbow_price:.4f}")
                    st.metric("Error Estándar", f"±${1.96*standard_error:.4f}")
                with col2:
                    st.metric("Probabilidad de Ejercicio", f"{np.mean(payoffs>0)*100:.1f}%")
                    st.metric("Pago Promedio (si se ejerce)", f"${np.mean(payoffs[payoffs>0]):.2f}")
                with col3:
                    st.metric(f"Precio Promedio {rainbow_type}", f"${np.mean(selected_prices):.2f}")
                
                fig_rainbow = px.histogram(selected_prices, nbins=50, 
                                           title=f"Distribución de Precios del Activo {rainbow_type}")
                fig_rainbow.add_vline(x=K_rainbow, line_dash="dash", annotation_text="Precio de Ejercicio")
                st.plotly_chart(fig_rainbow, use_container_width=True)

        # --------------------- OPCIÓN CLIQUET ---------------------
        elif exotic_type == "Opción Cliquet":
            st.write("**Opción Cliquet (Ratchet)**")
            st.write("Opción que asegura ganancias en fechas de reajuste, proporcionando protección a la baja con participación al alza.")
            
            col1, col2 = st.columns(2)
            with col1:
                S0_cliquet = st.number_input("Precio Inicial del Activo", value=100.0, key="cliquet_S0")
                T_cliquet = st.number_input("Tiempo Total (años)", value=3.0, key="cliquet_T")
                n_resets = st.number_input("Número de Reajustes", value=12, min_value=2, key="cliquet_resets")
                global_floor = st.number_input("Piso Global (%)", value=-10.0, key="cliquet_floor") / 100
            with col2:
                local_floor = st.number_input("Piso Local (%)", value=-5.0, key="cliquet_local_floor") / 100
                local_cap = st.number_input("Tope Local (%)", value=15.0, key="cliquet_local_cap") / 100
                volatility_cliquet = st.number_input("Volatilidad (%)", value=25.0, key="cliquet_vol") / 100
                r_cliquet = st.number_input("Tasa Libre de Riesgo (%)", value=4.0, key="cliquet_r") / 100
            
            if st.button("Valorar Opción Cliquet"):
                dt = T_cliquet / n_resets
                n_sims_cliquet = 50000
                np.random.seed(42)
                
                # Generar caminos del activo
                Z = np.random.standard_normal((n_sims_cliquet, n_resets))
                S = np.zeros((n_sims_cliquet, n_resets + 1))
                S[:, 0] = S0_cliquet
                
                # Registrar rendimientos Cliquet
                cliquet_returns = np.zeros((n_sims_cliquet, n_resets))
                
                for i in range(n_resets):
                    S[:, i + 1] = S[:, i] * np.exp(
                        (r_cliquet - 0.5 * volatility_cliquet**2) * dt +
                        volatility_cliquet * np.sqrt(dt) * Z[:, i]
                    )
                    
                    period_return = (S[:, i + 1] - S[:, i]) / S[:, i]
                    
                    capped_return = np.minimum(period_return, local_cap)
                    floored_return = np.maximum(capped_return, local_floor)
                    
                    cliquet_returns[:, i] = floored_return
                
                total_cliquet_return = np.sum(cliquet_returns, axis=1)
                
                final_payoff = np.maximum(total_cliquet_return, global_floor)
                
                cliquet_price = np.exp(-r_cliquet*T_cliquet) * np.mean(final_payoff) * S0_cliquet
                
                vanilla_price = FinancialCalculations.black_scholes_call(
                    S0_cliquet, S0_cliquet, T_cliquet, r_cliquet, volatility_cliquet
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precio Opción Cliquet", f"${cliquet_price:.2f}")
                    st.metric("Rendimiento Promedio Total", f"{np.mean(total_cliquet_return)*100:.2f}%")
                with col2:
                    st.metric("Precio Call Vanilla", f"${vanilla_price:.2f}")
                    st.metric("Protección a la Baja", f"{np.mean(final_payoff >= global_floor)*100:.1f}%")
                with col3:
                    st.metric("Rendimiento Máximo", f"{np.max(total_cliquet_return)*100:.2f}%")
                    st.metric("Rendimiento Mínimo", f"{np.min(final_payoff)*100:.2f}%")
                
                fig_cliquet = px.histogram(total_cliquet_return*100, nbins=50, 
                                           title="Distribución de Rendimiento Total Cliquet")
                fig_cliquet.add_vline(x=global_floor*100, line_dash="dash", annotation_text="Piso Global")
                fig_cliquet.update_layout(xaxis_title="Rendimiento Total (%)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_cliquet, use_container_width=True)
        
        # --------------------- VARIANCE SWAP ---------------------
        elif exotic_type == "Variance Swap":
            st.write("**Variance Swap**")
            st.write("Swap que permite negociar la volatilidad realizada frente a la volatilidad implícita.")
            
            col1, col2 = st.columns(2)
            with col1:
                S0_var = st.number_input("Precio Inicial del Activo", value=100.0, key="var_S0")
                T_var = st.number_input("Tiempo a Vencimiento (años)", value=0.25, key="var_T")
                n_obs = st.number_input("Número de Observaciones", value=63, key="var_obs")
                strike_vol = st.number_input("Volatilidad Strike (%)", value=20.0, key="var_strike") / 100
            with col2:
                true_vol = st.number_input("Volatilidad Real (%)", value=25.0, key="var_true") / 100
                notional = st.number_input("Monto Notional ($)", value=1000000.0, key="var_notional")
                r_var = st.number_input("Tasa Libre de Riesgo (%)", value=5.0, key="var_r") / 100
            
            if st.button("Analizar Variance Swap"):
                dt = T_var / n_obs
                n_sims_var = 10000
                np.random.seed(42)
                
                Z = np.random.standard_normal((n_sims_var, n_obs))
                S_var = np.zeros((n_sims_var, n_obs + 1))
                S_var[:, 0] = S0_var
                
                for i in range(n_obs):
                    S_var[:, i + 1] = S_var[:, i] * np.exp(
                        (r_var - 0.5 * true_vol**2) * dt + true_vol * np.sqrt(dt) * Z[:, i]
                    )
                
                log_returns = np.log(S_var[:, 1:] / S_var[:, :-1])
                realized_variance = np.sum(log_returns**2, axis=1) * (252 / n_obs)
                
                variance_diff = realized_variance - strike_vol**2
                swap_payoff = notional * variance_diff
                pv_payoff = np.exp(-r_var*T_var) * swap_payoff
                fair_value = np.mean(pv_payoff)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Valor Justo Variance Swap", f"${fair_value:,.0f}")
                    st.metric("Volatilidad Real Promedio", f"{np.sqrt(np.mean(realized_variance))*100:.2f}%")
                with col2:
                    st.metric("Volatilidad Strike", f"{strike_vol*100:.2f}%")
                    st.metric("Volatilidad Real", f"{true_vol*100:.2f}%")
                with col3:
                    st.metric("Probabilidad de Ganancia", f"{np.mean(variance_diff>0)*100:.1f}%")
                    st.metric("Pago Máximo", f"${np.max(swap_payoff):,.0f}")
                
                realized_vol = np.sqrt(realized_variance)
                fig_var = px.histogram(realized_vol*100, nbins=50, title="Distribución de Volatilidad Realizada")
                fig_var.add_vline(x=strike_vol*100, line_dash="dash", annotation_text="Volatilidad Strike")
                fig_var.add_vline(x=true_vol*100, line_dash="dot", annotation_text="Volatilidad Real")
                fig_var.update_layout(xaxis_title="Volatilidad Realizada (%)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_var, use_container_width=True)
                
                fig_pnl = px.histogram(pv_payoff/1000, nbins=50, title="Distribución P&L Variance Swap")
                fig_pnl.add_vline(x=0, line_dash="dash", annotation_text="Punto de Equilibrio")
                fig_pnl.update_layout(xaxis_title="P&L ($000)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_pnl, use_container_width=True)

    # --------------------- MODELOS DE RIESGO CREDITICIO ---------------------
    with tab2:
        st.subheader("Modelado de Riesgo Crediticio")
        
        st.write("""
        El riesgo crediticio es la posibilidad de pérdida financiera debido al incumplimiento de un contraparte.
        Los modelos modernos de riesgo crediticio utilizan distribuciones de probabilidad para estimar 
        la probabilidad de incumplimiento y la severidad de la pérdida.
        """)
        
        credit_model = st.selectbox("Selecciona Modelo de Crédito:", 
                                   ["Modelo Merton", "Modelo de Forma Reducida", "Riesgo de Cartera"])
        
        # --------------------- MODELO MERTON ---------------------
        if credit_model == "Modelo Merton":
            st.write("**Modelo Estructural de Crédito de Merton**")
            st.write("Modela el incumplimiento como un evento que ocurre cuando el valor de la empresa cae por debajo de su deuda.")
            
            with st.expander("📊 Teoría del Modelo Merton"):
                st.write("""
                El modelo de Merton trata la equity como una opción call sobre el valor de la empresa con strike igual a la deuda.
                El incumplimiento ocurre cuando el valor de la empresa < deuda al vencimiento.
                """)
                st.latex(r"P_{default} = N\left(\frac{\ln(D/V) + (r - 0.5\sigma_V^2)T}{\sigma_V\sqrt{T}}\right)")
                st.write("Donde D = deuda, V = valor de la empresa, σ_V = volatilidad de la empresa")
            
            col1, col2 = st.columns(2)
            with col1:
                firm_value = st.number_input("Valor de la Empresa ($B)", value=10.0, key="merton_V")
                debt_value = st.number_input("Valor de la Deuda ($B)", value=6.0, key="merton_D")
                firm_volatility = st.number_input("Volatilidad de la Empresa (%)", value=25.0, key="merton_vol") / 100
            with col2:
                time_horizon = st.number_input("Horizonte Temporal (años)", value=1.0, key="merton_T")
                risk_free_rate = st.number_input("Tasa Libre de Riesgo (%)", value=3.0, key="merton_r") / 100
            
            if st.button("Calcular Probabilidad de Incumplimiento"):
                # Cálculo Merton
                d1 = (np.log(firm_value / debt_value) + 
                     (risk_free_rate + 0.5 * firm_volatility**2) * time_horizon) / (firm_volatility * np.sqrt(time_horizon))
                d2 = d1 - firm_volatility * np.sqrt(time_horizon)
                
                default_prob = norm.cdf(-d2)
                
                distance_to_default = (np.log(firm_value / debt_value) + 
                                     (risk_free_rate - 0.5 * firm_volatility**2) * time_horizon) / (firm_volatility * np.sqrt(time_horizon))
                
                equity_value = firm_value * norm.cdf(d1) - debt_value * np.exp(-risk_free_rate * time_horizon) * norm.cdf(d2)
                
                recovery_rate = 0.4  # Recuperación asumida
                credit_spread = -np.log(1 - default_prob * (1 - recovery_rate)) / time_horizon
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Probabilidad de Incumplimiento", f"{default_prob*100:.2f}%")
                    st.metric("Distancia al Default", f"{distance_to_default:.3f}")
                with col2:
                    st.metric("Valor de la Equity ($B)", f"${equity_value:.2f}")
                    leverage = debt_value / firm_value
                    st.metric("Ratio de Apalancamiento", f"{leverage:.2f}")
                with col3:
                    st.metric("Spread de Crédito", f"{credit_spread*100:.0f} bps")
                    equity_vol = equity_value / firm_value * firm_volatility / (equity_value / firm_value)
                    st.metric("Volatilidad Implícita Equity", f"{equity_vol*100:.1f}%")
                
                # Sensibilidad
                firm_value_range = np.linspace(debt_value*0.8, debt_value*2.0, 50)
                default_probs = []
                for V in firm_value_range:
                    d2_sens = (np.log(V/debt_value) + (risk_free_rate - 0.5*firm_volatility**2) * time_horizon) / (firm_volatility * np.sqrt(time_horizon))
                    prob = norm.cdf(-d2_sens)
                    default_probs.append(prob)
                
                fig_merton = go.Figure()
                fig_merton.add_trace(go.Scatter(x=firm_value_range, y=np.array(default_probs)*100, mode='lines', name='Probabilidad de Incumplimiento'))
                fig_merton.add_vline(x=debt_value, line_dash="dash", annotation_text="Nivel de Deuda")
                fig_merton.add_vline(x=firm_value, line_dash="dot", annotation_text="Valor Actual Empresa")
                
                fig_merton.update_layout(title="Probabilidad de Incumplimiento vs Valor Empresa",
                                         xaxis_title="Valor de la Empresa ($B)",
                                         yaxis_title="Probabilidad de Incumplimiento (%)")
                st.plotly_chart(fig_merton, use_container_width=True)
        
        # --------------------- MODELO FORMA REDUCIDA ---------------------
        elif credit_model == "Modelo de Forma Reducida":
            st.write("**Modelo de Forma Reducida (Intensidad)**")
            st.write("Modela el default como un evento aleatorio con intensidad estocástica (hazard rate).")
            
            col1, col2 = st.columns(2)
            with col1:
                lambda_0 = st.number_input("Tasa Inicial de Incumplimiento (%)", value=2.0, key="rf_lambda") / 100
                lambda_reversion = st.number_input("Velocidad de Reversión Media", value=0.5, key="rf_kappa")
                lambda_vol = st.number_input("Volatilidad de Hazard Rate", value=0.3, key="rf_vol")
            with col2:
                time_horizon_rf = st.number_input("Horizonte Temporal (años)", value=5.0, key="rf_T")
                n_paths_rf = st.number_input("Número de Caminos", value=10000, key="rf_paths")
                recovery_rate_rf = st.number_input("Tasa de Recuperación (%)", value=40.0, key="rf_recovery") / 100
            
            if st.button("Simular Eventos de Crédito"):
                dt = 1/252  # Simulación diaria
                n_steps_rf = int(time_horizon_rf*252)
                
                np.random.seed(42)
                lambda_paths = np.zeros((n_paths_rf, n_steps_rf+1))
                lambda_paths[:,0] = lambda_0
                
                default_times = np.full(n_paths_rf, np.inf)
                default_indicators = np.zeros(n_paths_rf)
                
                for path in range(n_paths_rf):
                    for step in range(n_steps_rf):
                        dW = np.random.normal(0, np.sqrt(dt))
                        lambda_t = max(lambda_paths[path, step], 0.0001)
                        
                        dlambda = lambda_reversion*(lambda_0 - lambda_t)*dt + lambda_vol*np.sqrt(lambda_t)*dW
                        lambda_paths[path, step+1] = lambda_t + dlambda
                        
                        if default_indicators[path] == 0:
                            default_prob_step = lambda_t*dt
                            if np.random.uniform() < default_prob_step:
                                default_times[path] = step*dt
                                default_indicators[path] = 1
                
                time_points = np.linspace(0, time_horizon_rf, 21)
                survival_probs = [np.sum(default_times>t)/n_paths_rf for t in time_points]
                
                final_default_rate = np.mean(default_indicators)*100
                avg_default_time = np.mean(default_times[default_times<np.inf])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tasa de Incumplimiento", f"{final_default_rate:.2f}%")
                    st.metric("Tiempo Promedio a Default", f"{avg_default_time:.2f} años")
                with col2:
                    survival_1y = np.sum(default_times>1.0)/n_paths_rf*100
                    st.metric("Supervivencia 1 Año", f"{survival_1y:.1f}%")
                    survival_5y = np.sum(default_times>5.0)/n_paths_rf*100
                    st.metric("Supervivencia 5 Años", f"{survival_5y:.1f}%")
                with col3:
                    avg_lambda = np.mean(lambda_paths[:,-1])*100
                    st.metric("Tasa de Incumplimiento Promedio Final", f"{avg_lambda:.2f}%")
                
                fig_survival = go.Figure()
                fig_survival.add_trace(go.Scatter(x=time_points, y=np.array(survival_probs)*100, mode='lines+markers', name='Probabilidad de Supervivencia'))
                fig_survival.update_layout(title="Curva de Supervivencia", xaxis_title="Tiempo (años)", yaxis_title="Probabilidad de Supervivencia (%)")
                st.plotly_chart(fig_survival, use_container_width=True)
                
                fig_hazard = go.Figure()
                sample_paths = min(50, n_paths_rf)
                t_grid = np.linspace(0, time_horizon_rf, n_steps_rf+1)
                for i in range(sample_paths):
                    fig_hazard.add_trace(go.Scatter(x=t_grid, y=lambda_paths[i,:]*100, mode='lines', opacity=0.3, showlegend=False))
                mean_lambda = np.mean(lambda_paths, axis=0)
                fig_hazard.add_trace(go.Scatter(x=t_grid, y=mean_lambda*100, mode='lines', line=dict(color='red', width=3), name='Tasa Media Hazard'))
                fig_hazard.update_layout(title="Evolución de Hazard Rate", xaxis_title="Tiempo (años)", yaxis_title="Hazard Rate (%)")
                st.plotly_chart(fig_hazard, use_container_width=True)

        # --------------------- RIESGO DE CARTERA ---------------------
        elif credit_model == "Riesgo de Cartera":
            st.write("**Riesgo de Cartera (Modelo de Factor Único)**")
            st.write("Modela la correlación de incumplimiento utilizando un factor de riesgo sistemático.")
            
            col1, col2 = st.columns(2)
            with col1:
                n_obligors = st.number_input("Número de Deudores", value=100, min_value=10, max_value=1000)
                avg_pd = st.number_input("PD Promedio (%)", value=2.0, key="port_pd") / 100
                correlation = st.number_input("Correlación de Activos", value=0.15, min_value=0.0, max_value=0.99)
            with col2:
                lgd = st.number_input("Pérdida Dada Default (%)", value=60.0, key="port_lgd") / 100
                exposure = st.number_input("Exposición por Deudor ($)", value=1000000.0, key="port_exp")
                n_scenarios = st.number_input("Número de Escenarios", value=10000, key="port_scenarios")
            
            if st.button("Simular Pérdidas de Cartera"):
                np.random.seed(42)
                
                # Simulación modelo de factor único
                portfolio_losses = []
                
                for scenario in range(n_scenarios):
                    # Generar factor sistemático
                    Z = np.random.normal(0, 1)
                    
                    # Generar factores idiosincráticos
                    epsilon = np.random.normal(0, 1, n_obligors)
                    
                    # Valores de los activos
                    sqrt_rho = np.sqrt(correlation)
                    sqrt_1_rho = np.sqrt(1 - correlation)
                    asset_values = sqrt_rho * Z + sqrt_1_rho * epsilon
                    
                    # Umbral de default
                    threshold = norm.ppf(avg_pd)
                    
                    # Determinar defaults
                    defaults = asset_values < threshold
                    n_defaults = np.sum(defaults)
                    
                    # Calcular pérdida de la cartera
                    portfolio_loss = n_defaults * exposure * lgd
                    portfolio_losses.append(portfolio_loss)
                
                portfolio_losses = np.array(portfolio_losses)
                
                # Métricas de riesgo
                expected_loss = np.mean(portfolio_losses)
                portfolio_std = np.std(portfolio_losses)
                var_95 = np.percentile(portfolio_losses, 95)
                var_99 = np.percentile(portfolio_losses, 99)
                expected_shortfall = np.mean(portfolio_losses[portfolio_losses >= var_99])
                
                economic_capital = var_99 - expected_loss
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pérdida Esperada", f"${expected_loss/1e6:.2f}M")
                    st.metric("Desviación Estándar Cartera", f"${portfolio_std/1e6:.2f}M")
                with col2:
                    st.metric("VaR 95%", f"${var_95/1e6:.2f}M")
                    st.metric("VaR 99%", f"${var_99/1e6:.2f}M")
                with col3:
                    st.metric("Expected Shortfall", f"${expected_shortfall/1e6:.2f}M")
                    st.metric("Capital Económico", f"${economic_capital/1e6:.2f}M")
                
                # Distribución de pérdidas
                fig_port = px.histogram(portfolio_losses/1e6, nbins=50, 
                                       title="Distribución de Pérdidas de Cartera")
                fig_port.add_vline(x=expected_loss/1e6, line_dash="dash", annotation_text="Pérdida Esperada")
                fig_port.add_vline(x=var_99/1e6, line_dash="dot", line_color="red", annotation_text="VaR 99%")
                fig_port.update_layout(xaxis_title="Pérdida de Cartera ($M)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_port, use_container_width=True)
                
                # Distribución de tasas de incumplimiento
                default_rates = [loss / (n_obligors*exposure*lgd) * 100 for loss in portfolio_losses]
                fig_default = px.histogram(default_rates, nbins=30, title="Distribución de Tasa de Default")
                fig_default.add_vline(x=avg_pd*100, line_dash="dash", annotation_text="Tasa de Default Esperada")
                fig_default.update_layout(xaxis_title="Tasa de Default (%)", yaxis_title="Frecuencia")
                st.plotly_chart(fig_default, use_container_width=True)

    # --------------------- TRADING ALGORÍTMICO ---------------------
    with tab3:
        st.subheader("Conceptos de Trading Algorítmico")
        
        st.write("""
        El trading algorítmico utiliza algoritmos informáticos para ejecutar estrategias de trading 
        basadas en reglas predefinidas, análisis de datos del mercado y modelos matemáticos.
        """)
        
        algo_topic = st.selectbox("Seleccione el Tema:", 
                                 ["Estrategia de Reversión a la Media", "Estrategia de Momentum", 
                                  "Pairs Trading", "Optimización de Portafolio"])
        
        if algo_topic == "Estrategia de Reversión a la Media":
            st.write("**Estrategia de Reversión a la Media**")
            st.write("Se basa en la suposición de que los precios volverán a su media histórica.")
            
            col1, col2 = st.columns(2)
            with col1:
                lookback_period = st.number_input("Periodo de Observación (días)", value=20, min_value=5, max_value=100)
                entry_threshold = st.number_input("Umbral de Entrada (desv. estándar)", value=2.0, min_value=0.5, max_value=5.0)
                exit_threshold = st.number_input("Umbral de Salida (desv. estándar)", value=0.5, min_value=0.1, max_value=2.0)
            with col2:
                initial_capital = st.number_input("Capital Inicial ($)", value=100000.0, min_value=1000.0)
                position_size = st.number_input("Tamaño de Posición (%)", value=10.0, min_value=1.0, max_value=50.0) / 100
                transaction_cost = st.number_input("Costo de Transacción (%)", value=0.1, min_value=0.0) / 100
            
            if st.button("Backtest Estrategia Reversión a la Media"):
                # Generar datos sintéticos de precios
                np.random.seed(42)
                n_days = 252 * 2  # 2 años
                
                prices = [100.0]
                theta = 0.1  # Velocidad de reversión
                mu = 100.0   # Media a largo plazo
                sigma = 0.02  # Volatilidad diaria
                
                for i in range(n_days - 1):
                    price_change = theta * (mu - prices[-1]) + sigma * np.random.normal()
                    new_price = prices[-1] + price_change
                    prices.append(max(new_price, 1.0))  # Evitar precios negativos
                
                prices = np.array(prices)
                dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
                
                # Estadísticas móviles
                df = pd.DataFrame({'Fecha': dates, 'Precio': prices})
                df['SMA'] = df['Precio'].rolling(window=lookback_period).mean()
                df['StdDev'] = df['Precio'].rolling(window=lookback_period).std()
                df['Z_Score'] = (df['Precio'] - df['SMA']) / df['StdDev']
                
                # Señales
                df['Señal'] = 0
                df.loc[df['Z_Score'] > entry_threshold, 'Señal'] = -1  # Vender
                df.loc[df['Z_Score'] < -entry_threshold, 'Señal'] = 1   # Comprar
                df.loc[abs(df['Z_Score']) < exit_threshold, 'Señal'] = 0  # Salir
                
                # Backtest
                position = 0
                cash = initial_capital
                equity_curve = []
                trades = []
                
                for i in range(len(df)):
                    current_price = df.iloc[i]['Precio']
                    signal = df.iloc[i]['Señal']
                    
                    if signal != 0 and position == 0:  # Entrar posición
                        position_value = cash * position_size
                        shares = position_value / current_price
                        transaction_costs = position_value * transaction_cost
                        
                        if signal == 1:  # Comprar
                            position = shares
                            cash -= position_value + transaction_costs
                        else:  # Venta en corto
                            position = -shares
                            cash += position_value - transaction_costs
                        
                        trades.append({'Fecha': df.iloc[i]['Fecha'], 'Tipo': 'Comprar' if signal==1 else 'Vender',
                                       'Precio': current_price, 'Acciones': abs(shares), 'Valor': position_value})
                    
                    elif signal == 0 and position != 0:  # Cerrar posición
                        position_value = abs(position) * current_price
                        transaction_costs = position_value * transaction_cost
                        
                        if position > 0:
                            cash += position_value - transaction_costs
                        else:
                            cash -= position_value + transaction_costs
                        
                        trades.append({'Fecha': df.iloc[i]['Fecha'], 'Tipo': 'Cerrar', 'Precio': current_price,
                                       'Acciones': abs(position), 'Valor': position_value})
                        
                        position = 0
                    
                    # Equidad total
                    if position != 0:
                        position_value = position * current_price
                        total_equity = cash + position_value
                    else:
                        total_equity = cash
                    
                    equity_curve.append(total_equity)
                
                # Métricas
                df['Equidad'] = equity_curve
                df['Retornos'] = df['Equidad'].pct_change()
                df['BuyHold'] = initial_capital * (df['Precio']/df.iloc[0]['Precio'])
                
                total_return = (equity_curve[-1] / initial_capital - 1) * 100
                buy_hold_return = (df.iloc[-1]['BuyHold'] / initial_capital - 1) * 100
                annual_return = (equity_curve[-1]/initial_capital)**(252/len(df)) - 1
                annual_vol = df['Retornos'].std() * np.sqrt(252)
                sharpe_ratio = annual_return / annual_vol if annual_vol>0 else 0
                max_dd = ((df['Equidad'].expanding().max() - df['Equidad']) / df['Equidad'].expanding().max()).max()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retorno Total", f"{total_return:.2f}%")
                    st.metric("Retorno Anual", f"{annual_return*100:.2f}%")
                with col2:
                    st.metric("Buy & Hold", f"{buy_hold_return:.2f}%")
                    st.metric("Volatilidad Anual", f"{annual_vol*100:.2f}%")
                with col3:
                    st.metric("Ratio de Sharpe", f"{sharpe_ratio:.3f}")
                    st.metric("Máximo Drawdown", f"{max_dd*100:.2f}%")
                
                # Gráficos
                fig_backtest = go.Figure()
                fig_backtest.add_trace(go.Scatter(x=df['Fecha'], y=df['Precio'], mode='lines', name='Precio Acción'))
                fig_backtest.add_trace(go.Scatter(x=df['Fecha'], y=df['SMA'], mode='lines', name='Media Móvil'))
                
                buy_signals = df[df['Señal']==1]
                sell_signals = df[df['Señal']==-1]
                
                fig_backtest.add_trace(go.Scatter(x=buy_signals['Fecha'], y=buy_signals['Precio'],
                                                 mode='markers', marker=dict(color='green', size=8),
                                                 name='Señal Compra'))
                fig_backtest.add_trace(go.Scatter(x=sell_signals['Fecha'], y=sell_signals['Precio'],
                                                 mode='markers', marker=dict(color='red', size=8),
                                                 name='Señal Venta'))
                
                fig_backtest.update_layout(title="Estrategia Reversión a la Media - Precio y Señales",
                                          xaxis_title="Fecha", yaxis_title="Precio ($)")
                st.plotly_chart(fig_backtest, use_container_width=True)
                
                # Curva de equidad
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(x=df['Fecha'], y=df['Equidad'], mode='lines', name='Estrategia'))
                fig_equity.add_trace(go.Scatter(x=df['Fecha'], y=df['BuyHold'], mode='lines', name='Buy & Hold'))
                fig_equity.update_layout(title="Comparación Curva de Equidad",
                                        xaxis_title="Fecha", yaxis_title="Valor de Portafolio ($)")
                st.plotly_chart(fig_equity, use_container_width=True)
                
                # Resumen de operaciones
                if trades:
                    st.subheader("Últimas Operaciones")
                    trades_df = pd.DataFrame(trades[-10:])
                    st.dataframe(trades_df, use_container_width=True)

        elif algo_topic == "Estrategia de Momentum":
            st.write("**Estrategia de Momentum**")
            st.write("Comprar activos que han tenido buen desempeño y vender los que han tenido mal desempeño durante un periodo de observación.")
            
            # Parámetros de la estrategia
            lookback = st.number_input("Periodo de Observación (días)", value=90, min_value=10, max_value=250)
            top_n = st.number_input("Número de activos top a comprar", value=5, min_value=1)
            bottom_n = st.number_input("Número de activos bottom a vender", value=5, min_value=1)
            initial_capital = st.number_input("Capital Inicial ($)", value=100000.0, min_value=1000.0)
            position_size = st.number_input("Tamaño de Posición por Activo (%)", value=10.0, min_value=1.0, max_value=50.0) / 100
            transaction_cost = st.number_input("Costo de Transacción (%)", value=0.1, min_value=0.0) / 100
            n_assets_momentum = st.number_input("Número de Activos a Simular", value=10, min_value=2)
            
            if st.button("Backtest Estrategia Momentum"):
                np.random.seed(42)
                n_days = 252 * 2
                
                # Generar precios sintéticos
                prices = np.zeros((n_days, n_assets_momentum))
                for i in range(n_assets_momentum):
                    prices[:, i] = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_days))
                
                dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
                df_prices = pd.DataFrame(prices, columns=[f'Activo {i+1}' for i in range(n_assets_momentum)])
                df_prices['Fecha'] = dates
                
                # Calcular retornos pasados
                momentum_scores = df_prices.iloc[:, :-1].pct_change(lookback).iloc[lookback:]
                
                equity_curve = []
                cash = initial_capital
                positions = np.zeros(n_assets_momentum)
                
                for idx in range(len(momentum_scores)):
                    scores = momentum_scores.iloc[idx].values
                    buy_idx = np.argsort(scores)[-top_n:]
                    sell_idx = np.argsort(scores)[:bottom_n]
                    
                    # Liquidar posiciones previas
                    current_prices = prices[idx+lookback]
                    cash += np.sum(positions * current_prices) - np.sum(np.abs(positions * current_prices) * transaction_cost)
                    positions = np.zeros(n_assets_momentum)
                    
                    # Abrir nuevas posiciones
                    for i in buy_idx:
                        positions[i] = (cash * position_size) / current_prices[i]
                        cash -= current_prices[i] * positions[i] * (1 + transaction_cost)
                    for i in sell_idx:
                        positions[i] = -(cash * position_size) / current_prices[i]
                        cash += current_prices[i] * abs(positions[i]) * (1 - transaction_cost)
                    
                    total_equity = cash + np.sum(positions * current_prices)
                    equity_curve.append(total_equity)
                
                df_equity = pd.DataFrame({'Fecha': df_prices['Fecha'].iloc[lookback:], 'Equidad': equity_curve})
                
                # Métricas
                total_return = (equity_curve[-1]/initial_capital - 1) * 100
                returns = df_equity['Equidad'].pct_change().dropna()
                annual_return = (equity_curve[-1]/initial_capital)**(252/len(returns)) - 1
                annual_vol = returns.std() * np.sqrt(252)
                sharpe_ratio = annual_return / annual_vol if annual_vol>0 else 0
                max_dd = ((df_equity['Equidad'].expanding().max() - df_equity['Equidad']) / df_equity['Equidad'].expanding().max()).max()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Retorno Total", f"{total_return:.2f}%")
                    st.metric("Retorno Anual", f"{annual_return*100:.2f}%")
                with col2:
                    st.metric("Volatilidad Anual", f"{annual_vol*100:.2f}%")
                    st.metric("Ratio de Sharpe", f"{sharpe_ratio:.3f}")
                with col3:
                    st.metric("Máximo Drawdown", f"{max_dd*100:.2f}%")
                
                # Curva de equidad
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=df_equity['Fecha'], y=df_equity['Equidad'], mode='lines', name='Equidad'))
                fig_eq.update_layout(title="Curva de Equidad - Estrategia Momentum", xaxis_title="Fecha", yaxis_title="Valor de Portafolio ($)")
                st.plotly_chart(fig_eq, use_container_width=True)
        
        elif algo_topic == "Pairs Trading":
            st.write("**Estrategia Pairs Trading**")
            st.write("Estrategia neutral al mercado que opera sobre el spread entre dos activos correlacionados.")
            
            if st.button("Generar Ejemplo Pairs Trading"):
                np.random.seed(42)
                n_days = 252
                dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
                
                # Generar dos activos correlacionados
                correlation = 0.8
                returns_A = np.random.normal(0.0005, 0.02, n_days)
                returns_B = correlation*returns_A + np.sqrt(1-correlation**2)*np.random.normal(0.0005,0.02,n_days)
                
                # Agregar divergencia temporal
                divergence_period = slice(100, 150)
                returns_B[divergence_period] += np.linspace(0, -0.05, 50)
                returns_B[slice(150, 200)] += np.linspace(-0.05,0,50)
                
                price_A = 100 * np.cumprod(1 + returns_A)
                price_B = 100 * np.cumprod(1 + returns_B)
                
                df_pairs = pd.DataFrame({'Fecha': dates, 'Activo_A': price_A, 'Activo_B': price_B})
                
                # Spread y Z-Score
                df_pairs['Spread'] = df_pairs['Activo_A'] - df_pairs['Activo_B']
                df_pairs['Spread_MA'] = df_pairs['Spread'].rolling(window=20).mean()
                df_pairs['Spread_Std'] = df_pairs['Spread'].rolling(window=20).std()
                df_pairs['Z_Score'] = (df_pairs['Spread'] - df_pairs['Spread_MA']) / df_pairs['Spread_Std']
                
                # Señales
                entry_threshold = 2.0
                exit_threshold = 0.5
                df_pairs['Señal'] = 0
                df_pairs.loc[df_pairs['Z_Score'] > entry_threshold, 'Señal'] = -1
                df_pairs.loc[df_pairs['Z_Score'] < -entry_threshold, 'Señal'] = 1
                df_pairs.loc[abs(df_pairs['Z_Score']) < exit_threshold, 'Señal'] = 0
                
                # Visualización precios
                fig_pairs = go.Figure()
                fig_pairs.add_trace(go.Scatter(x=df_pairs['Fecha'], y=df_pairs['Activo_A'], mode='lines', name='Activo A'))
                fig_pairs.add_trace(go.Scatter(x=df_pairs['Fecha'], y=df_pairs['Activo_B'], mode='lines', name='Activo B'))
                fig_pairs.update_layout(title="Pairs Trading - Precios", xaxis_title="Fecha", yaxis_title="Precio ($)")
                st.plotly_chart(fig_pairs, use_container_width=True)
                
                # Spread Z-Score
                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(x=df_pairs['Fecha'], y=df_pairs['Z_Score'], mode='lines', name='Z-Score'))
                fig_spread.add_hline(y=entry_threshold, line_dash="dash", line_color="red", annotation_text="Umbral Entrada")
                fig_spread.add_hline(y=-entry_threshold, line_dash="dash", line_color="red")
                fig_spread.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_spread.update_layout(title="Z-Score del Spread", xaxis_title="Fecha", yaxis_title="Z-Score")
                st.plotly_chart(fig_spread, use_container_width=True)
                
                # Métricas
                correlation_coef = np.corrcoef(returns_A, returns_B)[0,1]
                mean_reversion_speed = abs(df_pairs['Z_Score'].diff().mean())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Coeficiente de Correlación", f"{correlation_coef:.3f}")
                    st.metric("Spread Medio", f"{df_pairs['Spread'].mean():.2f}")
                with col2:
                    st.metric("Volatilidad del Spread", f"{df_pairs['Spread'].std():.2f}")
                    st.metric("Máximo Z-Score", f"{df_pairs['Z_Score'].abs().max():.2f}")
                with col3:
                    signal_frequency = (df_pairs['Señal']!=0).sum()/len(df_pairs)*100
                    st.metric("Frecuencia de Señales", f"{signal_frequency:.1f}%")
        
        elif algo_topic == "Optimización de Portafolio":
            st.write("**Optimización de Portafolio**")
            st.write("Construir un portafolio óptimo usando optimización media-varianza (Markowitz).")
            
            n_assets = st.number_input("Número de Activos", value=5, min_value=2)
            initial_capital = st.number_input("Capital Inicial ($)", value=100000.0, min_value=1000.0)
            
            if st.button("Ejecutar Optimización"):
                np.random.seed(42)
                mean_returns = np.random.uniform(0.05, 0.15, n_assets)
                cov_matrix = np.random.uniform(0.01, 0.05, (n_assets, n_assets))
                cov_matrix = (cov_matrix + cov_matrix.T)/2
                np.fill_diagonal(cov_matrix, np.random.uniform(0.02, 0.06, n_assets))
                
                def portfolio_metrics(weights):
                    port_return = np.dot(weights, mean_returns)
                    port_vol = np.sqrt(weights @ cov_matrix @ weights)
                    return port_vol, port_return
                
                def min_vol(weights):
                    return portfolio_metrics(weights)[0]
                
                constraints = ({'type':'eq','fun':lambda w: np.sum(w)-1})
                bounds = tuple((0,1) for _ in range(n_assets))
                result = minimize(min_vol, x0=np.ones(n_assets)/n_assets, bounds=bounds, constraints=constraints)
                opt_weights = result.x
                opt_vol, opt_return = portfolio_metrics(opt_weights)
                
                # Mostrar resultados
                st.subheader("Pesos Óptimos del Portafolio")
                for i, w in enumerate(opt_weights):
                    st.write(f"Activo {i+1}: {w*100:.2f}%")
                
                st.subheader("Métricas del Portafolio")
                st.write(f"Retorno Esperado: {opt_return*100:.2f}%")
                st.write(f"Volatilidad Esperada: {opt_vol*100:.2f}%")
                st.write(f"Ratio de Sharpe (Rf=0): {opt_return/opt_vol:.3f}")
                
                n_sims = 1000
                portfolio_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_sims) @ opt_weights
                portfolio_values = initial_capital*(1+portfolio_returns)
                
                fig_port = px.histogram(portfolio_values, nbins=50, title="Distribución de Valor del Portafolio Óptimo")
                st.plotly_chart(fig_port, use_container_width=True)

    with tab4:
        st.subheader("Síntesis del Curso y Aplicaciones Avanzadas")
        
        st.write("""
        Esta sección final integra los conceptos de todas las clases anteriores para demostrar 
        cómo las técnicas de ingeniería financiera se aplican conjuntamente en casos reales.
        """)
        
        synthesis_topic = st.selectbox("Selecciona un Tema de Integración:", 
                                      ["Mesa de Derivados Completa", "Marco de Gestión de Riesgos", "Revisión del Curso"])
        
        if synthesis_topic == "Mesa de Derivados Completa":
            st.write("**Mesa Integrada de Derivados**")
            st.write("Combinar valuación, cobertura y gestión de riesgos para una operación de trading completa.")
            
            # Configuración del portafolio
            st.subheader("Portafolio de la Mesa")
            
            # Ejemplo de book de derivados
            book_data = {
                'Instrumento': ['Opciones Call SPX', 'Forwards EUR/USD', 'Swaps de Tasas', 'Credit Default Swaps'],
                'Nocional ($M)': [50.0, 100.0, 200.0, 25.0],
                'Delta': [25.0, -15.0, 0.0, -5.0],
                'Gamma': [2.5, 0.0, 0.0, 0.0],
                'Vega': [15.0, 0.0, 50.0, 8.0],
                'Theta': [-0.8, 0.1, -0.2, -0.1]
            }
            
            book_df = pd.DataFrame(book_data)
            st.dataframe(book_df, use_container_width=True)
            
            # Cálculo de riesgos
            total_delta = book_df['Delta'].sum()
            total_gamma = book_df['Gamma'].sum()
            total_vega = book_df['Vega'].sum()
            total_theta = book_df['Theta'].sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Delta Total", f"{total_delta:.1f}")
            with col2:
                st.metric("Gamma Total", f"{total_gamma:.1f}")
            with col3:
                st.metric("Vega Total", f"{total_vega:.1f}")
            with col4:
                st.metric("Theta Total", f"{total_theta:.1f}")
            
            # Análisis de escenarios
            st.subheader("Análisis de Escenarios")
            
            scenarios = {
                'Caso Base': {'Acciones': 0, 'FX': 0, 'Tasas': 0, 'Vol': 0, 'Crédito': 0},
                'Riesgo-On': {'Acciones': 5, 'FX': 2, 'Tasas': 10, 'Vol': -20, 'Crédito': -50},
                'Riesgo-Off': {'Acciones': -10, 'FX': -5, 'Tasas': -25, 'Vol': 50, 'Crédito': 100},
                'Banco Central': {'Acciones': 2, 'FX': -3, 'Tasas': 25, 'Vol': 10, 'Crédito': 20}
            }
            
            scenario_pnl = []
            for scenario_name, changes in scenarios.items():
                pnl = (total_delta*changes['Acciones']*0.01 +
                       total_vega*changes['Vol']*0.01 +
                       total_theta*1 +
                       0.5*total_gamma*(changes['Acciones']*0.01)**2)
                
                scenario_pnl.append({'Escenario': scenario_name, 'P&L ($000)': pnl})
            
            scenario_df = pd.DataFrame(scenario_pnl)
            st.dataframe(scenario_df, use_container_width=True)
            
            # Visualización P&L
            fig_pnl = px.bar(scenario_df, x='Escenario', y='P&L ($000)', 
                             title='Análisis P&L por Escenario',
                             color='P&L ($000)', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        elif synthesis_topic == "Marco de Gestión de Riesgos":
            st.write("**Marco Integral de Gestión de Riesgos**")
            st.write("Integrar todos los tipos de riesgos y técnicas de gestión estudiadas en el curso.")
            
            st.subheader("Tablero de Riesgos")
            
            # Métricas de ejemplo
            risk_metrics = {
                'Riesgo de Mercado': {
                    'VaR (1-día, 95%)': '$2.5M',
                    'Expected Shortfall': '$4.2M',
                    'Beta': '1.15',
                    'Duration': '4.2 años'
                },
                'Riesgo de Crédito': {
                    'Pérdida Esperada': '$800K',
                    'Pérdida Inesperada': '$3.2M',
                    'Probabilidad de Default': '2.3%',
                    'Credit VaR': '$4.1M'
                },
                'Riesgo Operativo': {
                    'Capital Op Risk': '$5.0M',
                    'Puntaje Frecuencia': '3.2/5',
                    'Puntaje Severidad': '2.8/5',
                    'Indicadores Clave': '85% Verde'
                },
                'Riesgo de Liquidez': {
                    'LCR': '125%',
                    'NSFR': '118%',
                    'Brecha de Financiamiento': '$0M',
                    'Buffer de Liquidez': '$25M'
                }
            }
            
            # Mostrar métricas en pestañas
            risk_tabs = st.tabs(list(risk_metrics.keys()))
            
            for i, (risk_type, metrics) in enumerate(risk_metrics.items()):
                with risk_tabs[i]:
                    cols = st.columns(len(metrics))
                    for j, (metric, value) in enumerate(metrics.items()):
                        with cols[j]:
                            st.metric(metric, value)
            
            # Resultados de pruebas de estrés
            st.subheader("Resultados Integrales de Stress Testing")
            
            stress_results = {
                'Escenario de Stress': ['Crisis Financiera 2008', 'Pandemia COVID-19', 'Shock de Tasas', 
                                       'Crisis Deuda Soberana', 'Ataque Cibernético'],
                'Impacto Mercado ($M)': [-15.2, -8.9, -12.1, -6.8, -2.1],
                'Impacto Crédito ($M)': [-8.5, -12.3, -4.2, -9.7, -1.5],
                'Impacto Operativo ($M)': [-1.0, -2.1, -0.5, -1.2, -8.9],
                'Impacto Total ($M)': [-24.7, -23.3, -16.8, -17.7, -12.5]
            }
            
            stress_df = pd.DataFrame(stress_results)
            st.dataframe(stress_df, use_container_width=True)
            
            # Marco de apetito de riesgo
            st.subheader("Marco de Apetito de Riesgo")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Límites de Riesgo:**")
                st.write("- VaR Riesgo de Mercado: $5.0M (Actual: $2.5M) ✅")
                st.write("- VaR Riesgo de Crédito: $8.0M (Actual: $4.1M) ✅") 
                st.write("- Capital Riesgo Operativo: $10.0M (Actual: $5.0M) ✅")
                st.write("- Liquidity Coverage Ratio: >110% (Actual: 125%) ✅")
            
            with col2:
                st.write("**Monitoreo de Riesgos:**")
                st.write("- Reporte diario de VaR ✅")
                st.write("- Stress testing semanal ✅")
                st.write("- Revisión mensual ALCO ✅")
                st.write("- Validación trimestral de modelos ✅")
        
        elif synthesis_topic == "Revisión del Curso":
            st.write("**Revisión del Curso de Ingeniería Financiera**")
            st.write("Resumen de conceptos clave y sus aplicaciones prácticas.")


            modules_summary = {
                'Clase 1: Introducción': {
                    'Conceptos Clave': ['Valor del dinero en el tiempo', 'Relación riesgo-retorno', 'Mercados financieros'],
                    'Aplicaciones': ['Decisiones de inversión', 'Valoración básica', 'Evaluación de riesgo'],
                    'Herramientas': ['Cálculos de valor presente', 'Fórmulas de valor futuro', 'Estadísticas básicas']
                },
                'Clase 2: Valor del Dinero': {
                    'Conceptos Clave': ['Valor presente/futuro', 'Anualidades', 'Capitalización'],
                    'Aplicaciones': ['Valuación de bonos', 'Cálculos de préstamos', 'Análisis de inversión'],
                    'Herramientas': ['Calculadoras financieras', 'Funciones de Excel', 'Librerías Python']
                },
                'Clase 3: Valuación de Bonos': {
                    'Conceptos Clave': ['Precio de bonos', 'Medidas de rendimiento', 'Duración/convexidad'],
                    'Aplicaciones': ['Trading de renta fija', 'Gestión de portafolio', 'Cobertura de riesgo'],
                    'Herramientas': ['Modelos de precios de bonos', 'Cálculos de rendimiento', 'Análisis de duración']
                },
                'Clase 4: Valuación de Acciones': {
                    'Conceptos Clave': ['Modelo de Dividendos Descontados (DDM)', 'Análisis DCF', 'Valuación relativa'],
                    'Aplicaciones': ['Investigación de acciones', 'Banca de inversión', 'Construcción de portafolios'],
                    'Herramientas': ['Modelos de valoración', 'Ratios financieros', 'Análisis de escenarios']
                },
                'Clase 5: Derivados': {
                    'Conceptos Clave': ['Forwards y Futuros', 'Swaps', 'Gestión de riesgos'],
                    'Aplicaciones': ['Cobertura', 'Especulación', 'Arbitraje'],
                    'Herramientas': ['Fórmulas de precios', 'Cálculo de griegas', 'Métodos de Monte Carlo']
                },
                'Clase 6: Opciones': {
                    'Conceptos Clave': ['Modelo Black-Scholes', 'Griegas de opciones', 'Estrategias de trading'],
                    'Aplicaciones': ['Trading de opciones', 'Cobertura de portafolio', 'Trading de volatilidad'],
                    'Herramientas': ['Modelos de precios de opciones', 'Análisis de griegas', 'Backtesting de estrategias']
                },
                'Clase 7: Teoría de Portafolio': {
                    'Conceptos Clave': ['Teoría Moderna de Portafolio (MPT)', 'Frontera Eficiente', 'CAPM'],
                    'Aplicaciones': ['Asignación de activos', 'Medición de desempeño', 'Presupuesto de riesgo'],
                    'Herramientas': ['Algoritmos de optimización', 'Modelos de riesgo', 'Atribución de rendimiento']
                },
                'Clase 8: Gestión de Riesgos': {
                    'Conceptos Clave': ['VaR', 'Stress testing', 'Métricas de riesgo'],
                    'Aplicaciones': ['Límites de riesgo', 'Asignación de capital', 'Cumplimiento regulatorio'],
                    'Herramientas': ['Sistemas de riesgo', 'Backtesting', 'Validación de modelos']
                },
                'Clase 9: Monte Carlo': {
                    'Conceptos Clave': ['Simulación aleatoria', 'Precios dependientes de trayectoria', 'Generación de escenarios'],
                    'Aplicaciones': ['Derivados complejos', 'Evaluación de riesgo', 'Validación de modelos'],
                    'Herramientas': ['Generación de números aleatorios', 'Reducción de varianza', 'Análisis de convergencia']
                },
                'Clase 10: Temas Avanzados': {
                    'Conceptos Clave': ['Derivados exóticos', 'Riesgo de crédito', 'Trading algorítmico'],
                    'Aplicaciones': ['Productos estructurados', 'Modelos de crédito', 'Estrategias cuantitativas'],
                    'Herramientas': ['Modelos avanzados', 'Machine learning', 'Sistemas de alta frecuencia']
                }
            }


            selected_module = st.selectbox("Selecciona Módulo para Detalle:", list(modules_summary.keys()))
            module_info = modules_summary[selected_module]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Conceptos Clave")
                for concept in module_info['Conceptos Clave']:
                    st.write(f"• {concept}")
            
            with col2:
                st.subheader("Aplicaciones")
                for app in module_info['Aplicaciones']:
                    st.write(f"• {app}")
            
            with col3:
                st.subheader("Herramientas")
                for tool in module_info['Herramientas']:
                    st.write(f"• {tool}")
            
            # Aplicaciones profesionales
            st.subheader("Aplicaciones Profesionales")
            career_paths = {
                'Banca de Inversión': ['Modelos de valoración', 'Análisis DCF', 'Análisis comparables'],
                'Gestión de Activos': ['Optimización de portafolio', 'Gestión de riesgos', 'Atribución de rendimiento'],
                'Trading': ['Pricing de derivados', 'Gestión de riesgos', 'Estrategias algorítmicas'],
                'Gestión de Riesgos': ['Modelos VaR', 'Stress testing', 'Cumplimiento regulatorio'],
                'Investigación Cuantitativa': ['Modelos matemáticos', 'Análisis estadístico', 'Desarrollo de algoritmos'],
                'Finanzas Corporativas': ['Presupuesto de capital', 'Costo de capital', 'Planificación financiera']
            }
            
            selected_career = st.selectbox("Explora Carrera Profesional:", list(career_paths.keys()))
            st.write(f"**Habilidades Clave para {selected_career}:**")
            for skill in career_paths[selected_career]:
                st.write(f"✓ {skill}")

            # Evaluación Final del Curso
            st.subheader("Evaluación Final del Curso")

            q1 = st.radio(
                "¿Qué técnica es más adecuada para valorar un derivado dependiente del camino?",
                ["Fórmula de Black-Scholes", "Simulación Monte Carlo", "Árbol binomial", "Solución analítica cerrada"],
                key="final_q1"
            )

            q2 = st.radio(
                "En la optimización de portafolios, la frontera eficiente representa:",
                ["Todos los portafolios posibles", "Portafolios de máximo retorno", "Portafolios de mínimo riesgo", "Combinaciones óptimas riesgo-retorno"],
                key="final_q2"
            )

            q3 = st.radio(
                "¿Qué medida de riesgo captura mejor el riesgo de cola que el VaR?",
                ["Desviación estándar", "Beta", "Expected Shortfall", "Correlación"],
                key="final_q3"
            )

            q4 = st.radio(
                "El propósito principal de los derivados financieros es:",
                ["Solo especulación", "Gestión de riesgo y descubrimiento de precios", "Solo arbitraje", "Cumplimiento regulatorio"],
                key="final_q4"
            )

            if st.button("Enviar Evaluación Final"):
                score = 0
                
                if q1 == "Simulación Monte Carlo":
                    st.success("P1: ¡Correcto! ✅")
                    score += 1
                else:
                    st.error("P1: Incorrecto. La simulación Monte Carlo es ideal para derivados dependientes del camino.")
                
                if q2 == "Combinaciones óptimas riesgo-retorno":
                    st.success("P2: ¡Correcto! ✅")
                    score += 1
                else:
                    st.error("P2: Incorrecto. La frontera eficiente muestra combinaciones óptimas riesgo-retorno.")
                
                if q3 == "Expected Shortfall":
                    st.success("P3: ¡Correcto! ✅")
                    score += 1
                else:
                    st.error("P3: Incorrecto. Expected Shortfall mide la severidad de las pérdidas en la cola.")
                
                if q4 == "Gestión de riesgo y descubrimiento de precios":
                    st.success("P4: ¡Correcto! ✅")
                    score += 1
                else:
                    st.error("P4: Incorrecto. Los derivados cumplen múltiples propósitos, incluyendo la gestión de riesgo.")
                
                st.write(f"Tu puntuación final: {score}/4 ({score/4*100:.0f}%)")
                
                if score >= 3:
                    st.balloons()
                    progress_tracker.mark_class_completed("Clase 10: Temas Avanzados")
                    progress_tracker.set_class_score("Clase 10: Temas Avanzados", (score/4) * 100)
                    
                    # Verificar si se completaron todas las clases
                    completed = progress_tracker.get_completed_classes()
                    if len(completed) == 10:
                        st.success("🎉🎉🎉 ¡FELICIDADES! 🎉🎉🎉")
                        st.success("¡Has completado exitosamente todo el curso de Ingeniería Financiera!")
                        st.success("Ahora estás capacitado en:")
                        st.success("✓ Finanzas Matemáticas • ✓ Valoración de Derivados • ✓ Gestión de Riesgo")
                        st.success("✓ Teoría de Portafolios • ✓ Métodos de Monte Carlo • ✓ Aplicaciones Avanzadas")
                        
                        # Generar certificado
                        completion_date = pd.Timestamp.now().strftime("%d de %B de %Y")
                        st.markdown(f"""
                        ---
                        ## 🏆 CERTIFICADO DE FINALIZACIÓN 🏆
                        
                        **Se certifica que has completado exitosamente**
                        
                        # Curso de Ingeniería Financiera
                        **Fundamentos y Aplicaciones Avanzadas**
                        
                        Fecha de finalización: **{completion_date}**
                        
                        **Cobertura del Curso:**
                        - 10 clases completas
                        - Fundamentos matemáticos
                        - Aplicaciones prácticas
                        - Ejercicios interactivos
                        - Estudios de caso del mundo real
                        
                        ---
                        """)
                    else:
                        st.success("🎉 ¡Excelente! Has completado los temas avanzados.")
                else:
                    st.info("Revisa el material del curso y vuelve a intentar la evaluación.")

            # Descarga de materiales
            st.sidebar.markdown("---")
            st.sidebar.subheader("📚 Recursos Avanzados")

            codigo_avanzado = """
            # Aplicaciones Avanzadas de Ingeniería Financiera

            import numpy as np
            from scipy.optimize import minimize

            def probabilidad_default_merton(V, D, r, sigma_V, T):
                '''Probabilidad de default según Merton'''
                from scipy.stats import norm
                
                d2 = (np.log(V/D) + (r - 0.5*sigma_V**2)*T) / (sigma_V*np.sqrt(T))
                return norm.cdf(-d2)

            def opcion_rainbow_mc(S0_list, K, T, r, sigma_list, corr_matrix, n_sims, tipo_opcion='best_of_calls'):
                '''Valorar opción rainbow mediante Monte Carlo'''
                n_assets = len(S0_list)
                
                # Generar números aleatorios correlacionados
                L = np.linalg.cholesky(corr_matrix)
                Z = np.random.standard_normal((n_sims, n_assets))
                Z_corr = Z @ L.T
                
                # Generar precios finales
                ST = np.zeros((n_sims, n_assets))
                for i in range(n_assets):
                    ST[:, i] = S0_list[i] * np.exp(
                        (r - 0.5*sigma_list[i]**2)*T + sigma_list[i]*np.sqrt(T)*Z_corr[:, i]
                    )
                
                # Calcular payoffs
                if tipo_opcion == 'best_of_calls':
                    mejores_precios = np.max(ST, axis=1)
                    payoffs = np.maximum(mejores_precios - K, 0)
                elif tipo_opcion == 'worst_of_calls':
                    peores_precios = np.min(ST, axis=1)
                    payoffs = np.maximum(peores_precios - K, 0)
                
                # Valor de la opción
                precio_opcion = np.exp(-r*T) * np.mean(payoffs)
                return precio_opcion

            def estrategia_reversion_media(precios, lookback=20, entrada_std=2.0, salida_std=0.5):
                '''Generar señales de trading de reversión a la media'''
                señales = np.zeros(len(precios))
                
                for i in range(lookback, len(precios)):
                    ventana = precios[i-lookback:i]
                    media = np.mean(ventana)
                    std = np.std(ventana)
                    
                    if std > 0:
                        z_score = (precios[i] - media) / std
                        
                        if z_score > entrada_std:
                            señales[i] = -1  # Señal de venta
                        elif z_score < -entrada_std:
                            señales[i] = 1   # Señal de compra
                        elif abs(z_score) < salida_std:
                            señales[i] = 0   # Salir de posición
                
                return señales

            def var_portafolio_mc(pesos, retornos_esperados, matriz_cov, valor_portafolio, nivel_confianza=0.05, n_sims=100000):
                '''Calcular VaR del portafolio mediante Monte Carlo'''
                
                retornos_portafolio = np.random.multivariate_normal(retornos_esperados, matriz_cov, n_sims)
                pnl_portafolio = valor_portafolio * (retornos_portafolio @ pesos)
                
                var = -np.percentile(pnl_portafolio, nivel_confianza * 100)
                es = -np.mean(pnl_portafolio[pnl_portafolio <= -var])
                
                return var, es

            # Ejemplo de uso - Opción Rainbow
            S0 = [100, 110, 95]
            sigma = [0.2, 0.25, 0.3]
            corr = np.array([[1.0, 0.5, 0.3],
                            [0.5, 1.0, 0.4],
                            [0.3, 0.4, 1.0]])

            precio_rainbow = opcion_rainbow_mc(S0, 100, 1.0, 0.05, sigma, corr, 50000)
            print(f"Precio de opción Rainbow: ${precio_rainbow:.4f}")
            """

            st.sidebar.download_button(
                label="💻 Descargar Código Avanzado",
                data=codigo_avanzado,
                file_name="ingenieria_financiera_avanzada.py",
                mime="text/python"
            )
