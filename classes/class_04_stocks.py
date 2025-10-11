import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.financial_functions import FinancialCalculations
from utils.progress_tracker import ProgressTracker

def run_class():
    st.header("Clase 4: Valoración de Acciones")

    progress_tracker = st.session_state.progress_tracker

    tab1, tab2, tab3, tab4 = st.tabs([
        "Modelos de Valoración", 
        "Modelos de Dividendos", 
        "Análisis DCF", 
        "Valoración Comparativa"
    ])

    # -------------------------------------------------------------------------
    # TAB 1 – Fundamentos de Valoración
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Fundamentos de la Valoración de Acciones")

        st.write("""
        La valoración de acciones es el proceso de determinar el valor intrínseco de los títulos de una empresa. 
        A diferencia de los bonos, que tienen flujos de efectivo fijos, las acciones representan propiedad en la empresa 
        y sus retornos dependen del desempeño y las expectativas del mercado.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.write("""
            **Enfoques Clave de Valoración:**
            - **Valor Intrínseco:** Basado en los fundamentos financieros.
            - **Valor Relativo:** Comparado con empresas similares.
            - **Valor Absoluto:** Basado en modelos DCF y de dividendos.
            - **Valor de Mercado:** Precio actual en el mercado.
            """)

        with col2:
            st.write("""
            **Factores que Afectan el Precio de una Acción:**
            - Ganancias y crecimiento de la empresa.
            - Condiciones de la industria.
            - Situación económica general.
            - Sentimiento del mercado.
            - Tasas de interés.
            """)

        st.subheader("Resumen de Métodos de Valoración")

        metodos = {
            "Modelo de Descuento de Dividendos (DDM)": {
                "Descripción": "Valora la acción con base en el valor presente de los dividendos esperados.",
                "Adecuado Para": "Empresas maduras que pagan dividendos.",
                "Fórmula": "P = D₁ / (r - g) para crecimiento constante."
            },
            "Flujo de Caja Descontado (DCF)": {
                "Descripción": "Valora la acción según los flujos de caja libres descontados al presente.",
                "Adecuado Para": "Empresas con flujos de caja predecibles.",
                "Fórmula": "Suma de los flujos de caja futuros descontados."
            },
            "Múltiplos de Mercado": {
                "Descripción": "Compara la acción con métricas financieras como P/U o P/VL.",
                "Adecuado Para": "Comparaciones rápidas entre compañías.",
                "Fórmula": "Precio = Múltiplo × Métrica."
            }
        }

        for metodo, detalles in metodos.items():
            with st.expander(f"📊 {metodo}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Descripción:** {detalles['Descripción']}")
                    st.write(f"**Adecuado Para:** {detalles['Adecuado Para']}")
                with col2:
                    st.write(f"**Fórmula:** {detalles['Fórmula']}")

    # -------------------------------------------------------------------------
    # TAB 2 – Modelos de Dividendos
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Modelos de Descuento de Dividendos (DDM)")

        tipo_modelo = st.selectbox(
            "Selecciona el modelo de dividendos:",
            ["Modelo de Crecimiento Constante (Gordon)", "Crecimiento en Dos Etapas", "Crecimiento Cero"]
        )

        # --- Modelo de Crecimiento Constante ---
        if tipo_modelo == "Modelo de Crecimiento Constante (Gordon)":
            st.write("**Modelo de Gordon (Crecimiento Constante)**")
            st.latex(r"P_0 = \frac{D_1}{r - g}")
            st.write("Donde: P₀ = Precio actual, D₁ = Dividendo del próximo año, r = tasa requerida, g = tasa de crecimiento.")

            col1, col2, col3 = st.columns(3)
            with col1:
                dividendo_actual = st.number_input("Dividendo actual (D₀) ($)", value=2.00, min_value=0.0)
                tasa_crecimiento = st.number_input("Tasa de crecimiento (g) (%)", value=5.0, min_value=0.0, max_value=50.0)
            with col2:
                tasa_requerida = st.number_input("Tasa requerida de retorno (r) (%)", value=10.0, min_value=0.1, max_value=50.0)
            with col3:
                st.write("")

            if tasa_requerida / 100 > tasa_crecimiento / 100:
                dividendo_siguiente = dividendo_actual * (1 + tasa_crecimiento / 100)
                valor_intrinseco = dividendo_siguiente / (tasa_requerida / 100 - tasa_crecimiento / 100)

                st.success(f"Valor Intrínseco: ${valor_intrinseco:.2f}")
                st.info(f"Dividendo del próximo año (D₁): ${dividendo_siguiente:.2f}")

                if st.button("Mostrar análisis de sensibilidad"):
                    rango_g = np.linspace(max(0, tasa_crecimiento - 3), min(tasa_requerida - 0.1, tasa_crecimiento + 3), 20)
                    valores = []

                    for g in rango_g:
                        if g < tasa_requerida:
                            d1 = dividendo_actual * (1 + g / 100)
                            valor = d1 / (tasa_requerida / 100 - g / 100)
                            valores.append(valor)
                        else:
                            valores.append(np.nan)

                    df_sensibilidad = pd.DataFrame({
                        'Tasa de Crecimiento (%)': rango_g,
                        'Valor Intrínseco ($)': valores
                    })

                    fig = px.line(df_sensibilidad, x='Tasa de Crecimiento (%)', y='Valor Intrínseco ($)',
                                  title='Sensibilidad del Valor ante cambios en la Tasa de Crecimiento')
                    fig.add_vline(x=tasa_crecimiento, line_dash="dash", annotation_text="Caso Base")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("¡La tasa de crecimiento debe ser menor que la tasa requerida!")

        # --- Modelo de Dos Etapas ---
        elif tipo_modelo == "Crecimiento en Dos Etapas":
            st.write("**Modelo de Crecimiento en Dos Etapas**")
            st.write("Permite distintas tasas de crecimiento en dos periodos.")

            col1, col2 = st.columns(2)
            with col1:
                dividendo_actual = st.number_input("Dividendo actual ($)", value=1.50, min_value=0.0, key="2stage_d0")
                tasa_alta = st.number_input("Tasa de alto crecimiento (%)", value=15.0, min_value=0.0, key="2stage_g1")
                años_alto_crecimiento = st.number_input("Periodo de alto crecimiento (años)", value=5, min_value=1, max_value=20, key="2stage_years")
            with col2:
                tasa_estable = st.number_input("Tasa de crecimiento estable (%)", value=3.0, min_value=0.0, key="2stage_g2")
                tasa_requerida = st.number_input("Tasa requerida (%)", value=12.0, min_value=0.1, key="2stage_r")

            if tasa_requerida / 100 > tasa_estable / 100:
                valor_etapa1 = 0
                dividendos_etapa1 = []

                for año in range(1, años_alto_crecimiento + 1):
                    div = dividendo_actual * (1 + tasa_alta / 100) ** año
                    vp = div / (1 + tasa_requerida / 100) ** año
                    valor_etapa1 += vp
                    dividendos_etapa1.append({'Año': año, 'Dividendo': div, 'VP': vp})

                dividendo_terminal = dividendo_actual * (1 + tasa_alta / 100) ** años_alto_crecimiento * (1 + tasa_estable / 100)
                valor_terminal = dividendo_terminal / (tasa_requerida / 100 - tasa_estable / 100)
                vp_terminal = valor_terminal / (1 + tasa_requerida / 100) ** años_alto_crecimiento

                valor_total = valor_etapa1 + vp_terminal

                st.success(f"Valor Intrínseco Total: ${valor_total:.2f}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Valor Etapa 1", f"${valor_etapa1:.2f}")
                    st.metric("Valor Terminal", f"${valor_terminal:.2f}")
                with col2:
                    st.metric("VP del Valor Terminal", f"${vp_terminal:.2f}")
                    st.metric("Porcentaje del Valor Terminal", f"{(vp_terminal / valor_total) * 100:.1f}%")

                df_div = pd.DataFrame(dividendos_etapa1)
                st.subheader("Proyección de Dividendos")
                st.dataframe(df_div, use_container_width=True)
            else:
                st.error("¡La tasa requerida debe ser mayor que la tasa estable!")

        # --- Modelo de Crecimiento Cero ---
        elif tipo_modelo == "Crecimiento Cero":
            st.write("**Modelo de Dividendos sin Crecimiento (Crecimiento Cero)**")
            st.latex(r"P_0 = \frac{D}{r}")
            st.write("Donde: P₀ = Precio actual, D = Dividendo anual constante, r = tasa requerida.")

            col1, col2 = st.columns(2)
            with col1:
                dividendo = st.number_input("Dividendo anual (D) ($)", value=3.00, min_value=0.0)
            with col2:
                tasa_requerida = st.number_input("Tasa requerida (r) (%)", value=8.0, min_value=0.1, max_value=50.0)

            if tasa_requerida > 0:
                valor_intrinseco = dividendo / (tasa_requerida / 100)
                st.success(f"Valor Intrínseco (P₀): ${valor_intrinseco:.2f}")

                if st.button("Mostrar sensibilidad ante la tasa requerida"):
                    r_range = np.linspace(max(0.1, tasa_requerida - 4), tasa_requerida + 4, 25)
                    valores = [dividendo / (r / 100) for r in r_range]

                    df_sens = pd.DataFrame({
                        'Tasa Requerida (%)': r_range,
                        'Valor Intrínseco ($)': valores
                    })

                    fig = px.line(df_sens, x='Tasa Requerida (%)', y='Valor Intrínseco ($)',
                                  title='Sensibilidad del Modelo de Crecimiento Cero',
                                  markers=True)
                    fig.add_vline(x=tasa_requerida, line_dash="dash", annotation_text="Caso Base")
                    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3 – Análisis DCF
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Análisis de Flujo de Caja Descontado (DCF)")

        st.write("""
        El método DCF determina el valor intrínseco descontando los flujos de caja futuros al valor presente.
        Este enfoque es esencial para valorar empresas según su capacidad de generar efectivo.
        """)

        st.write("**Modelo de Flujo de Caja Libre para el Accionista (FCFE)**")

        col1, col2 = st.columns(2)
        with col1:
            fcfe_actual = st.number_input("FCFE actual ($ millones)", value=100.0, min_value=0.0)
            años_crecimiento = st.number_input("Periodo de alto crecimiento (años)", value=5, min_value=1, max_value=15)
            crecimiento_alto = st.number_input("Tasa de crecimiento alto (%)", value=12.0, min_value=0.0)
        with col2:
            crecimiento_terminal = st.number_input("Tasa de crecimiento terminal (%)", value=3.0, min_value=0.0, max_value=10.0)
            tasa_descuento = st.number_input("Tasa de descuento (%)", value=10.0, min_value=1.0)
            acciones = st.number_input("Acciones en circulación (millones)", value=50.0, min_value=0.1)

        if tasa_descuento / 100 > crecimiento_terminal / 100:
            if st.button("Calcular Valor DCF"):
                pv_crecimiento = 0
                flujos = []

                for año in range(1, años_crecimiento + 1):
                    cf = fcfe_actual * (1 + crecimiento_alto / 100) ** año
                    vp_cf = cf / (1 + tasa_descuento / 100) ** año
                    pv_crecimiento += vp_cf
                    flujos.append({'Año': año, 'FCFE': cf, 'Valor Presente': vp_cf})

                cf_terminal = fcfe_actual * (1 + crecimiento_alto / 100) ** años_crecimiento * (1 + crecimiento_terminal / 100)
                valor_terminal = cf_terminal / (tasa_descuento / 100 - crecimiento_terminal / 100)
                vp_terminal = valor_terminal / (1 + tasa_descuento / 100) ** años_crecimiento

                valor_total = pv_crecimiento + vp_terminal
                valor_por_accion = valor_total / acciones

                st.subheader("Resultados de Valoración DCF")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Valor Total del Patrimonio", f"${valor_total:.1f}M")
                    st.metric("Valor del Periodo de Crecimiento", f"${pv_crecimiento:.1f}M")
                with col2:
                    st.metric("Valor Terminal", f"${valor_terminal:.1f}M")
                    st.metric("VP del Valor Terminal", f"${vp_terminal:.1f}M")
                with col3:
                    st.metric("Valor por Acción", f"${valor_por_accion:.2f}")
                    st.metric("Porcentaje del Valor Terminal", f"{(vp_terminal / valor_total) * 100:.1f}%")

                df_cf = pd.DataFrame(flujos)
                st.subheader("Proyección de Flujos de Caja")
                st.dataframe(df_cf, use_container_width=True)

                años = list(range(1, años_crecimiento + 1))
                fcfe_values = [fcfe_actual * (1 + crecimiento_alto / 100) ** año for año in años]

                fig = px.bar(x=años, y=fcfe_values, title="Crecimiento Proyectado del FCFE")
                fig.update_layout(xaxis_title="Año", yaxis_title="FCFE ($ millones)")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("¡La tasa de descuento debe ser mayor que la tasa de crecimiento terminal!")
    
    with tab4:
        st.subheader("Valoración Comparativa (Múltiplos)")
        
        st.write("""
        La valoración relativa utiliza múltiplos de mercado para estimar el valor de una acción 
        con base en compañías comparables. 
        Los múltiplos más comunes incluyen P/U, P/VL, P/Ventas y EV/EBITDA.
        """)
        
        tipo_multiple = st.selectbox(
            "Selecciona el múltiplo:", 
            ["Precio/Utilidad (P/U)", "Precio/Valor en Libros (P/VL)", "Precio/Ventas (P/V)"]
        )
        
        # --- Múltiplo P/U ---
        if tipo_multiple == "Precio/Utilidad (P/U)":
            st.write("**Valoración mediante el múltiplo P/U**")
            st.latex(r"Precio\ Objetivo = EPS \times P/U")
            
            col1, col2 = st.columns(2)
            with col1:
                eps_actual = st.number_input("EPS actual ($)", value=5.00, min_value=0.01)
                crecimiento_eps = st.number_input("Crecimiento del EPS (%)", value=8.0, min_value=-50.0, max_value=100.0)
                eps_proyectado = eps_actual * (1 + crecimiento_eps / 100)
                st.info(f"EPS proyectado: ${eps_proyectado:.2f}")
            
            with col2:
                pe_industria = st.number_input("P/U promedio de la industria", value=15.0, min_value=1.0, max_value=100.0)
                pe_empresa = st.number_input("P/U histórico de la empresa", value=18.0, min_value=1.0, max_value=100.0)
            
            if st.button("Calcular valoración P/U"):
                valor_industria = eps_proyectado * pe_industria
                valor_historico = eps_proyectado * pe_empresa
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Valor usando P/U de la industria", f"${valor_industria:.2f}")
                with col2:
                    st.metric("Valor usando P/U histórico", f"${valor_historico:.2f}")
                
                # Análisis de sensibilidad
                rango_pe = np.linspace(max(1, pe_industria - 5), pe_industria + 10, 30)
                valores = [eps_proyectado * pe for pe in rango_pe]
                
                df_pe = pd.DataFrame({
                    'Múltiplo P/U': rango_pe,
                    'Valor de la Acción ($)': valores
                })
                
                fig = px.line(df_pe, x='Múltiplo P/U', y='Valor de la Acción ($)', 
                            title='Valor de la Acción vs Múltiplo P/U')
                fig.add_vline(x=pe_industria, line_dash="dash", annotation_text="Promedio Industria")
                fig.add_vline(x=pe_empresa, line_dash="dot", annotation_text="Histórico")
                st.plotly_chart(fig, use_container_width=True)
        
        # --- Múltiplo P/VL ---
        elif tipo_multiple == "Precio/Valor en Libros (P/VL)":
            st.write("**Valoración mediante el múltiplo P/VL**")
            
            col1, col2 = st.columns(2)
            with col1:
                valor_libros = st.number_input("Valor en libros por acción ($)", value=20.00, min_value=0.01)
            with col2:
                pb_industria = st.number_input("P/VL promedio de la industria", value=2.5, min_value=0.1, max_value=20.0)
            
            valor_estimado = valor_libros * pb_industria
            st.success(f"Valor estimado usando P/VL: ${valor_estimado:.2f}")
        
        # --- Análisis comparativo ---
        st.subheader("Análisis Comparativo de Múltiplos")
        
        if st.button("Generar tabla comparativa"):
            datos_comparativos = {
                'Empresa': ['Compañía Objetivo', 'Competidor A', 'Competidor B', 'Competidor C', 'Promedio Industria'],
                'P/U': [0, 14.2, 16.8, 13.5, 15.0],
                'P/VL': [0, 2.1, 2.8, 1.9, 2.3],
                'P/Ventas': [0, 1.8, 2.2, 1.5, 1.9],
                'ROE (%)': [12.5, 11.2, 15.1, 9.8, 12.1],
                'Crecimiento (%)': [8.0, 7.2, 9.5, 5.8, 7.5]
            }
            
            df_comparativo = pd.DataFrame(datos_comparativos)
            st.dataframe(df_comparativo, use_container_width=True)
            
            st.write("**Resumen de valoración usando promedios de la industria:**")
            if 'eps_proyectado' in locals():
                valor_pe = eps_proyectado * 15.0
                st.write(f"Valor según P/U promedio: ${valor_pe:.2f}")
        
        # --- Ejercicio práctico ---
        st.subheader("Ejercicio de Valoración")
        
        st.write("""
        **Ejercicio: Valora la siguiente empresa**
        
        Datos de la empresa:
        - EPS actual: $4.00  
        - Crecimiento esperado del EPS: 10%  
        - Valor en libros por acción: $25.00  
        - P/U de la industria: 16.0x  
        - P/VL de la industria: 2.2x  
        """)
        
        val_pe_usuario = st.number_input("Tu valoración P/U ($):", min_value=0.0, format="%.2f", key="ex_pe")
        val_pb_usuario = st.number_input("Tu valoración P/VL ($):", min_value=0.0, format="%.2f", key="ex_pb")
        
        if st.button("Verificar ejercicio"):
            eps_correcto = 4.00 * 1.10
            val_pe_correcto = eps_correcto * 16.0
            val_pb_correcto = 25.00 * 2.2
            
            pe_ok = abs(val_pe_usuario - val_pe_correcto) < 2.0
            pb_ok = abs(val_pb_usuario - val_pb_correcto) < 2.0
            
            if pe_ok:
                st.success(f"✅ Valoración P/U correcta: ${val_pe_correcto:.2f}")
            else:
                st.error(f"❌ Valoración P/U incorrecta. Respuesta correcta: ${val_pe_correcto:.2f}")
            
            if pb_ok:
                st.success(f"✅ Valoración P/VL correcta: ${val_pb_correcto:.2f}")
            else:
                st.error(f"❌ Valoración P/VL incorrecta. Respuesta correcta: ${val_pb_correcto:.2f}")
            
            if pe_ok and pb_ok:
                progress_tracker.mark_class_completed("Clase 4: Valoración de Acciones")
                progress_tracker.set_class_score("Clase 4: Valoración de Acciones", 100)
                st.balloons()
                st.success("🎉 ¡Excelente! ¡Has dominado la valoración de acciones!")
        
        # --- Recursos descargables ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("📚 Recursos de Valoración")
        
        valuation_code = """
    # Modelos de Valoración de Acciones

    def modelo_gordon(dividendo, tasa_crecimiento, tasa_requerida):
        '''Modelo de Gordon'''
        if tasa_requerida <= tasa_crecimiento:
            return None  # Datos inválidos
        
        dividendo_siguiente = dividendo * (1 + tasa_crecimiento)
        return dividendo_siguiente / (tasa_requerida - tasa_crecimiento)

    def crecimiento_dos_etapas(dividendo_actual, crecimiento_alto, años, crecimiento_estable, tasa_requerida):
        '''Modelo de Crecimiento en Dos Etapas'''
        
        valor_etapa1 = 0
        for año in range(1, años + 1):
            dividendo = dividendo_actual * (1 + crecimiento_alto) ** año
            vp = dividendo / (1 + tasa_requerida) ** año
            valor_etapa1 += vp
        
        dividendo_terminal = dividendo_actual * (1 + crecimiento_alto) ** años * (1 + crecimiento_estable)
        valor_terminal = dividendo_terminal / (tasa_requerida - crecimiento_estable)
        vp_terminal = valor_terminal / (1 + tasa_requerida) ** años
        
        return valor_etapa1 + vp_terminal

    # Ejemplo de uso
    valor = modelo_gordon(2.0, 0.05, 0.10)  # $42.00
    valor_dos_etapas = crecimiento_dos_etapas(1.5, 0.15, 5, 0.03, 0.12)
    """
        
        st.sidebar.download_button(
            label="💻 Descargar código de valoración",
            data=valuation_code,
            file_name="valoracion_acciones.py",
            mime="text/python"
        )