import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.progress_tracker import ProgressTracker


def run_class():
    st.header("Clase 1: Introducción a las finanzas cuantitativas")
    
    # Progress tracking
    progress_tracker = st.session_state.progress_tracker
    
    # Class content
    tab1, tab2, tab3, tab4 = st.tabs(["Resumen", "Conceptos Clave", "Ejemplo Interactivo", "Conclusión"])
    
    with tab1:
        st.subheader("¿Qué son las finanzas cuantitativas?")
        
        st.write("""
        Las finanzas cuantitativas son un campo que utiliza modelos matemáticos y estadísticos para analizar y predecir el comportamiento de los mercados financieros. 
        Se enfoca en:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Áreas Principales:**
            - Gestión de Riesgos
            - Valoración de Activos
            - Optimización de Portafolios
            - Valores Derivados
            - Análisis Cuantitativo
            """)
        
        with col2:
            st.write("""
            **Herramientas Utilizadas:**
            - Modelos Matemáticos
            - Análisis Estadístico
            - ALGO trading
            - Datos de Mercados Financieros
            - Métodos de Simulación
            """)
        
        st.subheader("Estructura del Curso")
        
        course_data = {
            'Clase': [f'Clase {i}' for i in range(1, 11)],
            'Tema': [
                'Introducción a las finanzas cuantitativas',
                'Valor del Dinero en el Tiempo',
                'Valoración de Bonos',
                'Valoración de Acciones',
                'Introducción a Derivados',
                'Valoración de Opciones',
                'Teoría de Portafolios',
                'Gestión de Riesgos',
                'Métodos de Monte Carlo',
                'Temas Avanzados'
            ],
            'Dificultad': ['Principiante', 'Principiante', 'Intermedio', 'Intermedio', 
                          'Intermedio', 'Avanzado', 'Avanzado', 'Avanzado', 
                          'Avanzado', 'Experto']
        }
        
        df = pd.DataFrame(course_data)
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.subheader("Conceptos Financieros Clave")
        
        concepts = {
            "Valor del Dinero en el Tiempo": "El dinero disponible hoy vale más que la misma cantidad en el futuro",
            "Riesgo y Retorno": "Los retornos potenciales más altos generalmente vienen con mayor riesgo",
            "Diversificación": "Distribuir inversiones para reducir el riesgo",
            "Arbitraje": "Ganancia libre de riesgo por diferencias de precios",
            "Mercados Eficientes": "Mercados donde los precios reflejan toda la información disponible",
            "Derivados": "Instrumentos financieros cuyo valor deriva de activos subyacentes"
        }
        
        for concept, definition in concepts.items():
            with st.expander(f"📚 {concept}"):
                st.write(definition)
                
                # Add simple examples for key concepts
                if concept == "Valor del Dinero en el Tiempo":
                    st.write("**Ejemplo:** 100USD hoy vs 100USD en un año")

                    col1, col2 = st.columns(2)
                    with col1:
                        rate = st.slider("Tasa de Interés (%)", 0, 20, 5, key="tvm_rate")
                    with col2:
                        years = st.slider("Años", 1, 10, 1, key="tvm_years")
                    
                    future_val = 100 * (1 + rate/100) ** years
                    st.write(f"100USD invertidos hoy al {rate}% por {years} año(s) = ${future_val:.2f}")
    
    with tab3:
        st.subheader("Ejemplo Interactivo: Visualización de Precios de Acciones")
        
        st.write("Simulemos un movimiento simple de precio de acción usando caminata aleatoria:")
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            initial_price = st.number_input("Precio Inicial de la Acción ($)", value=100.0, min_value=1.0)
        with col2:
            num_days = st.slider("Número de Días", 30, 365, 100)
        with col3:
            volatility = st.slider("Volatilidad Diaria (%)", 0.5, 5.0, 2.0)

        if st.button("Generar Trayectoria del Precio"):
            # Generate random stock price path
            np.random.seed(42)  # For reproducibility
            returns = np.random.normal(0, volatility/100, num_days)
            prices = [initial_price]
            
            for i in range(num_days):
                new_price = prices[-1] * (1 + returns[i])
                prices.append(new_price)
            
            # Create DataFrame
            df = pd.DataFrame({
                'Día': range(len(prices)),
                'Precio': prices
            })
            
            # Plot
            fig = px.line(df, x='Día', y='Precio', title='Trayectoria Simulada del Precio de la Acción')
            fig.update_layout(xaxis_title="Días", yaxis_title="Precio de la Acción ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precio Final", f"${prices[-1]:.2f}")
            with col2:
                st.metric("Retorno Total", f"{((prices[-1] - initial_price) / initial_price * 100):.2f}%")
            with col3:
                st.metric("Precio Máximo", f"${max(prices):.2f}")
            with col4:
                st.metric("Precio Mínimo", f"${min(prices):.2f}")
    
    with tab4:
        st.subheader("Resumen de la Clase")
        
        st.write("""
        En esta clase introductoria, cubrimos:
        
        1. **Definición de Ingeniería Financiera**: La aplicación de métodos matemáticos y de ingeniería a problemas financieros
        2. **Áreas Principales**: Gestión de riesgos, valoración de activos, optimización de portafolios y derivados
        3. **Conceptos Clave**: Valor del dinero en el tiempo, relación riesgo-retorno, diversificación
        4. **Herramientas**: Modelos matemáticos, análisis estadístico y programación
        5. **Estructura del Curso**: 10 clases progresando de temas básicos a avanzados
        
        **Puntos Clave:**
        - La ingeniería financiera combina finanzas, matemáticas y programación
        - Entender el riesgo y retorno es fundamental para todas las decisiones financieras
        - Los modelos matemáticos nos ayudan a cuantificar y gestionar la incertidumbre financiera
        - Las habilidades de programación son esenciales para implementar modelos financieros
        """)
        
        # Quiz section
        st.subheader("Evaluación Rápida de Conocimientos")
        
        quiz_col1, quiz_col2 = st.columns(2)
        
        with quiz_col1:
            q1 = st.radio(
                "¿Cuál es el enfoque principal de las finanzas cuantitativas?",
                ["Solo comercio de acciones", "Aplicar métodos matemáticos a las finanzas", "Operaciones bancarias", "Prácticas contables"],
                key="q1"
            )
        
        with quiz_col2:
            q2 = st.radio(
                "¿Cuál afirmación sobre riesgo y retorno es correcta?",
                ["Mayor riesgo siempre significa mayor retorno", "Mayor retorno potencial usualmente viene con mayor riesgo", 
                 "Riesgo y retorno no están relacionados", "Menor riesgo significa mayor retorno"],
                key="q2"
            )
        
        if st.button("Verificar Respuestas"):
            score = 0
            if q1 == "Aplicar métodos matemáticos a las finanzas":
                st.success("Pregunta 1: ¡Correcto! ✅")
                score += 1
            else:
                st.error("Pregunta 1: Incorrecto. La ingeniería financiera aplica métodos matemáticos a las finanzas.")
            
            if q2 == "Mayor retorno potencial usualmente viene con mayor riesgo":
                st.success("Pregunta 2: ¡Correcto! ✅")
                score += 1
            else:
                st.error("Pregunta 2: Incorrecto. Mayor retorno potencial usualmente viene con mayor riesgo.")
            
            st.write(f"Tu puntuación: {score}/2")
            
            if score == 2:
                st.balloons()
                progress_tracker.mark_class_completed("Clase 1: Introducción a la Ingeniería Financiera")
                progress_tracker.set_class_score("Clase 1: Introducción a la Ingeniería Financiera", 100)
                st.success("🎉 Congratulations! You've completed Class 1!")
    
    # Download materials
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 Course Materials")
    
    # Create downloadable content
    notes = """
# Class 1: Introduction to Financial Engineering - Notes

## Key Concepts
1. Financial Engineering Definition
2. Core Areas of Study
3. Mathematical Tools
4. Risk and Return Relationship

## Important Formulas
- Future Value: FV = PV × (1 + r)^n
- Present Value: PV = FV / (1 + r)^n

## Python Libraries for Finance
- NumPy: Numerical computations
- Pandas: Data manipulation
- Matplotlib/Plotly: Visualizations
- SciPy: Scientific computing
"""
    
    st.sidebar.download_button(
        label="📄 Download Class Notes",
        data=notes,
        file_name="class_01_notes.md",
        mime="text/markdown"
    )
