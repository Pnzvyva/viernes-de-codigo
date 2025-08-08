import streamlit as st
import importlib
import sys
import os
from utils.progress_tracker import ProgressTracker

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize progress tracker
if 'progress_tracker' not in st.session_state:
    st.session_state.progress_tracker = ProgressTracker()

# Course configuration
CLASSES = {
    "Clase 1: Introducción a la Ingeniería Financiera": "classes.class_01_intro",
    "Clase 2: Valor del Dinero en el Tiempo": "classes.class_02_time_value",
    "Clase 3: Valoración de Bonos": "classes.class_03_bonds",
    "Clase 4: Valoración de Acciones": "classes.class_04_stocks",
    "Clase 5: Introducción a Derivados": "classes.class_05_derivatives",
    "Clase 6: Valoración de Opciones": "classes.class_06_options",
    "Clase 7: Teoría de Portafolios": "classes.class_07_portfolio",
    "Clase 8: Gestión de Riesgos": "classes.class_08_risk",
    "Clase 9: Métodos de Monte Carlo": "classes.class_09_monte_carlo",
    "Clase 10: Temas Avanzados": "classes.class_10_advanced"
}

def main():
    st.set_page_config(
        page_title="Viernes de Codigo",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title("📈 Viernes de codigo")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navegación del Curso")
        
        # Progress overview
        progress = st.session_state.progress_tracker.get_progress()
        st.metric("Progreso General", f"{progress:.1f}%")
        st.progress(progress / 100)
        
        st.markdown("---")
        
        # Class selection
        selected_class = st.selectbox(
            "Selecciona una Clase:",
            options=list(CLASSES.keys()),
            key="class_selector"
        )
        
        # Navigation buttons
        class_keys = list(CLASSES.keys())
        current_index = class_keys.index(selected_class)
        
        col1, col2 = st.columns(2)
        with col1:
            if current_index > 0:
                if st.button("← Anterior"):
                    st.session_state.class_selector = class_keys[current_index - 1]
                    st.rerun()
        
        with col2:
            if current_index < len(class_keys) - 1:
                if st.button("Siguiente →"):
                    st.session_state.class_selector = class_keys[current_index + 1]
                    st.rerun()
        
        st.markdown("---")
        
        # Course overview
        st.subheader("Resumen del Curso")
        completed_classes = st.session_state.progress_tracker.get_completed_classes()
        
        for i, class_name in enumerate(CLASSES.keys(), 1):
            status = "✅" if class_name in completed_classes else "⭕"
            st.write(f"{status} Clase {i}")
    
    # Main content area
    if selected_class:
        module_name = CLASSES[selected_class]
        try:
            # Import and run the selected class module
            module = importlib.import_module(module_name)
            if hasattr(module, 'run_class'):
                module.run_class()
            else:
                st.error(f"Module {module_name} does not have a 'run_class' function")
        except ImportError as e:
            st.error(f"No se pudo importar el módulo {module_name}: {e}")
        except Exception as e:
            st.error(f"Error ejecutando la clase: {e}")

if __name__ == "__main__":
    main()
