# Viernes de Codigo

<!-- Font Awesome (para iconos) -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">

## <i class="fa-solid fa-graduation-cap"></i> Visión general

Plataforma interactiva de aprendizaje en línea de **Ingeniería Financiera**, **desarrollada en *Python* con Streamlit** <i class="fa-brands fa-python"></i> <i class="fa-solid fa-bolt"></i>. Ofrece un curso integral de **10 clases** que cubre desde fundamentos hasta temas avanzados: valor del dinero en el tiempo, valoración de bonos y acciones, derivados, **pricing** de opciones, teoría de portafolios, gestión de riesgo, métodos de Monte Carlo y derivados avanzados.  
La aplicación incluye **cálculos interactivos**, **visualizaciones con Plotly**, **seguimiento de progreso** y **ejemplos prácticos** para comprender conceptos financieros complejos mediante implementación práctica.

---

## <i class="fa-solid fa-user-check"></i> Preferencias del usuario

Estilo de comunicación preferido: **lenguaje simple y cotidiano**.

---

## <i class="fa-solid fa-diagram-project"></i> Arquitectura del sistema

### <i class="fa-solid fa-display"></i> Arquitectura de Frontend
- **Framework**: **Streamlit** para interfaz web y componentes interactivos.
- **Patrón de diseño**: estructura con **múltiples pestañas** por clase para organizar contenidos.
- **Navegación**: barra lateral con navegación del curso y **seguimiento de progreso**.
- **Visualización**: **Plotly Express** y **Plotly Graph Objects** para gráficos financieros e interactivos.
- **Gestión de estado**: **`st.session_state`** de Streamlit para mantener progreso y estado de finalización por clase.

### <i class="fa-solid fa-server"></i> Arquitectura de Backend
- **Estructura de la aplicación**: diseño modular por clases, con **módulos separados** para cada lección.
- **Lógica de negocio**: cálculos financieros centralizados en **clases utilitarias**.
- **Gestión de progreso**: seguimiento basado en sesión con **estado de finalización** y **puntuación**.
- **Cómputo matemático**: **NumPy** y **SciPy** para cálculos financieros y análisis estadístico.

---

## <i class="fa-solid fa-cubes"></i> Componentes principales
- **Módulos de clase**: módulos Python individuales (`class_01_intro.py` a `class_10_advanced.py`) por lección.
- **Funciones financieras**: clase central **`FinancialCalculations`** con Black–Scholes, valoración de bonos, optimización de portafolios y otras fórmulas.
- **Rastreador de progreso**: sistema basado en **estado de sesión** para mantener avance del curso a través de sesiones.
- **Herramientas interactivas**: calculadoras en tiempo real y visualizaciones para *option pricing*, optimización de portafolios y análisis de riesgo.

---

## <i class="fa-solid fa-database"></i> Procesamiento de datos
- **Librerías**: **Pandas** para manipulación y análisis de datos; **NumPy** para cómputo numérico.
- **Análisis estadístico**: **SciPy** para funciones avanzadas, incluidas optimización y distribuciones estadísticas.
- **Modelado financiero**: implementaciones propias de **Black–Scholes**, **simulaciones de Monte Carlo** y **algoritmos de optimización de portafolios**.

---

## <i class="fa-solid fa-boxes-stacked"></i> Dependencias externas

### <i class="fa-solid fa-code"></i> Librerías principales
- **streamlit**: framework web para construir la plataforma interactiva de aprendizaje.
- **pandas**: manipulación y análisis de *datasets* financieros.
- **numpy**: cómputo numérico para cálculos matemáticos.
- **scipy**: funciones científicas para optimización y estadística.
- **plotly**: visualizaciones interactivas para gráficos y paneles financieros.

### <i class="fa-solid fa-square-root-variable"></i> Librerías matemáticas
- **scipy.stats**: distribuciones estadísticas (normal, t) para riesgo y *option pricing*.
- **scipy.optimize**: algoritmos de optimización para portafolios y estimación de parámetros.
- **math**: funciones matemáticas estándar para cálculos financieros.

### <i class="fa-solid fa-chart-line"></i> Visualización de datos
- **plotly.express**: interfaz de alto nivel para gráficos interactivos.
- **plotly.graph_objects**: interfaz de bajo nivel para visualizaciones financieras personalizadas y configuraciones complejas.

---

> <i class="fa-solid fa-circle-info"></i> **Nota:** La aplicación está diseñada como una plataforma educativa **autónoma**, **sin integraciones externas de API** ni dependencias de base de datos. Esto la hace **auto-contenida** y **fácil de desplegar** (ej. `streamlit run app.py`), manteniendo la complejidad al mínimo mientras aprovecha el ecosistema **Python + Streamlit**.
