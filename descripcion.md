# Viernes de Codigo

## Overview

This is an interactive online learning platform for Financial Engineering, built with Streamlit. The platform provides a comprehensive 10-class course covering fundamental to advanced topics in financial engineering, including time value of money, bond and stock valuation, derivatives, options pricing, portfolio theory, risk management, Monte Carlo methods, and advanced derivatives. The application features interactive calculations, visualizations using Plotly, progress tracking, and hands-on examples to help students understand complex financial concepts through practical implementation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface and interactive components
- **Layout Pattern**: Multi-tab structure within each class for organized content delivery
- **Navigation**: Sidebar-based course navigation with progress tracking
- **Visualization**: Plotly Express and Plotly Graph Objects for financial charts and interactive plots
- **State Management**: Streamlit session state for maintaining user progress and class completion status

### Backend Architecture
- **Application Structure**: Modular class-based design with separate modules for each course class
- **Business Logic**: Centralized financial calculations in utility classes
- **Progress Management**: Session-based progress tracking with completion status and scoring
- **Mathematical Computing**: NumPy and SciPy for financial computations and statistical analysis

### Core Components
- **Class Modules**: Individual Python modules (class_01_intro.py through class_10_advanced.py) for each course lesson
- **Financial Functions**: Centralized FinancialCalculations class containing Black-Scholes pricing, bond valuation, portfolio optimization, and other financial formulas
- **Progress Tracker**: Session-state based system for tracking course completion and maintaining user progress across sessions
- **Interactive Tools**: Real-time calculators and visualizations for financial concepts like option pricing, portfolio optimization, and risk analysis

### Data Processing
- **Libraries**: Pandas for data manipulation and analysis, NumPy for numerical computations
- **Statistical Analysis**: SciPy for advanced mathematical functions including optimization and statistical distributions
- **Financial Modeling**: Custom implementations of Black-Scholes model, Monte Carlo simulations, and portfolio optimization algorithms

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework for building the interactive learning platform
- **pandas**: Data manipulation and analysis for financial datasets
- **numpy**: Numerical computing for mathematical calculations
- **scipy**: Scientific computing library for optimization and statistical functions
- **plotly**: Interactive visualization library for financial charts and graphs

### Mathematical Libraries
- **scipy.stats**: Statistical distributions (normal, t-distribution) for risk calculations and option pricing
- **scipy.optimize**: Optimization algorithms for portfolio optimization and parameter estimation
- **math**: Standard mathematical functions for financial calculations

### Data Visualization
- **plotly.express**: High-level interface for creating interactive plots and charts
- **plotly.graph_objects**: Low-level interface for custom financial visualizations and complex chart configurations

Note: The application is designed as a standalone educational platform with no external API integrations or database dependencies, making it self-contained and easily deployable.