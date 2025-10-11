import numpy as np
import pandas as pd
from scipy.stats import norm
import math

class FinancialCalculations:
    
    @staticmethod
    def present_value(future_value, rate, periods):
        """Calculate present value"""
        return future_value / (1 + rate) ** periods
    
    @staticmethod
    def future_value(present_value, rate, periods):
        """Calculate future value"""
        return present_value * (1 + rate) ** periods
    
    @staticmethod
    def annuity_pv(payment, rate, periods):
        """Present value of annuity"""
        return payment * (1 - (1 + rate) ** -periods) / rate
    
    @staticmethod
    def bond_price(face_value, coupon_rate, market_rate, periods):
        """Calculate bond price"""
        coupon = face_value * coupon_rate
        pv_coupons = FinancialCalculations.annuity_pv(coupon, market_rate, periods)
        pv_face = FinancialCalculations.present_value(face_value, market_rate, periods)
        return pv_coupons + pv_face
    
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma):
        """Black-Scholes call option price"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    @staticmethod
    def black_scholes_put(S, K, T, r, sigma):
        """Black-Scholes put option price"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    @staticmethod
    def portfolio_return(weights, returns):
        """Calculate portfolio expected return"""
        return np.sum(weights * returns)
    
    @staticmethod
    def portfolio_volatility(weights, cov_matrix):
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate):
        """Calculate Sharpe ratio"""
        excess_return = np.mean(returns) - risk_free_rate
        volatility = np.std(returns)
        return excess_return / volatility if volatility != 0 else 0
    
    @staticmethod
    def var_historical(returns, confidence_level=0.05):
        """Calculate Value at Risk using historical method"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def monte_carlo_simulation(S0, mu, sigma, T, dt, n_simulations):
        """Monte Carlo simulation for stock price paths"""
        n_steps = int(T / dt)
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = S0
        
        for i in range(n_steps):
            z = np.random.standard_normal(n_simulations)
            paths[:, i + 1] = paths[:, i] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        
        return paths
    
    @staticmethod
    def lookback_option_mc(S0, T, r, sigma, n_sims, option_type='call', n_steps=252):
        np.random.seed(42)
        
        # Time step
        dt = T / n_steps
        # Generate random paths
        Z = np.random.standard_normal((n_sims, n_steps))
        S = np.zeros((n_sims, n_steps + 1))
        S[:, 0] = S0
        
        # Simulate paths
        for i in range(n_steps):
            S[:, i + 1] = S[:, i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, i])
        
        # Calculate the minimum and maximum prices during the life of the option
        min_prices = np.min(S[:, 1:], axis=1)  # Min prices excluding the initial price
        max_prices = np.max(S[:, 1:], axis=1)  # Max prices excluding the initial price
        
        # Calculate the payoff for the option
        if option_type == 'call':
            payoffs = np.maximum(S[:, -1] - min_prices, 0)  # Lookback call payoff
        else:
            payoffs = np.maximum(max_prices - S[:, -1], 0)  # Lookback put payoff
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
        
        # Devuelve tres valores: precio de la opción, error estándar, y payoffs
        return option_price, standard_error, payoffs