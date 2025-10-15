"""Technical indicators for market analysis."""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

class TechnicalIndicators:
    """Class for calculating technical indicators."""
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def calculate_wma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average (WMA)."""
        weights = np.arange(1, period + 1)
        return prices.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = TechnicalIndicators.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    @staticmethod
    def calculate_stochastic_rsi(prices: pd.Series, period: int = 14, k_period: int = 3, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic RSI."""
        # First calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Then calculate Stochastic of RSI
        rsi_min = rsi.rolling(window=period).min()
        rsi_max = rsi.rolling(window=period).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        
        k_percent = stoch_rsi.rolling(window=k_period).mean()
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'stoch_rsi': stoch_rsi,
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index (CCI)."""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On Balance Volume (OBV)."""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def calculate_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
        senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted 26 periods back
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range (ATR)."""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)."""
        plus_dm = high.diff()
        minus_dm = low.diff(-1).abs()
        
        cond1 = (plus_dm > 0) & (plus_dm > minus_dm)
        plus_dm = plus_dm.where(cond1, 0)
        
        cond2 = (minus_dm > 0) & (minus_dm > plus_dm)
        minus_dm = minus_dm.where(cond2, 0)
        
        tr = pd.concat([high - low, 
                       (high - close.shift()).abs(), 
                       (low - close.shift()).abs()], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(period).mean().iloc[-1]
        return adx
    
    @staticmethod
    def analyze_trend(close: pd.Series, ema9: pd.Series, ema21: pd.Series, ema200: pd.Series) -> str:
        """Analyze market trend based on EMA crossovers."""
        current_close = close.iloc[-1]
        current_ema9 = ema9.iloc[-1]
        current_ema21 = ema21.iloc[-1]
        current_ema200 = ema200.iloc[-1]
        
        # Check if EMAs are aligned for uptrend
        if (current_ema9 > current_ema21 > current_ema200 and 
            current_close > current_ema9):
            return "STRONG_UPTREND"
        
        # Check if EMAs are aligned for downtrend
        if (current_ema9 < current_ema21 < current_ema200 and 
            current_close < current_ema9):
            return "STRONG_DOWNTREND"
        
        # Check for potential trend reversals
        if current_ema9 > current_ema21 and current_close > current_ema200:
            return "WEAK_UPTREND"
            
        if current_ema9 < current_ema21 and current_close < current_ema200:
            return "WEAK_DOWNTREND"
            
        return "SIDEWAYS"
    
    @staticmethod
    def generate_signal(close: pd.Series, high: pd.Series, low: pd.Series, 
                       volume: pd.Series, rsi_period: int = 14, 
                       adx_period: int = 14, atr_period: int = 14) -> Dict[str, Any]:
        """Generate trading signal based on multiple indicators with condition tracking."""
        # Calculate all indicators
        ema9 = TechnicalIndicators.calculate_ema(close, 9)
        ema21 = TechnicalIndicators.calculate_ema(close, 21)
        ema200 = TechnicalIndicators.calculate_ema(close, 200)
        rsi = TechnicalIndicators.calculate_rsi(close, rsi_period)
        atr = TechnicalIndicators.calculate_atr(high, low, close, atr_period)
        adx = TechnicalIndicators.calculate_adx(high, low, close, adx_period)
        
        # Analyze trend
        trend = TechnicalIndicators.analyze_trend(close, ema9, ema21, ema200)
        
        # Track individual conditions
        conditions = {
            'ema_trend': False,
            'rsi_level': False,
            'adx_strength': False,
            'price_position': False
        }
        
        condition_details = {
            'ema_trend': f"EMA روند: {trend}",
            'rsi_level': f"RSI: {rsi:.2f}",
            'adx_strength': f"ADX: {adx:.2f}",
            'price_position': "موقعیت قیمت نسبت به EMA200"
        }
        
        # Determine signal type
        signal_type = "NEUTRAL"
        
        # Check EMA trend condition
        if "UPTREND" in trend or "DOWNTREND" in trend:
            conditions['ema_trend'] = True
            
        # Check RSI conditions
        if "UPTREND" in trend and rsi < 70:
            signal_type = "LONG"
            conditions['rsi_level'] = True
        elif "DOWNTREND" in trend and rsi > 30:
            signal_type = "SHORT"
            conditions['rsi_level'] = True
            
        # Check ADX strength
        if adx > 25:
            conditions['adx_strength'] = True
            
        # Check price position relative to EMA200
        current_price = close.iloc[-1]
        ema200_current = ema200.iloc[-1]
        if (signal_type == "LONG" and current_price > ema200_current) or \
           (signal_type == "SHORT" and current_price < ema200_current):
            conditions['price_position'] = True
        
        # Count satisfied conditions
        satisfied_conditions = sum(conditions.values())
        
        # Determine confidence and risk based on conditions
        if satisfied_conditions >= 3:
            if satisfied_conditions == 4:
                confidence = "HIGH"
                risk_level = "SAFE"
            else:
                confidence = "MEDIUM"
                risk_level = "MEDIUM"
        elif satisfied_conditions >= 1:
            confidence = "LOW"
            risk_level = "RISKY"
        else:
            confidence = "VERY_LOW"
            risk_level = "VERY_RISKY"
        
        # Create failed conditions list
        failed_conditions = []
        for condition, status in conditions.items():
            if not status:
                failed_conditions.append(condition_details[condition])
        
        return {
            "signal": signal_type,
            "confidence": confidence,
            "risk_level": risk_level,
            "failed_conditions": failed_conditions,
            "indicators": {
                "ema9": ema9.iloc[-1],
                "ema21": ema21.iloc[-1],
                "ema200": ema200.iloc[-1],
                "rsi": rsi,
                "atr": atr,
                "adx": adx,
                "trend": trend,
                "current_price": close.iloc[-1]
            }
        }
    
    @staticmethod
    def calculate_position_levels(entry_price: float, stop_loss: float, 
                                 take_profit: float, risk_reward_ratio: float = 3.0) -> Dict[str, float]:
        """Calculate position levels with 5 take profit targets."""
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = risk_amount * risk_reward_ratio
        
        if entry_price > stop_loss:  # Long position
            tp_levels = [
                entry_price + (reward_amount * 0.2),
                entry_price + (reward_amount * 0.4),
                entry_price + (reward_amount * 0.6),
                entry_price + (reward_amount * 0.8),
                entry_price + reward_amount
            ]
        else:  # Short position
            tp_levels = [
                entry_price - (reward_amount * 0.2),
                entry_price - (reward_amount * 0.4),
                entry_price - (reward_amount * 0.6),
                entry_price - (reward_amount * 0.8),
                entry_price - reward_amount
            ]
        
        return {
            "entry": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "tp1": tp_levels[0],
            "tp2": tp_levels[1],
            "tp3": tp_levels[2],
            "tp4": tp_levels[3],
            "tp5": tp_levels[4],
            "risk_reward_ratio": risk_reward_ratio
        }
