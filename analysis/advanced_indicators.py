"""
Advanced Technical Indicators for Enhanced Analysis
اندیکاتورهای تکنیکال پیشرفته برای تحلیل‌های قوی‌تر
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from .technical_indicators import TechnicalIndicators

class AdvancedIndicators:
    """کلاس اندیکاتورهای پیشرفته"""
    
    @staticmethod
    def calculate_multi_timeframe_ema(prices: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
        """محاسبه EMA در چندین تایم‌فریم"""
        emas = {}
        for period in periods:
            emas[f'ema_{period}'] = TechnicalIndicators.calculate_ema(prices, period)
        return emas
    
    @staticmethod
    def calculate_trend_strength(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series) -> Dict[str, float]:
        """محاسبه قدرت روند با ترکیب چندین اندیکاتور"""
        
        # EMA Alignment Score
        ema9 = TechnicalIndicators.calculate_ema(close, 9)
        ema21 = TechnicalIndicators.calculate_ema(close, 21)
        ema50 = TechnicalIndicators.calculate_ema(close, 50)
        ema200 = TechnicalIndicators.calculate_ema(close, 200)
        
        current_price = close.iloc[-1]
        current_ema9 = ema9.iloc[-1]
        current_ema21 = ema21.iloc[-1]
        current_ema50 = ema50.iloc[-1]
        current_ema200 = ema200.iloc[-1]
        
        # EMA Alignment (0-25 points)
        ema_score = 0
        if current_ema9 > current_ema21 > current_ema50 > current_ema200:
            ema_score = 25  # Perfect bullish alignment
        elif current_ema9 > current_ema21 > current_ema50:
            ema_score = 20
        elif current_ema9 > current_ema21:
            ema_score = 15
        elif current_price > current_ema200:
            ema_score = 10
        elif current_ema9 < current_ema21 < current_ema50 < current_ema200:
            ema_score = 25  # Perfect bearish alignment (also strong)
        elif current_ema9 < current_ema21 < current_ema50:
            ema_score = 20
        elif current_ema9 < current_ema21:
            ema_score = 15
        else:
            ema_score = 5
        
        # MACD Strength (0-20 points)
        macd_data = TechnicalIndicators.calculate_macd(close)
        macd = macd_data['macd'].iloc[-1]
        signal = macd_data['signal'].iloc[-1]
        histogram = macd_data['histogram'].iloc[-1]
        
        macd_score = 0
        if abs(histogram) > 0.001:  # Strong MACD signal
            if (macd > signal and macd > 0) or (macd < signal and macd < 0):
                macd_score = 20
            else:
                macd_score = 15
        elif abs(macd - signal) < 0.0005:  # Near crossover
            macd_score = 10
        else:
            macd_score = 5
        
        # Volume Confirmation (0-15 points)
        obv = TechnicalIndicators.calculate_obv(close, volume)
        obv_trend = (obv.iloc[-1] - obv.iloc[-10]) / obv.iloc[-10] * 100 if obv.iloc[-10] != 0 else 0
        
        volume_score = 0
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        if volume_ratio > 1.5 and abs(obv_trend) > 2:
            volume_score = 15
        elif volume_ratio > 1.2:
            volume_score = 10
        else:
            volume_score = 5
        
        # RSI Momentum (0-15 points)
        rsi = TechnicalIndicators.calculate_rsi(close)
        rsi_score = 0
        
        if 40 <= rsi <= 60:
            rsi_score = 15  # Neutral zone - good for trend continuation
        elif 30 <= rsi <= 70:
            rsi_score = 12
        elif rsi < 30 or rsi > 70:
            rsi_score = 10  # Extreme levels
        else:
            rsi_score = 5
        
        # ADX Trend Strength (0-15 points)
        adx = TechnicalIndicators.calculate_adx(high, low, close)
        adx_score = 0
        
        if adx > 50:
            adx_score = 15
        elif adx > 25:
            adx_score = 12
        elif adx > 20:
            adx_score = 8
        else:
            adx_score = 3
        
        # Bollinger Bands Position (0-10 points)
        bb = TechnicalIndicators.calculate_bollinger_bands(close)
        bb_upper = bb['upper'].iloc[-1]
        bb_lower = bb['lower'].iloc[-1]
        bb_middle = bb['middle'].iloc[-1]
        
        bb_score = 0
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        if 0.2 <= bb_position <= 0.8:
            bb_score = 10  # Good position for trend continuation
        elif bb_position < 0.1 or bb_position > 0.9:
            bb_score = 8   # Extreme positions - potential reversal
        else:
            bb_score = 6
        
        total_score = ema_score + macd_score + volume_score + rsi_score + adx_score + bb_score
        
        return {
            'total_score': total_score,
            'max_score': 100,
            'percentage': (total_score / 100) * 100,
            'ema_score': ema_score,
            'macd_score': macd_score,
            'volume_score': volume_score,
            'rsi_score': rsi_score,
            'adx_score': adx_score,
            'bb_score': bb_score,
            'strength_level': AdvancedIndicators._get_strength_level(total_score)
        }
    
    @staticmethod
    def _get_strength_level(score: float) -> str:
        """تعیین سطح قدرت بر اساس امتیاز"""
        if score >= 85:
            return "بسیار قوی 🔥"
        elif score >= 70:
            return "قوی 💪"
        elif score >= 55:
            return "متوسط 📊"
        elif score >= 40:
            return "ضعیف 📉"
        else:
            return "بسیار ضعیف ⚠️"
    
    @staticmethod
    def calculate_support_resistance(high: pd.Series, low: pd.Series, close: pd.Series, 
                                   lookback: int = 20) -> Dict[str, List[float]]:
        """محاسبه سطوح حمایت و مقاومت"""
        
        # Find local highs and lows
        highs = []
        lows = []
        
        for i in range(lookback, len(high) - lookback):
            # Local high
            if high.iloc[i] == high.iloc[i-lookback:i+lookback+1].max():
                highs.append(high.iloc[i])
            
            # Local low
            if low.iloc[i] == low.iloc[i-lookback:i+lookback+1].min():
                lows.append(low.iloc[i])
        
        # Remove duplicates and sort
        resistance_levels = sorted(list(set(highs)), reverse=True)[:5]  # Top 5 resistance
        support_levels = sorted(list(set(lows)))[-5:]  # Top 5 support
        
        current_price = close.iloc[-1]
        
        # Filter relevant levels (within 10% of current price)
        relevant_resistance = [r for r in resistance_levels if r > current_price and r <= current_price * 1.1]
        relevant_support = [s for s in support_levels if s < current_price and s >= current_price * 0.9]
        
        return {
            'resistance': relevant_resistance[:3],  # Top 3 resistance
            'support': relevant_support[-3:],       # Top 3 support
            'current_price': current_price
        }
    
    @staticmethod
    def calculate_fibonacci_levels(high: pd.Series, low: pd.Series, lookback: int = 50) -> Dict[str, float]:
        """محاسبه سطوح فیبوناچی"""
        
        # Find swing high and low in lookback period
        recent_high = high.iloc[-lookback:].max()
        recent_low = low.iloc[-lookback:].min()
        
        diff = recent_high - recent_low
        
        # Fibonacci retracement levels
        fib_levels = {
            'high': recent_high,
            'low': recent_low,
            'fib_23.6': recent_high - (diff * 0.236),
            'fib_38.2': recent_high - (diff * 0.382),
            'fib_50.0': recent_high - (diff * 0.5),
            'fib_61.8': recent_high - (diff * 0.618),
            'fib_78.6': recent_high - (diff * 0.786)
        }
        
        return fib_levels
    
    def detect_divergence(self, prices: pd.Series, indicator: pd.Series, lookback: int = 20) -> Dict[str, Any]:
        """Enhanced divergence detection between price and indicator"""
        try:
            if len(prices) < lookback or len(indicator) < lookback:
                return {'divergence': 'none', 'strength': 0, 'type': None}
            
            # Find peaks and valleys with improved algorithm
            price_peaks = self._find_enhanced_peaks_valleys(prices.tail(lookback), 'peaks')
            price_valleys = self._find_enhanced_peaks_valleys(prices.tail(lookback), 'valleys')
            
            indicator_peaks = self._find_enhanced_peaks_valleys(indicator.tail(lookback), 'peaks')
            indicator_valleys = self._find_enhanced_peaks_valleys(indicator.tail(lookback), 'valleys')
            
            # Check for multiple divergence types
            regular_bullish = self._check_regular_bullish_divergence(price_valleys, indicator_valleys)
            hidden_bullish = self._check_hidden_bullish_divergence(price_valleys, indicator_valleys)
            regular_bearish = self._check_regular_bearish_divergence(price_peaks, indicator_peaks)
            hidden_bearish = self._check_hidden_bearish_divergence(price_peaks, indicator_peaks)
            
            # Prioritize regular divergences over hidden ones
            if regular_bullish['detected']:
                return {
                    'divergence': 'bullish',
                    'strength': regular_bullish['strength'],
                    'type': 'regular',
                    'confidence': regular_bullish['confidence'],
                    'time_span': regular_bullish['time_span']
                }
            elif regular_bearish['detected']:
                return {
                    'divergence': 'bearish',
                    'strength': regular_bearish['strength'],
                    'type': 'regular',
                    'confidence': regular_bearish['confidence'],
                    'time_span': regular_bearish['time_span']
                }
            elif hidden_bullish['detected']:
                return {
                    'divergence': 'bullish',
                    'strength': hidden_bullish['strength'],
                    'type': 'hidden',
                    'confidence': hidden_bullish['confidence'],
                    'time_span': hidden_bullish['time_span']
                }
            elif hidden_bearish['detected']:
                return {
                    'divergence': 'bearish',
                    'strength': hidden_bearish['strength'],
                    'type': 'hidden',
                    'confidence': hidden_bearish['confidence'],
                    'time_span': hidden_bearish['time_span']
                }
            else:
                return {'divergence': 'none', 'strength': 0, 'type': None}
                
        except Exception as e:
            logger.error(f"Error in divergence detection: {e}")
            return {'divergence': 'none', 'strength': 0, 'type': None}
    
    @staticmethod
    def calculate_divergence(prices: pd.Series, indicator: pd.Series, lookback: int = 14) -> Dict[str, Any]:
        """تشخیص واگرایی بین قیمت و اندیکاتور"""
        
        if len(prices) < lookback * 2 or len(indicator) < lookback * 2:
            return {'bullish_divergence': False, 'bearish_divergence': False, 'strength': 0}
        
        # Find recent peaks and troughs
        price_peaks = []
        price_troughs = []
        indicator_peaks = []
        indicator_troughs = []
        
        for i in range(lookback, len(prices) - lookback):
            # Price peaks and troughs
            if prices.iloc[i] == prices.iloc[i-lookback:i+lookback+1].max():
                price_peaks.append((i, prices.iloc[i]))
            if prices.iloc[i] == prices.iloc[i-lookback:i+lookback+1].min():
                price_troughs.append((i, prices.iloc[i]))
            
            # Indicator peaks and troughs
            if indicator.iloc[i] == indicator.iloc[i-lookback:i+lookback+1].max():
                indicator_peaks.append((i, indicator.iloc[i]))
            if indicator.iloc[i] == indicator.iloc[i-lookback:i+lookback+1].min():
                indicator_troughs.append((i, indicator.iloc[i]))
        
        bullish_divergence = False
        bearish_divergence = False
        strength = 0
        
        # Check for bullish divergence (price makes lower low, indicator makes higher low)
        if len(price_troughs) >= 2 and len(indicator_troughs) >= 2:
            last_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            
            # Find corresponding indicator troughs
            last_ind_trough = None
            prev_ind_trough = None
            
            for ind_trough in indicator_troughs:
                if abs(ind_trough[0] - last_price_trough[0]) <= 5:  # Within 5 periods
                    last_ind_trough = ind_trough
                if abs(ind_trough[0] - prev_price_trough[0]) <= 5:
                    prev_ind_trough = ind_trough
            
            if (last_ind_trough and prev_ind_trough and 
                last_price_trough[1] < prev_price_trough[1] and 
                last_ind_trough[1] > prev_ind_trough[1]):
                bullish_divergence = True
                strength += 1
        
        # Check for bearish divergence (price makes higher high, indicator makes lower high)
        if len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
            last_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            
            # Find corresponding indicator peaks
            last_ind_peak = None
            prev_ind_peak = None
            
            for ind_peak in indicator_peaks:
                if abs(ind_peak[0] - last_price_peak[0]) <= 5:
                    last_ind_peak = ind_peak
                if abs(ind_peak[0] - prev_price_peak[0]) <= 5:
                    prev_ind_peak = ind_peak
            
            if (last_ind_peak and prev_ind_peak and 
                last_price_peak[1] > prev_price_peak[1] and 
                last_ind_peak[1] < prev_ind_peak[1]):
                bearish_divergence = True
                strength += 1
        
        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'strength': strength,
            'description': AdvancedIndicators._get_divergence_description(bullish_divergence, bearish_divergence, strength)
        }
    
    @staticmethod
    def _get_divergence_description(bullish: bool, bearish: bool, strength: int) -> str:
        """توضیح واگرایی"""
        if bullish and bearish:
            return "واگرایی مختلط - نیاز به بررسی بیشتر"
        elif bullish:
            return f"واگرایی صعودی - قدرت {strength}/2"
        elif bearish:
            return f"واگرایی نزولی - قدرت {strength}/2"
        else:
            return "واگرایی مشاهده نشد"
    
    @staticmethod
    def calculate_market_structure(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, Any]:
        """تحلیل ساختار بازار (Higher Highs, Lower Lows, etc.)"""
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        lookback = 5
        
        for i in range(lookback, len(high) - lookback):
            if high.iloc[i] == high.iloc[i-lookback:i+lookback+1].max():
                swing_highs.append((i, high.iloc[i]))
            if low.iloc[i] == low.iloc[i-lookback:i+lookback+1].min():
                swing_lows.append((i, low.iloc[i]))
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                'trend': 'نامشخص',
                'structure': 'داده ناکافی',
                'strength': 0
            }
        
        # Analyze recent structure
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        # Check for higher highs and higher lows (uptrend)
        higher_highs = all(recent_highs[i][1] > recent_highs[i-1][1] for i in range(1, len(recent_highs)))
        higher_lows = all(recent_lows[i][1] > recent_lows[i-1][1] for i in range(1, len(recent_lows)))
        
        # Check for lower highs and lower lows (downtrend)
        lower_highs = all(recent_highs[i][1] < recent_highs[i-1][1] for i in range(1, len(recent_highs)))
        lower_lows = all(recent_lows[i][1] < recent_lows[i-1][1] for i in range(1, len(recent_lows)))
        
        # Determine trend and structure
        if higher_highs and higher_lows:
            trend = 'صعودی قوی'
            structure = 'HH & HL'
            strength = 3
        elif higher_highs or higher_lows:
            trend = 'صعودی ضعیف'
            structure = 'HH یا HL'
            strength = 2
        elif lower_highs and lower_lows:
            trend = 'نزولی قوی'
            structure = 'LH & LL'
            strength = 3
        elif lower_highs or lower_lows:
            trend = 'نزولی ضعیف'
            structure = 'LH یا LL'
            strength = 2
        else:
            trend = 'خنثی'
            structure = 'رنج'
            strength = 1
        
        return {
            'trend': trend,
            'structure': structure,
            'strength': strength,
            'recent_highs': [h[1] for h in recent_highs],
            'recent_lows': [l[1] for l in recent_lows]
        }
    
    @staticmethod
    def calculate_volume_profile(close: pd.Series, volume: pd.Series, bins: int = 20) -> Dict[str, Any]:
        """محاسبه پروفایل حجم"""
        
        if len(close) < 50 or len(volume) < 50:
            return {'error': 'داده ناکافی'}
        
        # Create price bins
        price_min = close.min()
        price_max = close.max()
        price_range = price_max - price_min
        bin_size = price_range / bins
        
        volume_profile = {}
        
        for i in range(bins):
            bin_low = price_min + (i * bin_size)
            bin_high = price_min + ((i + 1) * bin_size)
            bin_center = (bin_low + bin_high) / 2
            
            # Find volumes in this price range
            mask = (close >= bin_low) & (close < bin_high)
            bin_volume = volume[mask].sum()
            
            volume_profile[bin_center] = bin_volume
        
        # Find high volume nodes (HVN) and low volume nodes (LVN)
        sorted_volumes = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        
        hvn = sorted_volumes[:3]  # Top 3 high volume nodes
        lvn = sorted(volume_profile.items(), key=lambda x: x[1])[:3]  # Bottom 3 low volume nodes
        
        # Point of Control (POC) - highest volume price level
        poc = max(volume_profile.items(), key=lambda x: x[1])
        
        return {
            'poc': poc[0],  # Point of Control price
            'poc_volume': poc[1],
            'hvn': [{'price': h[0], 'volume': h[1]} for h in hvn],
            'lvn': [{'price': l[0], 'volume': l[1]} for l in lvn],
            'current_price': close.iloc[-1]
        }
