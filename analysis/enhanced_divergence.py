"""
Enhanced Divergence Detection System
Provides advanced divergence analysis for technical indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
# from scipy.signal import find_peaks  # Commented out due to DLL issues
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EnhancedDivergenceDetector:
    """Advanced divergence detection with multiple algorithms and confidence scoring"""
    
    def __init__(self, min_peak_distance: int = 5, peak_prominence: float = 0.01):
        self.min_peak_distance = min_peak_distance
        self.peak_prominence = peak_prominence
    
    def detect_all_divergences(self, prices: pd.Series, indicator: pd.Series, 
                             lookback: int = 20) -> Dict[str, Any]:
        """Comprehensive divergence detection with multiple types"""
        try:
            if len(prices) < lookback or len(indicator) < lookback:
                return self._empty_divergence_result()
            
            # Normalize data for better peak detection
            norm_prices = self._normalize_series(prices.tail(lookback))
            norm_indicator = self._normalize_series(indicator.tail(lookback))
            
            # Find peaks and valleys
            price_peaks, price_valleys = self._find_peaks_and_valleys(norm_prices)
            indicator_peaks, indicator_valleys = self._find_peaks_and_valleys(norm_indicator)
            
            # Check all divergence types
            results = {
                'regular_bullish': self._check_regular_bullish_divergence(
                    price_valleys, indicator_valleys, prices.tail(lookback), indicator.tail(lookback)
                ),
                'regular_bearish': self._check_regular_bearish_divergence(
                    price_peaks, indicator_peaks, prices.tail(lookback), indicator.tail(lookback)
                ),
                'hidden_bullish': self._check_hidden_bullish_divergence(
                    price_valleys, indicator_valleys, prices.tail(lookback), indicator.tail(lookback)
                ),
                'hidden_bearish': self._check_hidden_bearish_divergence(
                    price_peaks, indicator_peaks, prices.tail(lookback), indicator.tail(lookback)
                )
            }
            
            # Find the strongest divergence
            strongest = self._find_strongest_divergence(results)
            
            return {
                'primary_divergence': strongest,
                'all_divergences': results,
                'confidence_score': strongest.get('confidence', 0) if strongest else 0,
                'analysis_timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive divergence detection: {e}")
            return self._empty_divergence_result()
    
    def _normalize_series(self, series: pd.Series) -> pd.Series:
        """Normalize series to 0-1 range for better comparison"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    def _find_peaks_and_valleys(self, series: pd.Series) -> Tuple[List[int], List[int]]:
        """Find peaks and valleys using simple algorithm (scipy alternative)"""
        try:
            values = series.values
            peaks = []
            valleys = []
            
            # Simple peak detection algorithm
            for i in range(self.min_peak_distance, len(values) - self.min_peak_distance):
                # Check for peak
                is_peak = True
                for j in range(1, self.min_peak_distance + 1):
                    if values[i] <= values[i-j] or values[i] <= values[i+j]:
                        is_peak = False
                        break
                
                if is_peak:
                    # Check prominence
                    left_min = min(values[max(0, i-10):i])
                    right_min = min(values[i+1:min(len(values), i+11)])
                    prominence = values[i] - max(left_min, right_min)
                    if prominence >= self.peak_prominence:
                        peaks.append(i)
                
                # Check for valley
                is_valley = True
                for j in range(1, self.min_peak_distance + 1):
                    if values[i] >= values[i-j] or values[i] >= values[i+j]:
                        is_valley = False
                        break
                
                if is_valley:
                    # Check prominence for valley
                    left_max = max(values[max(0, i-10):i])
                    right_max = max(values[i+1:min(len(values), i+11)])
                    prominence = min(left_max, right_max) - values[i]
                    if prominence >= self.peak_prominence:
                        valleys.append(i)
            
            return peaks, valleys
            
        except Exception as e:
            logger.error(f"Error finding peaks and valleys: {e}")
            return [], []
    
    def _check_regular_bullish_divergence(self, price_valleys: List[int], 
                                        indicator_valleys: List[int],
                                        prices: pd.Series, indicator: pd.Series) -> Dict[str, Any]:
        """Check for regular bullish divergence: lower price lows, higher indicator lows"""
        if len(price_valleys) < 2 or len(indicator_valleys) < 2:
            return {'detected': False, 'strength': 0, 'confidence': 0}
        
        try:
            # Get the two most recent valleys
            recent_price_valleys = sorted(price_valleys)[-2:]
            recent_indicator_valleys = sorted(indicator_valleys)[-2:]
            
            # Check if we have valid valleys within reasonable time distance
            if abs(recent_price_valleys[1] - recent_indicator_valleys[1]) > 5:
                return {'detected': False, 'strength': 0, 'confidence': 0}
            
            # Price analysis: lower low
            price_low1 = prices.iloc[recent_price_valleys[0]]
            price_low2 = prices.iloc[recent_price_valleys[1]]
            price_lower_low = price_low2 < price_low1
            
            # Indicator analysis: higher low
            ind_low1 = indicator.iloc[recent_indicator_valleys[0]]
            ind_low2 = indicator.iloc[recent_indicator_valleys[1]]
            indicator_higher_low = ind_low2 > ind_low1
            
            if price_lower_low and indicator_higher_low:
                # Calculate strength and confidence
                price_diff_pct = abs((price_low2 - price_low1) / price_low1) * 100
                indicator_diff_pct = abs((ind_low2 - ind_low1) / ind_low1) * 100
                
                strength = min(10, (price_diff_pct + indicator_diff_pct) * 2)
                confidence = min(95, 60 + strength * 3)
                
                time_span = recent_price_valleys[1] - recent_price_valleys[0]
                
                return {
                    'detected': True,
                    'strength': strength,
                    'confidence': confidence,
                    'time_span': time_span,
                    'price_change_pct': -price_diff_pct,
                    'indicator_change_pct': indicator_diff_pct
                }
            
            return {'detected': False, 'strength': 0, 'confidence': 0}
            
        except Exception as e:
            logger.error(f"Error checking regular bullish divergence: {e}")
            return {'detected': False, 'strength': 0, 'confidence': 0}
    
    def _check_regular_bearish_divergence(self, price_peaks: List[int], 
                                        indicator_peaks: List[int],
                                        prices: pd.Series, indicator: pd.Series) -> Dict[str, Any]:
        """Check for regular bearish divergence: higher price highs, lower indicator highs"""
        if len(price_peaks) < 2 or len(indicator_peaks) < 2:
            return {'detected': False, 'strength': 0, 'confidence': 0}
        
        try:
            # Get the two most recent peaks
            recent_price_peaks = sorted(price_peaks)[-2:]
            recent_indicator_peaks = sorted(indicator_peaks)[-2:]
            
            # Check if we have valid peaks within reasonable time distance
            if abs(recent_price_peaks[1] - recent_indicator_peaks[1]) > 5:
                return {'detected': False, 'strength': 0, 'confidence': 0}
            
            # Price analysis: higher high
            price_high1 = prices.iloc[recent_price_peaks[0]]
            price_high2 = prices.iloc[recent_price_peaks[1]]
            price_higher_high = price_high2 > price_high1
            
            # Indicator analysis: lower high
            ind_high1 = indicator.iloc[recent_indicator_peaks[0]]
            ind_high2 = indicator.iloc[recent_indicator_peaks[1]]
            indicator_lower_high = ind_high2 < ind_high1
            
            if price_higher_high and indicator_lower_high:
                # Calculate strength and confidence
                price_diff_pct = abs((price_high2 - price_high1) / price_high1) * 100
                indicator_diff_pct = abs((ind_high2 - ind_high1) / ind_high1) * 100
                
                strength = min(10, (price_diff_pct + indicator_diff_pct) * 2)
                confidence = min(95, 60 + strength * 3)
                
                time_span = recent_price_peaks[1] - recent_price_peaks[0]
                
                return {
                    'detected': True,
                    'strength': strength,
                    'confidence': confidence,
                    'time_span': time_span,
                    'price_change_pct': price_diff_pct,
                    'indicator_change_pct': -indicator_diff_pct
                }
            
            return {'detected': False, 'strength': 0, 'confidence': 0}
            
        except Exception as e:
            logger.error(f"Error checking regular bearish divergence: {e}")
            return {'detected': False, 'strength': 0, 'confidence': 0}
    
    def _check_hidden_bullish_divergence(self, price_valleys: List[int], 
                                       indicator_valleys: List[int],
                                       prices: pd.Series, indicator: pd.Series) -> Dict[str, Any]:
        """Check for hidden bullish divergence: higher price lows, lower indicator lows"""
        if len(price_valleys) < 2 or len(indicator_valleys) < 2:
            return {'detected': False, 'strength': 0, 'confidence': 0}
        
        try:
            # Get the two most recent valleys
            recent_price_valleys = sorted(price_valleys)[-2:]
            recent_indicator_valleys = sorted(indicator_valleys)[-2:]
            
            # Check if we have valid valleys within reasonable time distance
            if abs(recent_price_valleys[1] - recent_indicator_valleys[1]) > 5:
                return {'detected': False, 'strength': 0, 'confidence': 0}
            
            # Price analysis: higher low
            price_low1 = prices.iloc[recent_price_valleys[0]]
            price_low2 = prices.iloc[recent_price_valleys[1]]
            price_higher_low = price_low2 > price_low1
            
            # Indicator analysis: lower low
            ind_low1 = indicator.iloc[recent_indicator_valleys[0]]
            ind_low2 = indicator.iloc[recent_indicator_valleys[1]]
            indicator_lower_low = ind_low2 < ind_low1
            
            if price_higher_low and indicator_lower_low:
                # Calculate strength and confidence (lower than regular divergence)
                price_diff_pct = abs((price_low2 - price_low1) / price_low1) * 100
                indicator_diff_pct = abs((ind_low2 - ind_low1) / ind_low1) * 100
                
                strength = min(8, (price_diff_pct + indicator_diff_pct) * 1.5)
                confidence = min(80, 45 + strength * 3)
                
                time_span = recent_price_valleys[1] - recent_price_valleys[0]
                
                return {
                    'detected': True,
                    'strength': strength,
                    'confidence': confidence,
                    'time_span': time_span,
                    'price_change_pct': price_diff_pct,
                    'indicator_change_pct': -indicator_diff_pct
                }
            
            return {'detected': False, 'strength': 0, 'confidence': 0}
            
        except Exception as e:
            logger.error(f"Error checking hidden bullish divergence: {e}")
            return {'detected': False, 'strength': 0, 'confidence': 0}
    
    def _check_hidden_bearish_divergence(self, price_peaks: List[int], 
                                       indicator_peaks: List[int],
                                       prices: pd.Series, indicator: pd.Series) -> Dict[str, Any]:
        """Check for hidden bearish divergence: lower price highs, higher indicator highs"""
        if len(price_peaks) < 2 or len(indicator_peaks) < 2:
            return {'detected': False, 'strength': 0, 'confidence': 0}
        
        try:
            # Get the two most recent peaks
            recent_price_peaks = sorted(price_peaks)[-2:]
            recent_indicator_peaks = sorted(indicator_peaks)[-2:]
            
            # Check if we have valid peaks within reasonable time distance
            if abs(recent_price_peaks[1] - recent_indicator_peaks[1]) > 5:
                return {'detected': False, 'strength': 0, 'confidence': 0}
            
            # Price analysis: lower high
            price_high1 = prices.iloc[recent_price_peaks[0]]
            price_high2 = prices.iloc[recent_price_peaks[1]]
            price_lower_high = price_high2 < price_high1
            
            # Indicator analysis: higher high
            ind_high1 = indicator.iloc[recent_indicator_peaks[0]]
            ind_high2 = indicator.iloc[recent_indicator_peaks[1]]
            indicator_higher_high = ind_high2 > ind_high1
            
            if price_lower_high and indicator_higher_high:
                # Calculate strength and confidence (lower than regular divergence)
                price_diff_pct = abs((price_high2 - price_high1) / price_high1) * 100
                indicator_diff_pct = abs((ind_high2 - ind_high1) / ind_high1) * 100
                
                strength = min(8, (price_diff_pct + indicator_diff_pct) * 1.5)
                confidence = min(80, 45 + strength * 3)
                
                time_span = recent_price_peaks[1] - recent_price_peaks[0]
                
                return {
                    'detected': True,
                    'strength': strength,
                    'confidence': confidence,
                    'time_span': time_span,
                    'price_change_pct': -price_diff_pct,
                    'indicator_change_pct': indicator_diff_pct
                }
            
            return {'detected': False, 'strength': 0, 'confidence': 0}
            
        except Exception as e:
            logger.error(f"Error checking hidden bearish divergence: {e}")
            return {'detected': False, 'strength': 0, 'confidence': 0}
    
    def _find_strongest_divergence(self, results: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
        """Find the strongest divergence from all detected divergences"""
        strongest = None
        max_strength = 0
        
        for div_type, div_result in results.items():
            if div_result.get('detected', False):
                strength = div_result.get('strength', 0)
                if strength > max_strength:
                    max_strength = strength
                    strongest = div_result.copy()
                    strongest['type'] = div_type
        
        return strongest
    
    def _empty_divergence_result(self) -> Dict[str, Any]:
        """Return empty divergence result structure"""
        return {
            'primary_divergence': None,
            'all_divergences': {
                'regular_bullish': {'detected': False, 'strength': 0, 'confidence': 0},
                'regular_bearish': {'detected': False, 'strength': 0, 'confidence': 0},
                'hidden_bullish': {'detected': False, 'strength': 0, 'confidence': 0},
                'hidden_bearish': {'detected': False, 'strength': 0, 'confidence': 0}
            },
            'confidence_score': 0,
            'analysis_timestamp': pd.Timestamp.now()
        }
