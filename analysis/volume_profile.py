"""
Advanced Volume Profile Analysis
Provides sophisticated volume analysis including VWAP, volume clusters, and market microstructure
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class VolumeProfileAnalyzer:
    """Advanced volume profile analysis for market microstructure insights"""
    
    def __init__(self, price_levels: int = 50):
        self.price_levels = price_levels
        self.min_volume_threshold = 0.01  # Minimum volume percentage to consider
    
    def analyze_volume_profile(self, df: pd.DataFrame, lookback_periods: int = 100) -> Dict[str, Any]:
        """Comprehensive volume profile analysis"""
        try:
            if len(df) < lookback_periods:
                lookback_periods = len(df)
            
            recent_data = df.tail(lookback_periods).copy()
            
            # Core volume profile calculations
            volume_profile = self._calculate_volume_profile(recent_data)
            poc_analysis = self._find_point_of_control(volume_profile)
            value_areas = self._calculate_value_areas(volume_profile)
            
            # Advanced volume metrics
            vwap_analysis = self._calculate_advanced_vwap(recent_data)
            volume_clusters = self._identify_volume_clusters(volume_profile)
            volume_gaps = self._identify_volume_gaps(volume_profile)
            
            # Market microstructure analysis
            order_flow = self._analyze_order_flow(recent_data)
            volume_momentum = self._calculate_volume_momentum(recent_data)
            
            # Price-volume relationship
            pv_correlation = self._analyze_price_volume_correlation(recent_data)
            
            return {
                'volume_profile': volume_profile,
                'point_of_control': poc_analysis,
                'value_areas': value_areas,
                'vwap_analysis': vwap_analysis,
                'volume_clusters': volume_clusters,
                'volume_gaps': volume_gaps,
                'order_flow': order_flow,
                'volume_momentum': volume_momentum,
                'price_volume_correlation': pv_correlation,
                'market_structure_score': self._calculate_market_structure_score(
                    poc_analysis, value_areas, volume_clusters, order_flow
                ),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in volume profile analysis: {e}")
            return self._empty_volume_profile_result()
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile across price levels"""
        try:
            high_price = df['high'].max()
            low_price = df['low'].min()
            price_range = high_price - low_price
            
            if price_range == 0:
                return {'levels': [], 'total_volume': 0}
            
            # Create price levels
            price_step = price_range / self.price_levels
            price_levels = np.arange(low_price, high_price + price_step, price_step)
            
            volume_at_price = {}
            total_volume = 0
            
            # Calculate volume at each price level
            for _, row in df.iterrows():
                candle_high = row['high']
                candle_low = row['low']
                candle_volume = row['volume']
                candle_range = candle_high - candle_low
                
                if candle_range == 0:
                    # All volume at one price
                    price_level = self._find_price_level(candle_low, price_levels)
                    volume_at_price[price_level] = volume_at_price.get(price_level, 0) + candle_volume
                else:
                    # Distribute volume across the candle's price range
                    for price in price_levels:
                        if candle_low <= price <= candle_high:
                            # Volume distribution based on position within candle
                            volume_portion = candle_volume / len([p for p in price_levels if candle_low <= p <= candle_high])
                            volume_at_price[price] = volume_at_price.get(price, 0) + volume_portion
                
                total_volume += candle_volume
            
            # Convert to sorted list
            volume_profile_list = []
            for price in sorted(price_levels):
                volume = volume_at_price.get(price, 0)
                volume_percentage = (volume / total_volume * 100) if total_volume > 0 else 0
                
                if volume_percentage >= self.min_volume_threshold:
                    volume_profile_list.append({
                        'price': round(price, 8),
                        'volume': volume,
                        'volume_percentage': round(volume_percentage, 2)
                    })
            
            return {
                'levels': volume_profile_list,
                'total_volume': total_volume,
                'price_range': {'high': high_price, 'low': low_price},
                'num_levels': len(volume_profile_list)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {e}")
            return {'levels': [], 'total_volume': 0}
    
    def _find_price_level(self, price: float, price_levels: np.ndarray) -> float:
        """Find the closest price level for a given price"""
        return price_levels[np.argmin(np.abs(price_levels - price))]
    
    def _find_point_of_control(self, volume_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Find Point of Control (POC) - price level with highest volume"""
        try:
            levels = volume_profile.get('levels', [])
            if not levels:
                return {'price': 0, 'volume': 0, 'volume_percentage': 0}
            
            poc_level = max(levels, key=lambda x: x['volume'])
            
            return {
                'price': poc_level['price'],
                'volume': poc_level['volume'],
                'volume_percentage': poc_level['volume_percentage'],
                'significance': 'high' if poc_level['volume_percentage'] > 5 else 'medium' if poc_level['volume_percentage'] > 2 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error finding point of control: {e}")
            return {'price': 0, 'volume': 0, 'volume_percentage': 0, 'significance': 'none'}
    
    def _calculate_value_areas(self, volume_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Value Area High (VAH) and Value Area Low (VAL) - 70% of volume"""
        try:
            levels = volume_profile.get('levels', [])
            total_volume = volume_profile.get('total_volume', 0)
            
            if not levels or total_volume == 0:
                return {'vah': 0, 'val': 0, 'value_area_volume': 0}
            
            # Sort levels by volume (descending)
            sorted_levels = sorted(levels, key=lambda x: x['volume'], reverse=True)
            
            # Find levels that make up 70% of total volume
            target_volume = total_volume * 0.7
            accumulated_volume = 0
            value_area_levels = []
            
            for level in sorted_levels:
                accumulated_volume += level['volume']
                value_area_levels.append(level)
                if accumulated_volume >= target_volume:
                    break
            
            if not value_area_levels:
                return {'vah': 0, 'val': 0, 'value_area_volume': 0}
            
            # Find highest and lowest prices in value area
            prices = [level['price'] for level in value_area_levels]
            vah = max(prices)  # Value Area High
            val = min(prices)  # Value Area Low
            
            return {
                'vah': vah,
                'val': val,
                'value_area_volume': accumulated_volume,
                'value_area_percentage': round(accumulated_volume / total_volume * 100, 2),
                'num_levels_in_va': len(value_area_levels)
            }
            
        except Exception as e:
            logger.error(f"Error calculating value areas: {e}")
            return {'vah': 0, 'val': 0, 'value_area_volume': 0}
    
    def _calculate_advanced_vwap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced VWAP metrics"""
        try:
            # Standard VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            # VWAP bands (standard deviation bands)
            price_volume = typical_price * df['volume']
            cumulative_pv = price_volume.cumsum()
            cumulative_volume = df['volume'].cumsum()
            
            # Calculate variance for VWAP bands
            squared_diff = ((typical_price - vwap) ** 2) * df['volume']
            cumulative_squared_diff = squared_diff.cumsum()
            variance = cumulative_squared_diff / cumulative_volume
            std_dev = np.sqrt(variance)
            
            current_vwap = vwap.iloc[-1]
            current_std = std_dev.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # VWAP bands
            vwap_upper_1 = current_vwap + current_std
            vwap_lower_1 = current_vwap - current_std
            vwap_upper_2 = current_vwap + (current_std * 2)
            vwap_lower_2 = current_vwap - (current_std * 2)
            
            # Price position relative to VWAP
            vwap_deviation = (current_price - current_vwap) / current_vwap * 100
            
            # VWAP trend analysis
            vwap_slope = self._calculate_slope(vwap.tail(10))
            
            return {
                'current_vwap': current_vwap,
                'current_price': current_price,
                'vwap_deviation_pct': round(vwap_deviation, 2),
                'bands': {
                    'upper_1': vwap_upper_1,
                    'lower_1': vwap_lower_1,
                    'upper_2': vwap_upper_2,
                    'lower_2': vwap_lower_2
                },
                'price_position': self._determine_vwap_position(current_price, current_vwap, vwap_upper_1, vwap_lower_1),
                'vwap_trend': 'bullish' if vwap_slope > 0.001 else 'bearish' if vwap_slope < -0.001 else 'neutral',
                'vwap_slope': vwap_slope
            }
            
        except Exception as e:
            logger.error(f"Error calculating advanced VWAP: {e}")
            return {'current_vwap': 0, 'vwap_deviation_pct': 0}
    
    def _calculate_slope(self, series: pd.Series) -> float:
        """Calculate slope of a series"""
        try:
            if len(series) < 2:
                return 0
            
            x = np.arange(len(series))
            y = series.values
            
            # Linear regression slope
            slope = np.polyfit(x, y, 1)[0]
            return slope
            
        except Exception as e:
            logger.error(f"Error calculating slope: {e}")
            return 0
    
    def _determine_vwap_position(self, price: float, vwap: float, upper_band: float, lower_band: float) -> str:
        """Determine price position relative to VWAP bands"""
        if price > upper_band:
            return 'above_upper_band'
        elif price > vwap:
            return 'above_vwap'
        elif price < lower_band:
            return 'below_lower_band'
        else:
            return 'below_vwap'
    
    def _identify_volume_clusters(self, volume_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify significant volume clusters"""
        try:
            levels = volume_profile.get('levels', [])
            if len(levels) < 3:
                return []
            
            clusters = []
            avg_volume_pct = sum(level['volume_percentage'] for level in levels) / len(levels)
            threshold = avg_volume_pct * 2  # Clusters are 2x average volume
            
            current_cluster = []
            
            for i, level in enumerate(levels):
                if level['volume_percentage'] >= threshold:
                    current_cluster.append(level)
                else:
                    if len(current_cluster) >= 2:  # Minimum 2 levels for a cluster
                        cluster_info = self._analyze_cluster(current_cluster)
                        clusters.append(cluster_info)
                    current_cluster = []
            
            # Check final cluster
            if len(current_cluster) >= 2:
                cluster_info = self._analyze_cluster(current_cluster)
                clusters.append(cluster_info)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error identifying volume clusters: {e}")
            return []
    
    def _analyze_cluster(self, cluster_levels: List[Dict]) -> Dict[str, Any]:
        """Analyze a volume cluster"""
        try:
            prices = [level['price'] for level in cluster_levels]
            volumes = [level['volume'] for level in cluster_levels]
            volume_percentages = [level['volume_percentage'] for level in cluster_levels]
            
            return {
                'price_range': {'high': max(prices), 'low': min(prices)},
                'center_price': sum(prices) / len(prices),
                'total_volume': sum(volumes),
                'total_volume_percentage': sum(volume_percentages),
                'num_levels': len(cluster_levels),
                'strength': 'strong' if sum(volume_percentages) > 10 else 'moderate' if sum(volume_percentages) > 5 else 'weak'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cluster: {e}")
            return {}
    
    def _identify_volume_gaps(self, volume_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify volume gaps (low volume areas)"""
        try:
            levels = volume_profile.get('levels', [])
            if len(levels) < 5:
                return []
            
            gaps = []
            avg_volume_pct = sum(level['volume_percentage'] for level in levels) / len(levels)
            gap_threshold = avg_volume_pct * 0.3  # Gaps are areas with <30% of average volume
            
            current_gap = []
            
            for level in levels:
                if level['volume_percentage'] <= gap_threshold:
                    current_gap.append(level)
                else:
                    if len(current_gap) >= 3:  # Minimum 3 levels for a gap
                        gap_info = self._analyze_gap(current_gap)
                        gaps.append(gap_info)
                    current_gap = []
            
            # Check final gap
            if len(current_gap) >= 3:
                gap_info = self._analyze_gap(current_gap)
                gaps.append(gap_info)
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error identifying volume gaps: {e}")
            return []
    
    def _analyze_gap(self, gap_levels: List[Dict]) -> Dict[str, Any]:
        """Analyze a volume gap"""
        try:
            prices = [level['price'] for level in gap_levels]
            volume_percentages = [level['volume_percentage'] for level in gap_levels]
            
            return {
                'price_range': {'high': max(prices), 'low': min(prices)},
                'center_price': sum(prices) / len(prices),
                'avg_volume_percentage': sum(volume_percentages) / len(volume_percentages),
                'num_levels': len(gap_levels),
                'gap_size': max(prices) - min(prices),
                'significance': 'high' if len(gap_levels) > 5 else 'medium' if len(gap_levels) > 3 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing gap: {e}")
            return {}
    
    def _analyze_order_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze order flow patterns"""
        try:
            # Calculate buying/selling pressure approximation
            buying_pressure = []
            selling_pressure = []
            
            for _, row in df.iterrows():
                high = row['high']
                low = row['low']
                close = row['close']
                volume = row['volume']
                
                # Approximate buying vs selling pressure
                if high != low:
                    buy_ratio = (close - low) / (high - low)
                    sell_ratio = 1 - buy_ratio
                else:
                    buy_ratio = sell_ratio = 0.5
                
                buying_pressure.append(volume * buy_ratio)
                selling_pressure.append(volume * sell_ratio)
            
            df_flow = pd.DataFrame({
                'buying_pressure': buying_pressure,
                'selling_pressure': selling_pressure
            })
            
            # Calculate cumulative order flow
            cumulative_buying = df_flow['buying_pressure'].sum()
            cumulative_selling = df_flow['selling_pressure'].sum()
            
            # Order flow balance
            total_flow = cumulative_buying + cumulative_selling
            buying_percentage = (cumulative_buying / total_flow * 100) if total_flow > 0 else 50
            
            # Recent order flow trend (last 20 periods)
            recent_buying = df_flow['buying_pressure'].tail(20).sum()
            recent_selling = df_flow['selling_pressure'].tail(20).sum()
            recent_total = recent_buying + recent_selling
            recent_buying_pct = (recent_buying / recent_total * 100) if recent_total > 0 else 50
            
            return {
                'cumulative_buying_pressure': cumulative_buying,
                'cumulative_selling_pressure': cumulative_selling,
                'buying_percentage': round(buying_percentage, 2),
                'selling_percentage': round(100 - buying_percentage, 2),
                'recent_buying_percentage': round(recent_buying_pct, 2),
                'order_flow_bias': 'bullish' if buying_percentage > 55 else 'bearish' if buying_percentage < 45 else 'neutral',
                'recent_bias': 'bullish' if recent_buying_pct > 55 else 'bearish' if recent_buying_pct < 45 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order flow: {e}")
            return {'order_flow_bias': 'neutral', 'recent_bias': 'neutral'}
    
    def _calculate_volume_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume momentum indicators"""
        try:
            volume = df['volume']
            
            # Volume moving averages
            vol_ma_short = volume.rolling(window=10).mean()
            vol_ma_long = volume.rolling(window=30).mean()
            
            # Volume momentum
            vol_momentum = vol_ma_short / vol_ma_long
            current_vol_momentum = vol_momentum.iloc[-1] if len(vol_momentum) > 0 else 1
            
            # Volume rate of change
            vol_roc = volume.pct_change(periods=10).iloc[-1] * 100
            
            # Volume trend
            vol_trend = self._calculate_slope(volume.tail(20))
            
            return {
                'volume_momentum': round(current_vol_momentum, 3),
                'volume_roc_pct': round(vol_roc, 2) if not np.isnan(vol_roc) else 0,
                'volume_trend': 'increasing' if vol_trend > 0 else 'decreasing' if vol_trend < 0 else 'stable',
                'momentum_signal': 'strong' if current_vol_momentum > 1.5 else 'weak' if current_vol_momentum < 0.7 else 'normal'
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume momentum: {e}")
            return {'volume_momentum': 1, 'momentum_signal': 'normal'}
    
    def _analyze_price_volume_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price-volume correlation"""
        try:
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            
            # Remove NaN values
            valid_data = pd.DataFrame({
                'price_change': price_change,
                'volume_change': volume_change
            }).dropna()
            
            if len(valid_data) < 10:
                return {'correlation': 0, 'relationship': 'insufficient_data'}
            
            correlation = valid_data['price_change'].corr(valid_data['volume_change'])
            
            # Interpret correlation
            if correlation > 0.3:
                relationship = 'positive_strong'
            elif correlation > 0.1:
                relationship = 'positive_weak'
            elif correlation < -0.3:
                relationship = 'negative_strong'
            elif correlation < -0.1:
                relationship = 'negative_weak'
            else:
                relationship = 'no_correlation'
            
            return {
                'correlation': round(correlation, 3) if not np.isnan(correlation) else 0,
                'relationship': relationship,
                'interpretation': self._interpret_pv_correlation(relationship)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing price-volume correlation: {e}")
            return {'correlation': 0, 'relationship': 'unknown'}
    
    def _interpret_pv_correlation(self, relationship: str) -> str:
        """Interpret price-volume correlation"""
        interpretations = {
            'positive_strong': 'Price and volume move together strongly - healthy trend',
            'positive_weak': 'Price and volume have weak positive relationship',
            'negative_strong': 'Price and volume move opposite - potential divergence',
            'negative_weak': 'Price and volume have weak negative relationship',
            'no_correlation': 'Price and volume are not correlated',
            'insufficient_data': 'Not enough data for correlation analysis'
        }
        return interpretations.get(relationship, 'Unknown relationship')
    
    def _calculate_market_structure_score(self, poc: Dict, value_areas: Dict, 
                                        clusters: List[Dict], order_flow: Dict) -> float:
        """Calculate overall market structure score"""
        try:
            score = 0
            
            # POC significance
            poc_significance = poc.get('significance', 'none')
            if poc_significance == 'high':
                score += 3
            elif poc_significance == 'medium':
                score += 2
            elif poc_significance == 'low':
                score += 1
            
            # Value area strength
            va_percentage = value_areas.get('value_area_percentage', 0)
            if va_percentage > 75:
                score += 3
            elif va_percentage > 65:
                score += 2
            elif va_percentage > 55:
                score += 1
            
            # Volume clusters
            strong_clusters = sum(1 for cluster in clusters if cluster.get('strength') == 'strong')
            score += min(3, strong_clusters)
            
            # Order flow bias
            order_flow_bias = order_flow.get('order_flow_bias', 'neutral')
            buying_pct = order_flow.get('buying_percentage', 50)
            if order_flow_bias != 'neutral':
                if abs(buying_pct - 50) > 15:
                    score += 2
                elif abs(buying_pct - 50) > 10:
                    score += 1
            
            return min(10, score)  # Cap at 10
            
        except Exception as e:
            logger.error(f"Error calculating market structure score: {e}")
            return 0
    
    def _empty_volume_profile_result(self) -> Dict[str, Any]:
        """Return empty volume profile result"""
        return {
            'volume_profile': {'levels': [], 'total_volume': 0},
            'point_of_control': {'price': 0, 'volume': 0, 'significance': 'none'},
            'value_areas': {'vah': 0, 'val': 0, 'value_area_volume': 0},
            'vwap_analysis': {'current_vwap': 0, 'vwap_deviation_pct': 0},
            'volume_clusters': [],
            'volume_gaps': [],
            'order_flow': {'order_flow_bias': 'neutral', 'recent_bias': 'neutral'},
            'volume_momentum': {'volume_momentum': 1, 'momentum_signal': 'normal'},
            'price_volume_correlation': {'correlation': 0, 'relationship': 'unknown'},
            'market_structure_score': 0,
            'analysis_timestamp': datetime.now().isoformat()
        }
