"""
Multi-Timeframe Analysis System
Provides comprehensive analysis across multiple timeframes for stronger signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """Advanced multi-timeframe analysis for enhanced signal confirmation"""
    
    def __init__(self, aggregator):
        self.aggregator = aggregator
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        self.weights = {
            '5m': 1.0,   # Current timeframe
            '15m': 1.5,  # Short-term confirmation
            '1h': 2.0,   # Medium-term trend
            '4h': 2.5,   # Strong trend confirmation
            '1d': 3.0    # Long-term trend
        }
    
    async def analyze_multi_timeframe(self, symbol: str) -> Dict[str, Any]:
        """Comprehensive multi-timeframe analysis"""
        try:
            results = {}
            timeframe_data = {}
            
            # Fetch data for all timeframes
            for tf in self.timeframes:
                try:
                    klines = await self.aggregator.get_klines(symbol, tf, 200)
                    if klines and len(klines) >= 50:
                        df = self._klines_to_dataframe(klines)
                        timeframe_data[tf] = df
                        results[tf] = await self._analyze_single_timeframe(df, tf)
                    else:
                        logger.warning(f"Insufficient data for {symbol} on {tf}")
                        results[tf] = self._empty_timeframe_result()
                except Exception as e:
                    logger.error(f"Error fetching {tf} data for {symbol}: {e}")
                    results[tf] = self._empty_timeframe_result()
            
            # Calculate composite analysis
            composite_score = self._calculate_composite_score(results)
            trend_alignment = self._analyze_trend_alignment(results)
            momentum_confluence = self._analyze_momentum_confluence(results)
            
            return {
                'symbol': symbol,
                'timeframe_results': results,
                'composite_score': composite_score,
                'trend_alignment': trend_alignment,
                'momentum_confluence': momentum_confluence,
                'signal_strength': self._determine_signal_strength(composite_score, trend_alignment),
                'recommended_action': self._get_recommended_action(composite_score, trend_alignment),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return self._empty_multi_timeframe_result(symbol)
    
    def _klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """Convert klines data to pandas DataFrame"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert to appropriate data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.set_index('timestamp', inplace=True)
        return df
    
    async def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Analyze a single timeframe"""
        try:
            # Calculate technical indicators
            indicators = self._calculate_timeframe_indicators(df)
            
            # Trend analysis
            trend_analysis = self._analyze_trend(df, indicators)
            
            # Momentum analysis
            momentum_analysis = self._analyze_momentum(indicators)
            
            # Support/Resistance levels
            sr_levels = self._find_support_resistance(df)
            
            # Volume analysis
            volume_analysis = self._analyze_volume(df)
            
            return {
                'timeframe': timeframe,
                'indicators': indicators,
                'trend': trend_analysis,
                'momentum': momentum_analysis,
                'support_resistance': sr_levels,
                'volume': volume_analysis,
                'score': self._calculate_timeframe_score(trend_analysis, momentum_analysis, volume_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {timeframe}: {e}")
            return self._empty_timeframe_result()
    
    def _calculate_timeframe_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for a timeframe"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # EMAs
            ema9 = close.ewm(span=9).mean()
            ema21 = close.ewm(span=21).mean()
            ema50 = close.ewm(span=50).mean()
            ema200 = close.ewm(span=200).mean() if len(close) >= 200 else ema50
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            # Bollinger Bands
            bb_middle = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            # ATR
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            return {
                'current_price': close.iloc[-1],
                'ema9': ema9.iloc[-1],
                'ema21': ema21.iloc[-1],
                'ema50': ema50.iloc[-1],
                'ema200': ema200.iloc[-1],
                'rsi': rsi.iloc[-1],
                'macd': macd.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'macd_histogram': macd_histogram.iloc[-1],
                'bb_upper': bb_upper.iloc[-1],
                'bb_middle': bb_middle.iloc[-1],
                'bb_lower': bb_lower.iloc[-1],
                'atr': atr.iloc[-1],
                'volume_avg': volume.rolling(window=20).mean().iloc[-1],
                'current_volume': volume.iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def _analyze_trend(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        try:
            price = indicators.get('current_price', 0)
            ema9 = indicators.get('ema9', 0)
            ema21 = indicators.get('ema21', 0)
            ema50 = indicators.get('ema50', 0)
            ema200 = indicators.get('ema200', 0)
            
            # Trend direction
            if price > ema9 > ema21 > ema50 > ema200:
                trend_direction = 'strong_bullish'
                trend_score = 10
            elif price > ema9 > ema21 > ema50:
                trend_direction = 'bullish'
                trend_score = 8
            elif price > ema9 > ema21:
                trend_direction = 'weak_bullish'
                trend_score = 6
            elif price < ema9 < ema21 < ema50 < ema200:
                trend_direction = 'strong_bearish'
                trend_score = -10
            elif price < ema9 < ema21 < ema50:
                trend_direction = 'bearish'
                trend_score = -8
            elif price < ema9 < ema21:
                trend_direction = 'weak_bearish'
                trend_score = -6
            else:
                trend_direction = 'sideways'
                trend_score = 0
            
            # Calculate trend strength based on EMA separation
            ema_separation = abs(ema9 - ema21) / price * 100
            trend_strength = min(10, ema_separation * 50)
            
            return {
                'direction': trend_direction,
                'score': trend_score,
                'strength': trend_strength,
                'ema_alignment': price > ema9 > ema21 > ema50
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {'direction': 'unknown', 'score': 0, 'strength': 0, 'ema_alignment': False}
    
    def _analyze_momentum(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        try:
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_histogram = indicators.get('macd_histogram', 0)
            
            # RSI analysis
            if rsi > 70:
                rsi_signal = 'overbought'
                rsi_score = -5
            elif rsi < 30:
                rsi_signal = 'oversold'
                rsi_score = 5
            elif 40 <= rsi <= 60:
                rsi_signal = 'neutral'
                rsi_score = 0
            else:
                rsi_signal = 'trending'
                rsi_score = 3 if rsi > 50 else -3
            
            # MACD analysis
            if macd > macd_signal and macd_histogram > 0:
                macd_signal_type = 'bullish'
                macd_score = 5
            elif macd < macd_signal and macd_histogram < 0:
                macd_signal_type = 'bearish'
                macd_score = -5
            else:
                macd_signal_type = 'neutral'
                macd_score = 0
            
            total_momentum_score = rsi_score + macd_score
            
            return {
                'rsi_value': rsi,
                'rsi_signal': rsi_signal,
                'rsi_score': rsi_score,
                'macd_signal': macd_signal_type,
                'macd_score': macd_score,
                'total_score': total_momentum_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {'total_score': 0}
    
    def _find_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
        """Find support and resistance levels"""
        try:
            high = df['high'].tail(lookback)
            low = df['low'].tail(lookback)
            close = df['close'].tail(lookback)
            
            # Find recent highs and lows
            resistance = high.max()
            support = low.min()
            
            current_price = close.iloc[-1]
            
            # Calculate distance to levels
            resistance_distance = (resistance - current_price) / current_price * 100
            support_distance = (current_price - support) / current_price * 100
            
            return {
                'resistance': resistance,
                'support': support,
                'current_price': current_price,
                'resistance_distance_pct': resistance_distance,
                'support_distance_pct': support_distance,
                'near_resistance': resistance_distance < 2,
                'near_support': support_distance < 2
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance: {e}")
            return {}
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            volume = df['volume'].tail(20)
            current_volume = volume.iloc[-1]
            avg_volume = volume.mean()
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 2:
                volume_signal = 'high'
                volume_score = 5
            elif volume_ratio > 1.5:
                volume_signal = 'above_average'
                volume_score = 3
            elif volume_ratio < 0.5:
                volume_signal = 'low'
                volume_score = -2
            else:
                volume_signal = 'normal'
                volume_score = 0
            
            return {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'signal': volume_signal,
                'score': volume_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {'score': 0}
    
    def _calculate_timeframe_score(self, trend: Dict, momentum: Dict, volume: Dict) -> float:
        """Calculate overall score for a timeframe"""
        try:
            trend_score = trend.get('score', 0)
            momentum_score = momentum.get('total_score', 0)
            volume_score = volume.get('score', 0)
            
            # Weighted combination
            total_score = (trend_score * 0.5) + (momentum_score * 0.3) + (volume_score * 0.2)
            
            return round(total_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating timeframe score: {e}")
            return 0.0
    
    def _calculate_composite_score(self, results: Dict[str, Dict]) -> float:
        """Calculate weighted composite score across all timeframes"""
        try:
            total_weighted_score = 0
            total_weight = 0
            
            for tf, result in results.items():
                if tf in self.weights and 'score' in result:
                    weight = self.weights[tf]
                    score = result['score']
                    total_weighted_score += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                composite_score = total_weighted_score / total_weight
                return round(composite_score, 2)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return 0.0
    
    def _analyze_trend_alignment(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze trend alignment across timeframes"""
        try:
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            
            for tf, result in results.items():
                trend = result.get('trend', {})
                score = trend.get('score', 0)
                
                if score > 2:
                    bullish_count += 1
                elif score < -2:
                    bearish_count += 1
                else:
                    neutral_count += 1
            
            total_timeframes = len(results)
            
            if bullish_count >= total_timeframes * 0.7:
                alignment = 'strong_bullish'
                alignment_score = 10
            elif bullish_count >= total_timeframes * 0.5:
                alignment = 'bullish'
                alignment_score = 7
            elif bearish_count >= total_timeframes * 0.7:
                alignment = 'strong_bearish'
                alignment_score = -10
            elif bearish_count >= total_timeframes * 0.5:
                alignment = 'bearish'
                alignment_score = -7
            else:
                alignment = 'mixed'
                alignment_score = 0
            
            return {
                'alignment': alignment,
                'score': alignment_score,
                'bullish_timeframes': bullish_count,
                'bearish_timeframes': bearish_count,
                'neutral_timeframes': neutral_count,
                'total_timeframes': total_timeframes
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend alignment: {e}")
            return {'alignment': 'unknown', 'score': 0}
    
    def _analyze_momentum_confluence(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze momentum confluence across timeframes"""
        try:
            momentum_scores = []
            
            for tf, result in results.items():
                momentum = result.get('momentum', {})
                score = momentum.get('total_score', 0)
                momentum_scores.append(score)
            
            if not momentum_scores:
                return {'confluence': 'none', 'score': 0}
            
            avg_momentum = sum(momentum_scores) / len(momentum_scores)
            
            # Check for confluence (most timeframes agreeing)
            positive_count = sum(1 for score in momentum_scores if score > 2)
            negative_count = sum(1 for score in momentum_scores if score < -2)
            
            total_timeframes = len(momentum_scores)
            
            if positive_count >= total_timeframes * 0.6:
                confluence = 'bullish'
                confluence_score = 8
            elif negative_count >= total_timeframes * 0.6:
                confluence = 'bearish'
                confluence_score = -8
            else:
                confluence = 'mixed'
                confluence_score = 0
            
            return {
                'confluence': confluence,
                'score': confluence_score,
                'average_momentum': round(avg_momentum, 2),
                'positive_timeframes': positive_count,
                'negative_timeframes': negative_count
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum confluence: {e}")
            return {'confluence': 'unknown', 'score': 0}
    
    def _determine_signal_strength(self, composite_score: float, trend_alignment: Dict) -> str:
        """Determine overall signal strength"""
        try:
            alignment_score = trend_alignment.get('score', 0)
            
            # Combine composite score and alignment
            total_strength = abs(composite_score) + abs(alignment_score) * 0.5
            
            if total_strength >= 12:
                return 'very_strong'
            elif total_strength >= 8:
                return 'strong'
            elif total_strength >= 5:
                return 'moderate'
            elif total_strength >= 2:
                return 'weak'
            else:
                return 'very_weak'
                
        except Exception as e:
            logger.error(f"Error determining signal strength: {e}")
            return 'unknown'
    
    def _get_recommended_action(self, composite_score: float, trend_alignment: Dict) -> str:
        """Get recommended trading action"""
        try:
            alignment = trend_alignment.get('alignment', 'mixed')
            
            if composite_score >= 8 and 'bullish' in alignment:
                return 'STRONG_BUY'
            elif composite_score >= 5 and 'bullish' in alignment:
                return 'BUY'
            elif composite_score <= -8 and 'bearish' in alignment:
                return 'STRONG_SELL'
            elif composite_score <= -5 and 'bearish' in alignment:
                return 'SELL'
            elif abs(composite_score) < 3:
                return 'HOLD'
            else:
                return 'WAIT'
                
        except Exception as e:
            logger.error(f"Error getting recommended action: {e}")
            return 'WAIT'
    
    def _empty_timeframe_result(self) -> Dict[str, Any]:
        """Return empty timeframe result"""
        return {
            'indicators': {},
            'trend': {'direction': 'unknown', 'score': 0, 'strength': 0},
            'momentum': {'total_score': 0},
            'support_resistance': {},
            'volume': {'score': 0},
            'score': 0.0
        }
    
    def _empty_multi_timeframe_result(self, symbol: str) -> Dict[str, Any]:
        """Return empty multi-timeframe result"""
        return {
            'symbol': symbol,
            'timeframe_results': {},
            'composite_score': 0.0,
            'trend_alignment': {'alignment': 'unknown', 'score': 0},
            'momentum_confluence': {'confluence': 'unknown', 'score': 0},
            'signal_strength': 'unknown',
            'recommended_action': 'WAIT',
            'analysis_timestamp': datetime.now().isoformat()
        }
