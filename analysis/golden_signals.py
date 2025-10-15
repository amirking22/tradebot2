"""
Golden Signals Detection System
سیستم تشخیص سیگنال‌های طلایی برای شناسایی بهترین فرصت‌های معاملاتی
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from .technical_indicators import TechnicalIndicators
from .advanced_indicators import AdvancedIndicators
from .enhanced_divergence import EnhancedDivergenceDetector
from .multi_timeframe import MultiTimeframeAnalyzer
from .volume_profile import VolumeProfileAnalyzer
from .signal_rating import SignalRatingSystem
from .trading_details import TradingDetailsCalculator
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GoldenSignalsDetector:
    """کلاس تشخیص سیگنال‌های طلایی"""
    
    def __init__(self, aggregator=None):
        self.min_score = 90  # حداقل امتیاز برای سیگنال طلایی (افزایش یافته)
        self.active_golden_signals = {}
        self.advanced_analysis_enabled = True
        
        # Initialize advanced analyzers
        self.divergence_detector = EnhancedDivergenceDetector()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.rating_system = SignalRatingSystem()
        self.trading_calculator = TradingDetailsCalculator()
        if aggregator:
            self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(aggregator)
        else:
            self.multi_timeframe_analyzer = None
        
    def calculate_signal_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """محاسبه امتیاز سیگنال بر اساس اندیکاتورهای مختلف"""
        try:
            if not data or 'close' not in data:
                return {'score': 0, 'details': 'داده‌های ناکافی'}
            
            close = pd.Series(data['close'])
            high = pd.Series(data['high'])
            low = pd.Series(data['low'])
            volume = pd.Series(data['volume'])
            
            if len(close) < 200:  # حداقل داده برای محاسبه EMA200
                return {'score': 0, 'details': 'داده‌های تاریخی ناکافی'}
            
            # محاسبه اندیکاتورها
            ema9 = TechnicalIndicators.calculate_ema(close, 9)
            ema21 = TechnicalIndicators.calculate_ema(close, 21)
            ema50 = TechnicalIndicators.calculate_ema(close, 50)
            ema200 = TechnicalIndicators.calculate_ema(close, 200)
            
            macd_data = TechnicalIndicators.calculate_macd(close)
            rsi = TechnicalIndicators.calculate_rsi(close)
            stoch_rsi = TechnicalIndicators.calculate_stochastic_rsi(close)
            bollinger = TechnicalIndicators.calculate_bollinger_bands(close)
            atr = TechnicalIndicators.calculate_atr(high, low, close)
            adx = TechnicalIndicators.calculate_adx(high, low, close)
            obv = TechnicalIndicators.calculate_obv(close, volume)
            vwap = TechnicalIndicators.calculate_vwap(high, low, close, volume)
            
            current_price = close.iloc[-1]
            score = 0
            details = []
            
            # 1. تحلیل روند EMA (25 امتیاز)
            ema_score = self._analyze_ema_trend(current_price, ema9.iloc[-1], 
                                              ema21.iloc[-1], ema50.iloc[-1], ema200.iloc[-1])
            score += ema_score['score']
            details.append(f"روند EMA: {ema_score['description']} ({ema_score['score']}/25)")
            
            # 2. تحلیل MACD (20 امتیاز)
            macd_score = self._analyze_macd(macd_data['macd'].iloc[-1], 
                                          macd_data['signal'].iloc[-1], 
                                          macd_data['histogram'].iloc[-1])
            score += macd_score['score']
            details.append(f"MACD: {macd_score['description']} ({macd_score['score']}/20)")
            
            # 3. تحلیل RSI و Stochastic RSI (20 امتیاز)
            rsi_score = self._analyze_rsi_momentum(rsi, stoch_rsi['k'].iloc[-1], stoch_rsi['d'].iloc[-1])
            score += rsi_score['score']
            details.append(f"مومنتوم RSI: {rsi_score['description']} ({rsi_score['score']}/20)")
            
            # 4. تحلیل Bollinger Bands (15 امتیاز)
            bb_score = self._analyze_bollinger_bands(current_price, bollinger['upper'].iloc[-1], 
                                                   bollinger['middle'].iloc[-1], bollinger['lower'].iloc[-1])
            score += bb_score['score']
            details.append(f"Bollinger Bands: {bb_score['description']} ({bb_score['score']}/15)")
            
            # 5. تحلیل حجم و VWAP (10 امتیاز)
            volume_score = self._analyze_volume_vwap(current_price, vwap.iloc[-1], 
                                                   volume.iloc[-1], volume.rolling(20).mean().iloc[-1])
            score += volume_score['score']
            details.append(f"حجم و VWAP: {volume_score['description']} ({volume_score['score']}/10)")
            
            # 6. تحلیل قدرت روند ADX (10 امتیاز)
            adx_score = self._analyze_adx_strength(adx)
            score += adx_score['score']
            details.append(f"قدرت روند ADX: {adx_score['description']} ({adx_score['score']}/10)")
            
            # Advanced analysis integration
            advanced_score = 0
            advanced_details = []
            
            if self.advanced_analysis_enabled:
                # Enhanced divergence detection
                try:
                    df = pd.DataFrame(data)
                    divergence_result = self.divergence_detector.detect_all_divergences(
                        close, pd.Series([rsi] * len(close)), lookback=20
                    )
                    
                    if divergence_result['primary_divergence']:
                        div = divergence_result['primary_divergence']
                        if div['type'] in ['regular_bullish', 'regular_bearish']:
                            advanced_score += div['strength'] * 2
                            advanced_details.append(f"Strong {div['type']} divergence")
                        else:
                            advanced_score += div['strength']
                            advanced_details.append(f"Hidden {div['type']} divergence")
                except Exception as e:
                    logger.error(f"Error in divergence detection: {e}")
                
                # Volume profile analysis
                try:
                    df = pd.DataFrame(data)
                    volume_result = self.volume_analyzer.analyze_volume_profile(df)
                    market_structure_score = volume_result.get('market_structure_score', 0)
                    advanced_score += market_structure_score * 2
                    
                    if market_structure_score > 7:
                        advanced_details.append("Strong market structure")
                    elif market_structure_score > 4:
                        advanced_details.append("Moderate market structure")
                        
                    # VWAP analysis enhancement
                    vwap_analysis = volume_result.get('vwap_analysis', {})
                    vwap_position = vwap_analysis.get('price_position', 'unknown')
                    if vwap_position in ['above_vwap', 'above_upper_band']:
                        advanced_score += 3
                        advanced_details.append("Price above VWAP")
                    elif vwap_position in ['below_vwap', 'below_lower_band']:
                        advanced_score += 3
                        advanced_details.append("Price below VWAP")
                        
                except Exception as e:
                    logger.error(f"Error in volume analysis: {e}")
            
            # Combine base score with advanced analysis
            total_score = score + advanced_score
            if advanced_details:
                details.extend(advanced_details)
            
            # Calculate rating and enhanced details
            signal_result = {
                'score': total_score,
                'base_score': score,
                'advanced_score': advanced_score,
                'max_score': 120,  # Increased to account for advanced features
                'percentage': (total_score / 120) * 100,
                'details': details,
                'is_golden': total_score >= self.min_score,
                'indicators': {
                    'current_price': current_price,
                    'ema9': ema9.iloc[-1],
                    'ema21': ema21.iloc[-1],
                    'ema50': ema50.iloc[-1],
                    'ema200': ema200.iloc[-1],
                    'rsi': rsi,
                    'macd': macd_data['macd'].iloc[-1],
                    'macd_signal': macd_data['signal'].iloc[-1],
                    'atr': atr,
                    'adx': adx,
                    'vwap': vwap.iloc[-1]
                }
            }
            
            # Add rating and trading details if golden signal
            if signal_result['is_golden']:
                rating_data = self.rating_system.calculate_signal_rating(signal_result)
                signal_result['rating_data'] = rating_data
                signal_result['diamond_signal'] = rating_data['is_diamond']
                signal_result['signal_rating'] = rating_data['rating']
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Error calculating signal score: {e}")
            return {'score': 0, 'details': f'خطا: {str(e)}'}
    
    def get_trading_details(self, symbol: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """محاسبه جزئیات معاملاتی برای سیگنال"""
        try:
            if not signal_data.get('is_golden', False):
                return {}
            
            rating_data = signal_data.get('rating_data', {})
            if not rating_data:
                rating_data = self.rating_system.calculate_signal_rating(signal_data)
            
            trading_details = self.trading_calculator.calculate_trading_details(
                symbol, signal_data, rating_data
            )
            
            return trading_details
            
        except Exception as e:
            logger.error(f"Error getting trading details for {symbol}: {e}")
            return {}
    
    def format_enhanced_signal_message(self, symbol: str, signal_data: Dict[str, Any]) -> str:
        """فرمت پیام سیگنال با امتیاز و علامت الماس"""
        try:
            if not signal_data.get('is_golden', False):
                return f"{symbol} - سیگنال عادی"
            
            rating_data = signal_data.get('rating_data', {})
            if not rating_data:
                rating_data = self.rating_system.calculate_signal_rating(signal_data)
            
            return self.rating_system.format_signal_message(symbol, signal_data, rating_data)
            
        except Exception as e:
            logger.error(f"Error formatting enhanced signal message: {e}")
            return f"{symbol} - خطا در فرمت پیام"
    
    def get_top_golden_signals(self, all_signals: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """انتخاب بهترین سیگنال‌های طلایی"""
        try:
            golden_signals = [s for s in all_signals if s.get('is_golden', False)]
            return self.rating_system.get_top_signals(golden_signals, limit)
        except Exception as e:
            logger.error(f"Error getting top golden signals: {e}")
            return all_signals[:limit] if all_signals else []
    
    def _analyze_ema_trend(self, price: float, ema9: float, ema21: float, ema50: float, ema200: float) -> Dict[str, Any]:
        """تحلیل روند بر اساس EMA"""
        score = 0
        
        # بررسی ترتیب EMA برای روند صعودی
        if ema9 > ema21 > ema50 > ema200 and price > ema9:
            score = 25
            description = "روند قوی صعودی - همه EMA منظم"
        elif ema9 > ema21 > ema50 and price > ema9:
            score = 20
            description = "روند متوسط صعودی"
        elif ema9 > ema21 and price > ema200:
            score = 15
            description = "روند ضعیف صعودی"
        elif price > ema200:
            score = 10
            description = "بالای EMA200 - روند بلندمدت مثبت"
        # بررسی ترتیب EMA برای روند نزولی
        elif ema9 < ema21 < ema50 < ema200 and price < ema9:
            score = 25
            description = "روند قوی نزولی - همه EMA منظم"
        elif ema9 < ema21 < ema50 and price < ema9:
            score = 20
            description = "روند متوسط نزولی"
        else:
            score = 5
            description = "روند نامشخص یا خنثی"
        
        return {'score': score, 'description': description}
    
    def _analyze_macd(self, macd: float, signal: float, histogram: float) -> Dict[str, Any]:
        """تحلیل MACD"""
        score = 0
        
        if macd > signal and histogram > 0:
            if macd > 0 and signal > 0:
                score = 20
                description = "MACD مثبت و بالای سیگنال - قوی"
            else:
                score = 15
                description = "MACD بالای سیگنال - متوسط"
        elif macd < signal and histogram < 0:
            if macd < 0 and signal < 0:
                score = 20
                description = "MACD منفی و زیر سیگنال - قوی نزولی"
            else:
                score = 15
                description = "MACD زیر سیگنال - متوسط نزولی"
        elif abs(histogram) < 0.001:  # نزدیک به کراس
            score = 10
            description = "MACD نزدیک به کراس"
        else:
            score = 5
            description = "MACD خنثی"
        
        return {'score': score, 'description': description}
    
    def _analyze_rsi_momentum(self, rsi: float, stoch_k: float, stoch_d: float) -> Dict[str, Any]:
        """تحلیل RSI و Stochastic RSI"""
        score = 0
        
        # RSI تحلیل
        rsi_desc = "RSI normal"
        if 30 < rsi < 70:
            if 40 <= rsi <= 60:
                score += 10  # محدوده خنثی
                rsi_desc = "RSI neutral"
            elif rsi > 60:
                score += 8  # کمی قوی
                rsi_desc = "RSI strong"
            else:
                score += 8  # کمی ضعیف
                rsi_desc = "RSI weak"
        elif rsi <= 30:
            score += 15  # اشباع فروش - فرصت خرید
            rsi_desc = "RSI oversold"
        elif rsi >= 70:
            score += 15  # اشباع خرید - فرصت فروش
            rsi_desc = "RSI overbought"
        
        # Stochastic RSI
        if stoch_k > stoch_d:
            score += 5
            stoch_desc = "StochRSI bullish"
        else:
            score += 5
            stoch_desc = "StochRSI bearish"
        
        return {'score': score, 'description': f"{rsi_desc}, {stoch_desc}"}
    
    def _analyze_bollinger_bands(self, price: float, upper: float, middle: float, lower: float) -> Dict[str, Any]:
        """تحلیل Bollinger Bands"""
        score = 0
        
        band_width = (upper - lower) / middle * 100
        
        if price <= lower:
            score = 15
            description = "قیمت در باند پایین - فرصت خرید"
        elif price >= upper:
            score = 15
            description = "قیمت در باند بالا - فرصت فروش"
        elif abs(price - middle) / middle < 0.01:  # نزدیک به میانگین
            score = 10
            description = "قیمت نزدیک میانگین"
        elif band_width < 5:  # فشردگی باندها
            score = 12
            description = "فشردگی باندها - آماده شکست"
        else:
            score = 8
            description = "قیمت در محدوده عادی"
        
        return {'score': score, 'description': description}
    
    def _analyze_volume_vwap(self, price: float, vwap: float, current_volume: float, avg_volume: float) -> Dict[str, Any]:
        """تحلیل حجم و VWAP"""
        score = 0
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # تحلیل VWAP
        if price > vwap:
            score += 3
            vwap_desc = "بالای VWAP"
        else:
            score += 3
            vwap_desc = "زیر VWAP"
        
        # تحلیل حجم
        if volume_ratio > 2:
            score += 7
            volume_desc = "حجم بالا"
        elif volume_ratio > 1.5:
            score += 5
            volume_desc = "حجم متوسط"
        else:
            score += 2
            volume_desc = "حجم پایین"
        
        return {'score': score, 'description': f"{vwap_desc}, {volume_desc}"}
    
    def _analyze_adx_strength(self, adx: float) -> Dict[str, Any]:
        """تحلیل قدرت روند با ADX"""
        if adx > 50:
            score = 10
            description = "روند بسیار قوی"
        elif adx > 25:
            score = 8
            description = "روند قوی"
        elif adx > 20:
            score = 5
            description = "روند متوسط"
        else:
            score = 2
            description = "روند ضعیف"
        
        return {'score': score, 'description': description}
    
    def is_golden_signal(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """بررسی اینکه آیا سیگنال طلایی است یا نه"""
        try:
            signal_analysis = self.calculate_signal_score(market_data)
            
            if signal_analysis.get('is_golden', False):
                # تعیین نوع سیگنال (خرید/فروش)
                indicators = signal_analysis['indicators']
                
                if (indicators['current_price'] > indicators['ema9'] > indicators['ema21'] and
                    indicators['rsi'] < 70 and indicators['macd'] > indicators['macd_signal']):
                    signal_type = "LONG"
                    signal_strength = "قوی"
                elif (indicators['current_price'] < indicators['ema9'] < indicators['ema21'] and
                      indicators['rsi'] > 30 and indicators['macd'] < indicators['macd_signal']):
                    signal_type = "SHORT"
                    signal_strength = "قوی"
                else:
                    signal_type = "NEUTRAL"
                    signal_strength = "متوسط"
                
                return {
                    'is_golden': True,
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'strength': signal_strength,
                    'score': signal_analysis['score'],
                    'percentage': signal_analysis['percentage'],
                    'details': signal_analysis['details'],
                    'indicators': indicators,
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'is_golden': False,
                'symbol': symbol,
                'score': signal_analysis.get('score', 0),
                'percentage': signal_analysis.get('percentage', 0),
                'reason': 'امتیاز کافی نیست',
                'details': signal_analysis.get('details', [])
            }
            
        except Exception as e:
            logger.error(f"خطا در بررسی سیگنال طلایی برای {symbol}: {e}")
            return {
                'is_golden': False,
                'symbol': symbol,
                'error': str(e)
            }
    
    async def scan_for_golden_signals(self, symbols: List[str], get_market_data_func) -> List[Dict[str, Any]]:
        """اسکن ارزها برای یافتن سیگنال‌های طلایی"""
        golden_signals = []
        
        for symbol in symbols:
            try:
                # دریافت داده‌های بازار
                market_data = await get_market_data_func(symbol)
                if not market_data:
                    continue
                
                # بررسی سیگنال طلایی
                signal_result = self.is_golden_signal(symbol, market_data)
                
                if signal_result['is_golden']:
                    golden_signals.append(signal_result)
                    logger.info(f"سیگنال طلایی یافت شد: {symbol} - امتیاز: {signal_result['score']}")
                
            except Exception as e:
                logger.error(f"خطا در اسکن {symbol}: {e}")
                continue
        
        # مرتب‌سازی بر اساس امتیاز
        golden_signals.sort(key=lambda x: x['score'], reverse=True)
        return golden_signals
