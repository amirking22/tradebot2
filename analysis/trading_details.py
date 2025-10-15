"""
Trading Details Calculator
محاسبه جزئیات معاملاتی شامل نقاط ورود، پله‌ها، سود و ضرر
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingDetailsCalculator:
    """محاسبه‌گر جزئیات معاملاتی پیشرفته"""
    
    def __init__(self):
        self.risk_reward_ratios = {
            10: 4.0,  # الماسی - نسبت ریسک به ریوارد 1:4
            9: 3.5,   # عالی - نسبت 1:3.5
            8: 3.0,   # خیلی خوب - نسبت 1:3
            7: 2.5,   # خوب - نسبت 1:2.5
            6: 2.0,   # متوسط به بالا - نسبت 1:2
            5: 1.8,   # متوسط - نسبت 1:1.8
            4: 1.5,   # کمتر از متوسط - نسبت 1:1.5
            3: 1.2,   # ضعیف - نسبت 1:1.2
            2: 1.0,   # خیلی ضعیف - نسبت 1:1
            1: 0.8    # بسیار ضعیف - نسبت 1:0.8
        }
    
    def calculate_trading_details(self, signal_data: Dict[str, Any], 
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """محاسبه جزئیات کامل معاملاتی"""
        try:
            # Get rating data from signal_data if available
            rating_data = signal_data.get('rating_data', {})
            current_price = signal_data.get('indicators', {}).get('current_price', 0)
            rating = rating_data.get('rating', 5)
            is_diamond = rating_data.get('is_diamond', False)
            
            # If no current price in signal_data, try to get from market_data
            if current_price <= 0 and market_data:
                close_prices = market_data.get('close', [])
                if close_prices:
                    current_price = close_prices[-1]
            
            if current_price <= 0:
                return self._default_trading_details()
            
            # تشخیص نوع سیگنال (خرید/فروش)
            signal_type = self._determine_signal_type(signal_data)
            
            # محاسبه نقاط ورود
            entry_points = self._calculate_entry_points(current_price, signal_type, rating)
            
            # محاسبه نقاط سود (Take Profit)
            take_profit_levels = self._calculate_take_profit_levels(
                current_price, signal_type, rating, is_diamond
            )
            
            # محاسبه نقطه ضرر (Stop Loss)
            stop_loss = self._calculate_stop_loss(current_price, signal_type, rating)
            
            # محاسبه پله‌های خرید/فروش
            scaling_levels = self._calculate_scaling_levels(
                current_price, signal_type, rating
            )
            
            # تخمین زمان معامله
            time_estimates = self._estimate_trade_duration(rating, signal_data)
            
            # محاسبه اندازه پوزیشن برای هر پله
            position_sizes = self._calculate_position_sizes(rating, len(entry_points))
            
            symbol = signal_data.get('symbol', 'نامشخص')
            return {
                'symbol': symbol,
                'signal_type': signal_type,
                'current_price': current_price,
                'entry_points': entry_points,
                'take_profit_levels': take_profit_levels,
                'stop_loss': stop_loss,
                'scaling_levels': scaling_levels,
                'position_sizes': position_sizes,
                'time_estimates': time_estimates,
                'risk_reward_ratio': self.risk_reward_ratios.get(rating, 1.5),
                'max_risk_per_trade': self._get_max_risk(rating),
                'recommended_leverage': self._get_recommended_leverage(rating),
                'market_conditions': self._analyze_market_conditions(signal_data),
                'trading_strategy': self._get_trading_strategy(signal_type, rating)
            }
            
        except Exception as e:
            logger.error(f"Error calculating trading details: {e}")
            return self._default_trading_details()
    
    def _determine_signal_type(self, signal_data: Dict[str, Any]) -> str:
        """تشخیص نوع سیگنال (خرید یا فروش)"""
        try:
            details = signal_data.get('details', [])
            indicators = signal_data.get('indicators', {})
            
            bullish_signals = 0
            bearish_signals = 0
            
            # بررسی جزئیات سیگنال
            for detail in details:
                detail_str = str(detail).lower()
                if any(word in detail_str for word in ['صعودی', 'خرید', 'bullish', 'buy', 'long']):
                    bullish_signals += 1
                elif any(word in detail_str for word in ['نزولی', 'فروش', 'bearish', 'sell', 'short']):
                    bearish_signals += 1
            
            # بررسی EMA ترتیب
            current_price = indicators.get('current_price', 0)
            ema9 = indicators.get('ema9', 0)
            ema21 = indicators.get('ema21', 0)
            
            if current_price > ema9 > ema21:
                bullish_signals += 2
            elif current_price < ema9 < ema21:
                bearish_signals += 2
            
            # بررسی RSI
            rsi = indicators.get('rsi', 50)
            if isinstance(rsi, (list, pd.Series)):
                rsi = rsi[-1] if len(rsi) > 0 else 50
            
            if rsi < 30:
                bullish_signals += 1  # oversold - احتمال برگشت صعودی
            elif rsi > 70:
                bearish_signals += 1  # overbought - احتمال برگشت نزولی
            
            return 'BUY' if bullish_signals > bearish_signals else 'SELL'
            
        except Exception as e:
            logger.error(f"Error determining signal type: {e}")
            return 'BUY'
    
    def _calculate_entry_points(self, current_price: float, signal_type: str, rating: int) -> List[Dict[str, Any]]:
        """محاسبه نقاط ورود متعدد"""
        try:
            entry_points = []
            
            if signal_type == 'BUY':
                # نقاط ورود برای خرید
                entry_points = [
                    {
                        'level': 1,
                        'price': round(current_price * 0.998, 6),  # 0.2% پایین‌تر
                        'percentage': 40,  # 40% از کل پوزیشن
                        'description': 'ورود اصلی - فوری'
                    },
                    {
                        'level': 2,
                        'price': round(current_price * 0.995, 6),  # 0.5% پایین‌تر
                        'percentage': 35,  # 35% از کل پوزیشن
                        'description': 'ورود دوم - در صورت ریزش'
                    },
                    {
                        'level': 3,
                        'price': round(current_price * 0.990, 6),  # 1% پایین‌تر
                        'percentage': 25,  # 25% از کل پوزیشن
                        'description': 'ورود سوم - فرصت طلایی'
                    }
                ]
            else:  # SELL
                # نقاط ورود برای فروش
                entry_points = [
                    {
                        'level': 1,
                        'price': round(current_price * 1.002, 6),  # 0.2% بالاتر
                        'percentage': 40,
                        'description': 'ورود اصلی - فوری'
                    },
                    {
                        'level': 2,
                        'price': round(current_price * 1.005, 6),  # 0.5% بالاتر
                        'percentage': 35,
                        'description': 'ورود دوم - در صورت پامپ'
                    },
                    {
                        'level': 3,
                        'price': round(current_price * 1.010, 6),  # 1% بالاتر
                        'percentage': 25,
                        'description': 'ورود سوم - فرصت طلایی'
                    }
                ]
            
            # تنظیم بر اساس امتیاز
            if rating >= 9:  # سیگنال‌های قوی - نقاط ورود نزدیک‌تر
                for entry in entry_points:
                    if signal_type == 'BUY':
                        entry['price'] = round(entry['price'] * 1.001, 6)  # نزدیک‌تر به قیمت فعلی
                    else:
                        entry['price'] = round(entry['price'] * 0.999, 6)
            
            return entry_points
            
        except Exception as e:
            logger.error(f"Error calculating entry points: {e}")
            return []
    
    def _calculate_take_profit_levels(self, current_price: float, signal_type: str, 
                                    rating: int, is_diamond: bool) -> List[Dict[str, Any]]:
        """محاسبه سطوح سود"""
        try:
            risk_reward = self.risk_reward_ratios.get(rating, 1.5)
            take_profits = []
            
            if signal_type == 'BUY':
                # سطوح سود برای خرید
                tp_levels = [
                    (1.5, 30),  # 1.5% سود - 30% از پوزیشن
                    (3.0, 40),  # 3% سود - 40% از پوزیشن
                    (5.0, 20),  # 5% سود - 20% از پوزیشن
                    (8.0, 10)   # 8% سود - 10% از پوزیشن (برای الماسی‌ها)
                ]
                
                for i, (profit_pct, position_pct) in enumerate(tp_levels):
                    if i == 3 and not is_diamond:  # سطح چهارم فقط برای الماسی‌ها
                        continue
                        
                    adjusted_profit = profit_pct * (risk_reward / 2.0)  # تنظیم بر اساس نسبت ریسک
                    take_profits.append({
                        'level': i + 1,
                        'price': round(current_price * (1 + adjusted_profit / 100), 6),
                        'percentage': position_pct,
                        'profit_pct': round(adjusted_profit, 2),
                        'description': f'سود {i+1} - {adjusted_profit:.1f}%'
                    })
            
            else:  # SELL
                # سطوح سود برای فروش
                tp_levels = [
                    (1.5, 30),
                    (3.0, 40),
                    (5.0, 20),
                    (8.0, 10)
                ]
                
                for i, (profit_pct, position_pct) in enumerate(tp_levels):
                    if i == 3 and not is_diamond:
                        continue
                        
                    adjusted_profit = profit_pct * (risk_reward / 2.0)
                    take_profits.append({
                        'level': i + 1,
                        'price': round(current_price * (1 - adjusted_profit / 100), 6),
                        'percentage': position_pct,
                        'profit_pct': round(adjusted_profit, 2),
                        'description': f'سود {i+1} - {adjusted_profit:.1f}%'
                    })
            
            return take_profits
            
        except Exception as e:
            logger.error(f"Error calculating take profit levels: {e}")
            return []
    
    def _calculate_stop_loss(self, current_price: float, signal_type: str, rating: int) -> Dict[str, Any]:
        """محاسبه نقطه ضرر"""
        try:
            # درصد ضرر بر اساس امتیاز
            stop_loss_percentages = {
                10: 1.0,  # الماسی - 1% ضرر
                9: 1.2,   # عالی - 1.2% ضرر
                8: 1.5,   # خیلی خوب - 1.5% ضرر
                7: 2.0,   # خوب - 2% ضرر
                6: 2.5,   # متوسط به بالا - 2.5% ضرر
                5: 3.0,   # متوسط - 3% ضرر
                4: 3.5,   # کمتر از متوسط - 3.5% ضرر
                3: 4.0,   # ضعیف - 4% ضرر
                2: 4.5,   # خیلی ضعیف - 4.5% ضرر
                1: 5.0    # بسیار ضعیف - 5% ضرر
            }
            
            stop_loss_pct = stop_loss_percentages.get(rating, 3.0)
            
            if signal_type == 'BUY':
                stop_price = round(current_price * (1 - stop_loss_pct / 100), 6)
            else:  # SELL
                stop_price = round(current_price * (1 + stop_loss_pct / 100), 6)
            
            return {
                'price': stop_price,
                'percentage': stop_loss_pct,
                'description': f'حد ضرر - {stop_loss_pct}%',
                'risk_amount': f'{stop_loss_pct}% از سرمایه در معرض ریسک'
            }
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return {'price': current_price * 0.95, 'percentage': 5.0}
    
    def _calculate_scaling_levels(self, current_price: float, signal_type: str, rating: int) -> List[Dict[str, Any]]:
        """محاسبه پله‌های خرید/فروش"""
        try:
            scaling_levels = []
            
            # تعداد پله‌ها بر اساس امتیاز
            num_levels = 3 if rating >= 7 else 2
            
            for i in range(num_levels):
                if signal_type == 'BUY':
                    # پله‌های خرید (در صورت ریزش بیشتر)
                    level_price = round(current_price * (1 - (i + 2) * 0.01), 6)
                    action = 'خرید اضافی'
                else:
                    # پله‌های فروش (در صورت پامپ بیشتر)
                    level_price = round(current_price * (1 + (i + 2) * 0.01), 6)
                    action = 'فروش اضافی'
                
                scaling_levels.append({
                    'level': i + 1,
                    'price': level_price,
                    'action': action,
                    'percentage': 15 - (i * 5),  # 15%, 10%, 5%
                    'description': f'پله {i+1} - {action} در صورت ادامه روند'
                })
            
            return scaling_levels
            
        except Exception as e:
            logger.error(f"Error calculating scaling levels: {e}")
            return []
    
    def _estimate_trade_duration(self, rating: int, signal_data: Dict[str, Any]) -> Dict[str, str]:
        """تخمین مدت زمان معامله"""
        try:
            # تخمین زمان بر اساس امتیاز و قدرت سیگنال
            if rating >= 9:
                return {
                    'short_term': '2-6 ساعت',
                    'medium_term': '1-3 روز',
                    'long_term': '1-2 هفته',
                    'recommended': 'کوتاه مدت (2-6 ساعت)'
                }
            elif rating >= 7:
                return {
                    'short_term': '4-12 ساعت',
                    'medium_term': '2-5 روز',
                    'long_term': '1-3 هفته',
                    'recommended': 'میان مدت (2-5 روز)'
                }
            elif rating >= 5:
                return {
                    'short_term': '6-24 ساعت',
                    'medium_term': '3-7 روز',
                    'long_term': '2-4 هفته',
                    'recommended': 'میان مدت (3-7 روز)'
                }
            else:
                return {
                    'short_term': '12-48 ساعت',
                    'medium_term': '5-10 روز',
                    'long_term': '3-6 هفته',
                    'recommended': 'بلند مدت (5-10 روز)'
                }
                
        except Exception as e:
            logger.error(f"Error estimating trade duration: {e}")
            return {'recommended': 'نامشخص'}
    
    def _calculate_position_sizes(self, rating: int, num_entries: int) -> Dict[str, str]:
        """محاسبه اندازه پوزیشن برای هر مرحله"""
        try:
            # درصد کل سرمایه بر اساس امتیاز
            total_position_percentages = {
                10: 3.0,  # الماسی - حداکثر 3%
                9: 2.5,   # عالی - حداکثر 2.5%
                8: 2.0,   # خیلی خوب - حداکثر 2%
                7: 1.5,   # خوب - حداکثر 1.5%
                6: 1.2,   # متوسط به بالا - حداکثر 1.2%
                5: 1.0,   # متوسط - حداکثر 1%
                4: 0.8,   # کمتر از متوسط - حداکثر 0.8%
                3: 0.6,   # ضعیف - حداکثر 0.6%
                2: 0.4,   # خیلی ضعیف - حداکثر 0.4%
                1: 0.2    # بسیار ضعیف - حداکثر 0.2%
            }
            
            total_percentage = total_position_percentages.get(rating, 1.0)
            
            return {
                'total_recommended': f'{total_percentage}% از کل سرمایه',
                'per_entry': f'{total_percentage/num_entries:.2f}% برای هر ورود',
                'max_risk': f'{total_percentage * 0.3:.2f}% حداکثر ریسک',
                'leverage_suggestion': self._get_recommended_leverage(rating)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position sizes: {e}")
            return {'total_recommended': '1% از کل سرمایه'}
    
    def _get_max_risk(self, rating: int) -> str:
        """حداکثر ریسک قابل قبول"""
        risk_percentages = {
            10: 0.5, 9: 0.6, 8: 0.8, 7: 1.0, 6: 1.2,
            5: 1.5, 4: 1.8, 3: 2.0, 2: 2.5, 1: 3.0
        }
        return f"{risk_percentages.get(rating, 1.5)}% از کل سرمایه"
    
    def _get_recommended_leverage(self, rating: int) -> str:
        """پیشنهاد اهرم"""
        if rating >= 9:
            return "5-10x (محتاطانه)"
        elif rating >= 7:
            return "3-5x (متعادل)"
        elif rating >= 5:
            return "2-3x (محافظه‌کارانه)"
        else:
            return "1-2x (بسیار محافظه‌کارانه)"
    
    def _analyze_market_conditions(self, signal_data: Dict[str, Any]) -> Dict[str, str]:
        """تحلیل شرایط بازار"""
        try:
            indicators = signal_data.get('indicators', {})
            
            # تحلیل ساده شرایط بازار
            conditions = {
                'trend': 'نامشخص',
                'volatility': 'متوسط',
                'volume': 'نرمال',
                'momentum': 'خنثی'
            }
            
            # تحلیل روند بر اساس EMA
            current_price = indicators.get('current_price', 0)
            ema21 = indicators.get('ema21', 0)
            ema50 = indicators.get('ema50', 0)
            
            if current_price > ema21 > ema50:
                conditions['trend'] = 'صعودی'
            elif current_price < ema21 < ema50:
                conditions['trend'] = 'نزولی'
            else:
                conditions['trend'] = 'خنثی'
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {'trend': 'نامشخص', 'volatility': 'متوسط'}
    
    def _get_trading_strategy(self, signal_type: str, rating: int) -> str:
        """استراتژی معاملاتی پیشنهادی"""
        strategies = {
            'BUY': {
                10: "خرید تدریجی با هدف‌گیری سریع - استراتژی اسکالپینگ",
                9: "خرید قوی با نگهداری کوتاه مدت",
                8: "خرید متعادل با هدف‌گیری مرحله‌ای",
                7: "خرید محتاطانه با مدیریت ریسک دقیق",
                6: "خرید کم با حد ضرر نزدیک",
                5: "خرید آزمایشی با پوزیشن کوچک"
            },
            'SELL': {
                10: "فروش تدریجی با هدف‌گیری سریع - استراتژی اسکالپینگ",
                9: "فروش قوی با نگهداری کوتاه مدت",
                8: "فروش متعادل با هدف‌گیری مرحله‌ای",
                7: "فروش محتاطانه با مدیریت ریسک دقیق",
                6: "فروش کم با حد ضرر نزدیک",
                5: "فروش آزمایشی با پوزیشن کوچک"
            }
        }
        
        return strategies.get(signal_type, {}).get(rating, "استراتژی محافظه‌کارانه")
    
    def _default_trading_details(self) -> Dict[str, Any]:
        """جزئیات پیش‌فرض در صورت خطا"""
        return {
            'symbol': 'نامشخص',
            'signal_type': 'BUY',
            'current_price': 0,
            'entry_points': [],
            'take_profit_levels': [],
            'stop_loss': {'price': 0, 'percentage': 3.0},
            'scaling_levels': [],
            'position_sizes': {'total_recommended': '1% از کل سرمایه'},
            'time_estimates': {'recommended': 'نامشخص'},
            'risk_reward_ratio': 1.5,
            'max_risk_per_trade': '1.5% از کل سرمایه',
            'recommended_leverage': '2-3x محافظه‌کارانه',
            'market_conditions': {'trend': 'نامشخص'},
            'trading_strategy': 'استراتژی محافظه‌کارانه'
        }
    
    def format_detailed_message(self, symbol: str, trading_details: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        """فرمت پیام جزئیات معاملاتی"""
        try:
            # Clean symbol name
            clean_symbol = symbol.replace('/USDT', '')
            signal_type = trading_details.get('signal_type', 'BUY')
            current_price = trading_details.get('current_price', 0)
            
            # Get rating info
            rating_data = signal_result.get('rating_data', {})
            rating = rating_data.get('rating', 5)
            is_diamond = rating_data.get('is_diamond', False)
            diamond_icon = '💎 ' if is_diamond else ''
            
            # Get detailed analysis explanation
            analysis_explanation = self._generate_golden_signal_explanation(signal_result, trading_details)
            
            message = f"""
{diamond_icon}📊 جزئیات معاملاتی {clean_symbol}
{'🟢 سیگنال خرید' if signal_type == 'BUY' else '🔴 سیگنال فروش'} - امتیاز: {rating}/10

💰 قیمت فعلی: {current_price:.6f}

{analysis_explanation}

🎯 نقاط ورود:"""
            
            # نقاط ورود
            for entry in trading_details.get('entry_points', []):
                message += f"\n   {entry['level']}. {entry['price']} ({entry['percentage']}%) - {entry['description']}"
            
            # سطوح سود
            message += "\n\n💎 سطوح سود:"
            for tp in trading_details.get('take_profit_levels', []):
                message += f"\n   TP{tp['level']}: {tp['price']} ({tp['profit_pct']}% سود) - {tp['percentage']}% پوزیشن"
            
            # حد ضرر
            stop_loss = trading_details.get('stop_loss', {})
            message += f"\n\n🛑 حد ضرر: {stop_loss.get('price', 0)} ({stop_loss.get('percentage', 0)}%)"
            
            # پله‌ها
            scaling_levels = trading_details.get('scaling_levels', [])
            if scaling_levels:
                message += "\n\n📈 پله‌های اضافی:"
                for scale in scaling_levels:
                    message += f"\n   {scale['level']}. {scale['price']} - {scale['action']} ({scale['percentage']}%)"
            
            # اطلاعات ریسک و زمان
            position_sizes = trading_details.get('position_sizes', {})
            time_estimates = trading_details.get('time_estimates', {})
            
            message += f"""

⏰ زمان پیشنهادی: {time_estimates.get('recommended', 'نامشخص')}
💼 اندازه پوزیشن: {position_sizes.get('total_recommended', 'نامشخص')}
🎚️ اهرم پیشنهادی: {trading_details.get('recommended_leverage', 'نامشخص')}
⚖️ نسبت ریسک/ریوارد: 1:{trading_details.get('risk_reward_ratio', 1.5)}

📋 استراتژی: {trading_details.get('trading_strategy', 'محافظه‌کارانه')}
"""
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"Error formatting trading details message: {e}")
            return f"خطا در نمایش جزئیات معاملاتی {trading_details.get('symbol', 'نامشخص')}"
    
    def _generate_golden_signal_explanation(self, signal_result: Dict[str, Any], trading_details: Dict[str, Any]) -> str:
        """تولید توضیح دقیق چرا سیگنال طلایی شده"""
        try:
            indicators = signal_result.get('indicators', {})
            rating_data = signal_result.get('rating_data', {})
            rating = rating_data.get('rating', 5)
            is_diamond = rating_data.get('is_diamond', False)
            
            explanation = "🔍 **چرا این سیگنال طلایی است؟**\n"
            
            # Diamond signal explanation
            if is_diamond:
                explanation += "💎 **سیگنال الماسی:** این سیگنال دارای بالاترین کیفیت است\n"
            
            # Technical indicators analysis
            explanation += "\n📈 **تحلیل اندیکاتورها:**\n"
            
            # RSI Analysis
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                explanation += f"• RSI: {rsi:.1f} - منطقه فروش بیش از حد (فرصت خرید قوی)\n"
            elif rsi > 70:
                explanation += f"• RSI: {rsi:.1f} - منطقه خرید بیش از حد (فرصت فروش قوی)\n"
            else:
                explanation += f"• RSI: {rsi:.1f} - در محدوده متعادل\n"
            
            # MACD Analysis
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                explanation += f"• MACD: {macd:.4f} > Signal: {macd_signal:.4f} - روند صعودی\n"
            else:
                explanation += f"• MACD: {macd:.4f} < Signal: {macd_signal:.4f} - روند نزولی\n"
            
            # ADX Analysis
            adx = indicators.get('adx', 0)
            if adx > 50:
                explanation += f"• ADX: {adx:.1f} - روند بسیار قوی\n"
            elif adx > 25:
                explanation += f"• ADX: {adx:.1f} - روند قوی\n"
            else:
                explanation += f"• ADX: {adx:.1f} - روند ضعیف\n"
            
            # EMA Analysis
            current_price = indicators.get('current_price', 0)
            ema21 = indicators.get('ema21', 0)
            ema50 = indicators.get('ema50', 0)
            
            if current_price > ema21 > ema50:
                explanation += f"• قیمت بالای EMA21 و EMA50 - روند صعودی تأیید شده\n"
            elif current_price < ema21 < ema50:
                explanation += f"• قیمت زیر EMA21 و EMA50 - روند نزولی تأیید شده\n"
            
            # Volume Analysis
            volume_ratio = indicators.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                explanation += f"• حجم معاملات {volume_ratio:.1f}x بالاتر از میانگین - تأیید قوی\n"
            
            # Bollinger Bands Analysis
            bb_position = indicators.get('bb_position', 0.5)
            if bb_position < 0.2:
                explanation += "• نزدیک باند پایین بولینگر - احتمال بازگشت قیمت\n"
            elif bb_position > 0.8:
                explanation += "• نزدیک باند بالای بولینگر - احتمال تصحیح قیمت\n"
            
            # Score breakdown
            total_score = signal_result.get('score', 0)
            explanation += f"\n📊 **امتیاز کل:** {total_score}/120 ({(total_score/120*100):.1f}%)\n"
            
            # Why it's the best signal
            explanation += "\n🏆 **چرا بهترین سیگنال:**\n"
            if rating >= 9:
                explanation += "• امتیاز بالای 9/10 - کیفیت استثنایی\n"
                explanation += "• تأیید همزمان چندین اندیکاتور\n"
                explanation += "• نسبت ریسک به ریوارد عالی\n"
            elif rating >= 7:
                explanation += "• امتیاز بالای 7/10 - کیفیت بسیار خوب\n"
                explanation += "• تأیید اکثر اندیکاتورها\n"
            else:
                explanation += "• امتیاز قابل قبول - سیگنال معتبر\n"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating golden signal explanation: {e}")
            return "🔍 **تحلیل دقیق در دسترس نیست**\n"
