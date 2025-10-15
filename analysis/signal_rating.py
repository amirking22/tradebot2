"""
Signal Rating and Diamond Classification System
سیستم امتیازدهی و طبقه‌بندی سیگنال‌های طلایی با علامت الماس
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SignalRatingSystem:
    """سیستم امتیازدهی و طبقه‌بندی سیگنال‌های طلایی"""
    
    def __init__(self):
        self.diamond_threshold = 110  # حداقل امتیاز برای علامت الماس
        self.rating_ranges = {
            10: (115, float('inf')),  # فوق‌العاده (الماس)
            9: (110, 115),           # عالی
            8: (105, 110),           # خیلی خوب
            7: (100, 105),           # خوب
            6: (95, 100),            # متوسط به بالا
            5: (90, 95),             # متوسط
            4: (85, 90),             # کمتر از متوسط
            3: (80, 85),             # ضعیف
            2: (75, 80),             # خیلی ضعیف
            1: (0, 75)               # بسیار ضعیف
        }
    
    def calculate_signal_rating(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """محاسبه امتیاز 1 تا 10 برای سیگنال"""
        try:
            total_score = signal_data.get('score', 0)
            base_score = signal_data.get('base_score', 0)
            advanced_score = signal_data.get('advanced_score', 0)
            
            # محاسبه امتیاز نهایی با در نظر گیری عوامل اضافی
            bonus_factors = self._calculate_bonus_factors(signal_data)
            final_score = total_score + bonus_factors
            
            # تعیین رتبه از 1 تا 10
            rating = self._score_to_rating(final_score)
            
            # تشخیص سیگنال الماسی
            is_diamond = final_score >= self.diamond_threshold
            
            # محاسبه احتمال موفقیت
            success_probability = self._calculate_success_probability(final_score, signal_data)
            
            # تعیین کیفیت سیگنال
            quality_level = self._determine_quality_level(rating, is_diamond)
            
            return {
                'rating': rating,
                'final_score': round(final_score, 2),
                'base_score': base_score,
                'advanced_score': advanced_score,
                'bonus_factors': bonus_factors,
                'is_diamond': is_diamond,
                'diamond_icon': '💎' if is_diamond else '',
                'success_probability': success_probability,
                'quality_level': quality_level,
                'confidence_level': self._get_confidence_level(rating),
                'risk_level': self._get_risk_level(rating),
                'recommended_position_size': self._get_position_size_recommendation(rating)
            }
            
        except Exception as e:
            logger.error(f"Error calculating signal rating: {e}")
            return self._default_rating()
    
    def _calculate_bonus_factors(self, signal_data: Dict[str, Any]) -> float:
        """محاسبه عوامل امتیاز اضافی"""
        try:
            bonus = 0
            indicators = signal_data.get('indicators', {})
            details = signal_data.get('details', [])
            
            # بونوس برای تأیید چندگانه اندیکاتورها
            strong_signals = sum(1 for detail in details if 'قوی' in str(detail) or 'Strong' in str(detail))
            bonus += strong_signals * 2
            
            # بونوس برای حجم بالا
            if 'حجم بالا' in str(details) or 'High volume' in str(details):
                bonus += 3
            
            # بونوس برای واگرایی قوی
            if 'Strong' in str(details) and 'divergence' in str(details):
                bonus += 5
            
            # بونوس برای تأیید چند تایم‌فریم
            if 'timeframe' in str(details).lower() or 'تایم' in str(details):
                bonus += 4
            
            # بونوس برای ساختار بازار قوی
            if 'Strong market structure' in str(details):
                bonus += 3
            
            return min(bonus, 15)  # حداکثر 15 امتیاز بونوس
            
        except Exception as e:
            logger.error(f"Error calculating bonus factors: {e}")
            return 0
    
    def _score_to_rating(self, score: float) -> int:
        """تبدیل امتیاز به رتبه 1 تا 10"""
        for rating, (min_score, max_score) in self.rating_ranges.items():
            if min_score <= score < max_score:
                return rating
        return 1
    
    def _calculate_success_probability(self, final_score: float, signal_data: Dict[str, Any]) -> int:
        """محاسبه احتمال موفقیت به درصد"""
        try:
            # فرمول محاسبه احتمال بر اساس امتیاز
            base_probability = min(95, max(45, (final_score - 70) * 1.2))
            
            # تنظیم بر اساس عوامل اضافی
            if final_score >= self.diamond_threshold:
                base_probability = min(98, base_probability + 10)
            
            # در نظر گیری کیفیت اندیکاتورها
            indicators = signal_data.get('indicators', {})
            if indicators.get('rsi', 50) < 30 or indicators.get('rsi', 50) > 70:
                base_probability += 5
            
            return min(99, max(50, int(base_probability)))
            
        except Exception as e:
            logger.error(f"Error calculating success probability: {e}")
            return 70
    
    def _determine_quality_level(self, rating: int, is_diamond: bool) -> str:
        """تعیین سطح کیفیت سیگنال"""
        if is_diamond:
            return "💎 الماسی - فوق‌العاده"
        elif rating >= 9:
            return "🥇 عالی"
        elif rating >= 7:
            return "🥈 خوب"
        elif rating >= 5:
            return "🥉 متوسط"
        else:
            return "⚠️ ضعیف"
    
    def _get_confidence_level(self, rating: int) -> str:
        """تعیین سطح اطمینان"""
        if rating >= 9:
            return "بسیار بالا"
        elif rating >= 7:
            return "بالا"
        elif rating >= 5:
            return "متوسط"
        elif rating >= 3:
            return "پایین"
        else:
            return "بسیار پایین"
    
    def _get_risk_level(self, rating: int) -> str:
        """تعیین سطح ریسک"""
        if rating >= 9:
            return "کم"
        elif rating >= 7:
            return "متوسط"
        elif rating >= 5:
            return "متوسط به بالا"
        elif rating >= 3:
            return "بالا"
        else:
            return "بسیار بالا"
    
    def _get_position_size_recommendation(self, rating: int) -> str:
        """پیشنهاد اندازه پوزیشن"""
        if rating >= 9:
            return "2-3% از کل سرمایه"
        elif rating >= 7:
            return "1.5-2% از کل سرمایه"
        elif rating >= 5:
            return "1-1.5% از کل سرمایه"
        elif rating >= 3:
            return "0.5-1% از کل سرمایه"
        else:
            return "حداکثر 0.5% از کل سرمایه"
    
    def _default_rating(self) -> Dict[str, Any]:
        """امتیاز پیش‌فرض در صورت خطا"""
        return {
            'rating': 1,
            'final_score': 0,
            'base_score': 0,
            'advanced_score': 0,
            'bonus_factors': 0,
            'is_diamond': False,
            'diamond_icon': '',
            'success_probability': 50,
            'quality_level': "⚠️ نامشخص",
            'confidence_level': "نامشخص",
            'risk_level': "نامشخص",
            'recommended_position_size': "حداکثر 0.5% از کل سرمایه"
        }
    
    def format_signal_message(self, symbol: str, signal_data: Dict[str, Any], 
                            rating_data: Dict[str, Any]) -> str:
        """فرمت پیام سیگنال با امتیاز و علامت‌ها"""
        try:
            diamond_icon = rating_data.get('diamond_icon', '')
            rating = rating_data.get('rating', 1)
            quality_level = rating_data.get('quality_level', '')
            success_probability = rating_data.get('success_probability', 50)
            
            # ایجاد ستاره‌ها بر اساس امتیاز
            stars = '⭐' * rating
            
            message = f"""
{diamond_icon} {symbol} - امتیاز: {rating}/10 {stars}

🎯 کیفیت: {quality_level}
📊 احتمال موفقیت: {success_probability}%
💰 اندازه پوزیشن پیشنهادی: {rating_data.get('recommended_position_size', 'نامشخص')}

📈 امتیاز کل: {rating_data.get('final_score', 0):.1f}
   • امتیاز پایه: {rating_data.get('base_score', 0)}
   • امتیاز پیشرفته: {rating_data.get('advanced_score', 0)}
   • عوامل اضافی: +{rating_data.get('bonus_factors', 0)}

⚡ سطح اطمینان: {rating_data.get('confidence_level', 'نامشخص')}
⚠️ سطح ریسک: {rating_data.get('risk_level', 'نامشخص')}

قیمت فعلی: {signal_data.get('indicators', {}).get('current_price', 'نامشخص')}
"""
            
            # اضافه کردن جزئیات اندیکاتورها
            details = signal_data.get('details', [])
            if details:
                message += "\n📋 تحلیل تکنیکال:\n"
                for detail in details[:5]:  # نمایش 5 مورد اول
                    message += f"• {detail}\n"
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"Error formatting signal message: {e}")
            return f"{symbol} - خطا در فرمت پیام"
    
    def get_top_signals(self, signals: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """انتخاب بهترین سیگنال‌ها بر اساس امتیاز"""
        try:
            # امتیازدهی همه سیگنال‌ها
            rated_signals = []
            for signal in signals:
                rating_data = self.calculate_signal_rating(signal)
                signal_with_rating = signal.copy()
                signal_with_rating['rating_data'] = rating_data
                rated_signals.append(signal_with_rating)
            
            # مرتب‌سازی بر اساس امتیاز نهایی
            sorted_signals = sorted(rated_signals, 
                                  key=lambda x: x['rating_data']['final_score'], 
                                  reverse=True)
            
            return sorted_signals[:limit]
            
        except Exception as e:
            logger.error(f"Error getting top signals: {e}")
            return signals[:limit] if signals else []
