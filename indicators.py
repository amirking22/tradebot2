"""Technical indicators for trading signals with enhanced accuracy."""
from __future__ import annotations
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from exchanges import OHLCV

class SignalType(Enum):
    STRONG_BUY = 5
    BUY = 4
    NEUTRAL = 3
    SELL = 2
    STRONG_SELL = 1

@dataclass
class Signal:
    """Trading signal with detailed analysis."""
    symbol: str
    signal_type: SignalType
    entry: float
    stop_loss: float
    take_profit: List[float]  # List of TP levels
    confidence: float  # 0-100%
    indicators: Dict[str, float]  # All indicator values
    timestamp: pd.Timestamp
    
    @property
    def side(self) -> str:
        """Return 'long' or 'short' based on signal type."""
        return 'long' if self.signal_type in [SignalType.STRONG_BUY, SignalType.BUY] else 'short'


def calculate_indicators(ohlcv: OHLCV) -> dict[str, pd.Series]:
    """Calculate comprehensive technical indicators."""
    df = ohlcv.to_dataframe()
    
    # Moving Averages
    df['ema9'] = ta.ema(df['close'], length=9)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)
    
    # Oscillators
    df['rsi'] = ta.rsi(df['close'], length=14)
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    # Volatility
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Trend
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx['ADX_14']
    df['dmp'] = adx['DMP_14']
    df['dmn'] = adx['DMN_14']
    
    # Volume
    df['volume_ma'] = ta.sma(df['volume'], length=20)
    df['obv'] = ta.obv(df['close'], df['volume'])
    
    # Momentum
    df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
    df['macd_signal'] = ta.macd(df['close'])['MACDs_12_26_9']
    
    # Support/Resistance
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Ichimoku Cloud
    ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
    df['ichimoku_a'] = ichimoku['ITS_9']
    df['ichimoku_b'] = ichimoku['IKS_26']
    
    return df


def calculate_signal_score(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate a score for the current market condition."""
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    indicators = {}
    
    # 1. Trend Direction (30% weight)
    ema_trend = 1 if current['ema9'] > current['ema21'] > current['ema200'] else -1
    ichimoku_trend = 1 if current['close'] > current['ichimoku_a'] > current['ichimoku_b'] else -1
    score += (ema_trend + ichimoku_trend) * 15
    indicators['trend_score'] = (ema_trend + ichimoku_trend) * 15
    
    # 2. Momentum (25% weight)
    rsi_score = 0
    if current['rsi'] < 30:
        rsi_score = 1
    elif current['rsi'] > 70:
        rsi_score = -1
    
    macd_cross = 1 if (prev['macd'] < prev['macd_signal']) and (current['macd'] > current['macd_signal']) else -1
    score += (rsi_score * 10) + (macd_cross * 5)
    indicators['momentum_score'] = (rsi_score * 10) + (macd_cross * 5)
    
    # 3. Volume Analysis (20% weight)
    volume_score = 1 if current['volume'] > current['volume_ma'] * 1.5 else 0
    obv_trend = 1 if current['obv'] > prev['obv'] else -1
    score += (volume_score * 10) + (obv_trend * 5)
    indicators['volume_score'] = (volume_score * 10) + (obv_trend * 5)
    
    # 4. Volatility (15%)
    atr_ratio = current['atr'] / df['atr'].mean()
    volatility_score = 1 if atr_ratio > 1.2 else 0  # Higher volatility is better for trading
    score += volatility_score * 15
    indicators['volatility_score'] = volatility_score * 15
    
    # 5. Trend Strength (10%)
    adx_score = 1 if current['adx'] > 25 else 0
    score += adx_score * 10
    indicators['trend_strength'] = adx_score * 10
    
    # Normalize score to 0-100 range
    score = max(0, min(100, (score + 50) * 2))
    
    return {'score': score, 'indicators': indicators}

def generate_signal(ohlcv: OHLCV) -> Optional[Signal]:
    """
    Generate trading signal based on multiple technical indicators.
    Returns None if no valid signal.
    """
    if len(ohlcv.close) < 200:  # Need enough data for indicators
        return None
    
    df = calculate_indicators(ohlcv)
    analysis = calculate_signal_score(df)
    current = df.iloc[-1]
    
    # Only generate signals for strong conditions
    if analysis['score'] < 40 or analysis['score'] > 60:
        signal_type = (
            SignalType.STRONG_BUY if analysis['score'] >= 80 else
            SignalType.BUY if analysis['score'] >= 60 else
            SignalType.SELL if analysis['score'] <= 40 else
            SignalType.STRONG_SELL if analysis['score'] <= 20 else
            SignalType.NEUTRAL
        )
        
        if signal_type != SignalType.NEUTRAL:
            # Calculate position sizing
            atr = current['atr']
            price = current['close']
            
            # Set stop loss and take profit levels
            if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
                stop_loss = price - (atr * 2)
                take_profit = [
                    price + (atr * 2),
                    price + (atr * 3),
                    price + (atr * 4)
                ]
            else:  # SELL signals
                stop_loss = price + (atr * 2)
                take_profit = [
                    price - (atr * 2),
                    price - (atr * 3),
                    price - (atr * 4)
                ]
            
            # Prepare indicator values for the signal
            indicator_values = {
                'rsi': current['rsi'],
                'atr': current['atr'],
                'adx': current['adx'],
                'volume': current['volume'],
                'volume_ma': current['volume_ma'],
                'ema9': current['ema9'],
                'ema21': current['ema21'],
                'ema200': current['ema200']
            }
            
            return Signal(
                symbol='BTC/USDT',  # This should be dynamic based on your data
                signal_type=signal_type,
                entry=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=analysis['score'],
                indicators=indicator_values,
                timestamp=pd.Timestamp.now()
            )
    
    return None
    
    # Generate signal if conditions met
    signal = None
    if ema_cross_up and trend == 'up':
        entry = current['close']
        stop_loss = entry - (current['atr'] * 1.5)
        take_profit = [
            entry + (entry - stop_loss) * 1.0,
            entry + (entry - stop_loss) * 1.5,
            entry + (entry - stop_loss) * 2.0,
            entry + (entry - stop_loss) * 2.5,
            entry + (entry - stop_loss) * 3.0
        ]
        signal = Signal(
            symbol=ohlcv.symbol,
            side='long',
            entry=round(entry, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=[round(tp, 4) for tp in take_profit],
            risk_level=risk,
            rsi=round(current['rsi'], 1),
            atr=round(current['atr'] / entry * 100, 2),  # ATR as % of price
            adx=round(current['adx'], 1),
            volume=round(current['volume'], 2),
            timestamp=ohlcv.timestamp.iloc[-1]
        )
    
    elif ema_cross_down and trend == 'down':
        entry = current['close']
        stop_loss = entry + (current['atr'] * 1.5)
        take_profit = [
            entry - (stop_loss - entry) * 1.0,
            entry - (stop_loss - entry) * 1.5,
            entry - (stop_loss - entry) * 2.0,
            entry - (stop_loss - entry) * 2.5,
            entry - (stop_loss - entry) * 3.0
        ]
        signal = Signal(
            symbol=ohlcv.symbol,
            side='short',
            entry=round(entry, 4),
            stop_loss=round(stop_loss, 4),
            take_profit=[round(tp, 4) for tp in take_profit],
            risk_level=risk,
            rsi=round(current['rsi'], 1),
            atr=round(current['atr'] / entry * 100, 2),  # ATR as % of price
            adx=round(current['adx'], 1),
            volume=round(current['volume'], 2),
            timestamp=ohlcv.timestamp.iloc[-1]
        )
    
    return signal
