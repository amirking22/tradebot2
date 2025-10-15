"""In-memory cache for OHLCV data, signals, and active signal tracking."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import uuid
from enum import Enum, auto

from market_data import OHLCV

class SignalStatus(Enum):
    ACTIVE = auto()
    TARGET_REACHED = auto()
    STOP_LOSS_HIT = auto()
    CANCELLED = auto()

class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

# Cache expiration time (in seconds)
CACHE_TTL = 300  # 5 minutes

@dataclass
class CacheEntry:
    """Cache entry for OHLCV data."""
    data: OHLCV
    timestamp: datetime
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return (datetime.now() - self.timestamp).total_seconds() > CACHE_TTL


class DataCache:
    """In-memory cache for OHLCV data."""
    
    def __init__(self):
        self.cache: Dict[str, CacheEntry] = {}
    
    def _get_key(self, exchange: str, symbol: str, timeframe: str) -> str:
        """Generate a unique cache key."""
        return f"{exchange}:{symbol}:{timeframe}"
    
    def get(
        self,
        exchange: str,
        symbol: str,
        timeframe: str
    ) -> Optional[OHLCV]:
        """Get cached OHLCV data if it exists and is not expired."""
        key = self._get_key(exchange, symbol, timeframe)
        entry = self.cache.get(key)
        
        if entry is None:
            return None
            
        if entry.is_expired():
            del self.cache[key]
            return None
            
        return entry.data
    
    def set(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        data: OHLCV
    ) -> None:
        """Cache OHLCV data."""
        key = self._get_key(exchange, symbol, timeframe)
        self.cache[key] = CacheEntry(data=data, timestamp=datetime.now())
    
    def clear_expired(self) -> None:
        """Remove all expired cache entries."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()


class ActiveSignal:
    """Represents an active signal being tracked."""
    
    def __init__(
        self,
        signal_id: str,
        symbol: str,
        direction: SignalDirection,
        entry_price: float,
        stop_loss: float,
        take_profits: List[float],
        message_id: int = None,
        exchange: str = "binance",
        leverage: int = 1,
        risk_percentage: float = 1.0
    ):
        self.signal_id = signal_id or str(uuid.uuid4())
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profits = take_profits
        self.message_id = message_id
        self.exchange = exchange
        self.leverage = leverage
        self.risk_percentage = risk_percentage
        self.status = SignalStatus.ACTIVE
        self.entry_time = datetime.utcnow()
        self.hit_targets = []
        self.closed_at = None
        self.closed_price = None
        self.closed_reason = None
        self.indicators = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization."""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profits': self.take_profits,
            'message_id': self.message_id,
            'exchange': self.exchange,
            'leverage': self.leverage,
            'risk_percentage': self.risk_percentage,
            'status': self.status.name,
            'entry_time': self.entry_time.isoformat(),
            'hit_targets': self.hit_targets,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'closed_price': self.closed_price,
            'closed_reason': self.closed_reason,
            'indicators': self.indicators
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActiveSignal':
        """Create signal from dictionary."""
        signal = cls(
            signal_id=data['signal_id'],
            symbol=data['symbol'],
            direction=SignalDirection[data['direction']],
            entry_price=data['entry_price'],
            stop_loss=data['stop_loss'],
            take_profits=data['take_profits'],
            message_id=data['message_id'],
            exchange=data.get('exchange', 'binance'),
            leverage=data.get('leverage', 1),
            risk_percentage=data.get('risk_percentage', 1.0)
        )
        signal.status = SignalStatus[data['status']]
        signal.entry_time = datetime.fromisoformat(data['entry_time'])
        signal.hit_targets = data['hit_targets']
        if data['closed_at']:
            signal.closed_at = datetime.fromisoformat(data['closed_at'])
        signal.closed_price = data['closed_price']
        signal.closed_reason = data['closed_reason']
        signal.indicators = data.get('indicators', {})
        return signal


class SignalTracker:
    """Tracks active signals and their status."""
    
    def __init__(self):
        self.active_signals: Dict[str, ActiveSignal] = {}
        self.signal_history: List[Dict] = []
        self.max_history = 1000  # Maximum number of historical signals to keep

    def add_signal(self, signal: ActiveSignal) -> str:
        """Add a new signal to track."""
        self.active_signals[signal.signal_id] = signal
        return signal.signal_id

    def update_price(self, symbol: str, current_price: float) -> List[Dict]:
        """Update all signals for a symbol with current price and return any triggers."""
        triggers = []
        
        for signal_id, signal in list(self.active_signals.items()):
            if signal.symbol != symbol or signal.status != SignalStatus.ACTIVE:
                continue
                
            # Check for stop loss hit
            if ((signal.direction == SignalDirection.LONG and current_price <= signal.stop_loss) or
                (signal.direction == SignalDirection.SHORT and current_price >= signal.stop_loss)):
                
                signal.status = SignalStatus.STOP_LOSS_HIT
                signal.closed_at = datetime.utcnow()
                signal.closed_price = current_price
                signal.closed_reason = "Stop loss hit"
                
                triggers.append({
                    'signal_id': signal.signal_id,
                    'type': 'STOP_LOSS',
                    'price': current_price,
                    'message': f"🛑 Stop loss hit at {current_price}"
                })
                
                # Move to history
                self.signal_history.append(signal.to_dict())
                if len(self.signal_history) > self.max_history:
                    self.signal_history.pop(0)
                del self.active_signals[signal_id]
                continue
                
            # Check for take profit hits
            for i, tp in enumerate(signal.take_profits):
                tp_level = i + 1  # 1-based index
                
                if tp_level in signal.hit_targets:
                    continue  # Already hit this target
                    
                if ((signal.direction == SignalDirection.LONG and current_price >= tp) or
                    (signal.direction == SignalDirection.SHORT and current_price <= tp)):
                    
                    signal.hit_targets.append(tp_level)
                    
                    triggers.append({
                        'signal_id': signal.signal_id,
                        'type': 'TAKE_PROFIT',
                        'level': tp_level,
                        'price': tp,
                        'message': f"🎯 Target {tp_level} reached at {tp}"
                    })
                    
                    # If all targets hit, close the signal
                    if len(signal.hit_targets) == len(signal.take_profits):
                        signal.status = SignalStatus.TARGET_REACHED
                        signal.closed_at = datetime.utcnow()
                        signal.closed_price = current_price
                        signal.closed_reason = "All targets reached"
                        
                        # Move to history
                        self.signal_history.append(signal.to_dict())
                        if len(self.signal_history) > self.max_history:
                            self.signal_history.pop(0)
                        del self.active_signals[signal_id]
        
        return triggers

    def get_signal(self, signal_id: str) -> Optional[ActiveSignal]:
        """Get an active signal by ID."""
        return self.active_signals.get(signal_id)

    def close_signal(self, signal_id: str, price: float, reason: str = "Manual close") -> bool:
        """Manually close a signal."""
        if signal_id not in self.active_signals:
            return False
            
        signal = self.active_signals[signal_id]
        signal.status = SignalStatus.CANCELLED
        signal.closed_at = datetime.utcnow()
        signal.closed_price = price
        signal.closed_reason = reason
        
        # Move to history
        self.signal_history.append(signal.to_dict())
        if len(self.signal_history) > self.max_history:
            self.signal_history.pop(0)
            
        del self.active_signals[signal_id]
        return True

    def get_active_signals(self, symbol: str = None) -> List[ActiveSignal]:
        """Get all active signals, optionally filtered by symbol."""
        if symbol is None:
            return list(self.active_signals.values())
        return [s for s in self.active_signals.values() if s.symbol == symbol]


class SignalCache:
    """Cache for trading signals."""
    
    def __init__(self):
        self.signals: Dict[str, dict] = {}
        self.signal_ttl = 86400  # 24 hours in seconds
    
    def _get_key(self, exchange: str, symbol: str, timeframe: str) -> str:
        """Generate a unique cache key."""
        return f"{exchange}:{symbol}:{timeframe}"
    
    def add_signal(self, signal: dict) -> None:
        """Add a signal to the cache."""
        key = self._get_key(
            signal.get('exchange', ''),
            signal.get('symbol', ''),
            signal.get('timeframe', '')
        )
        self.signals[key] = {
            'signal': signal,
            'timestamp': datetime.now().timestamp()
        }
    
    def has_active_signal(
        self,
        exchange: str,
        symbol: str,
        timeframe: str
    ) -> bool:
        """Check if there's an active signal for the given parameters."""
        key = self._get_key(exchange, symbol, timeframe)
        signal_data = self.signals.get(key)
        
        if signal_data is None:
            return False
            
        # Check if signal is expired
        if (datetime.now().timestamp() - signal_data['timestamp']) > self.signal_ttl:
            del self.signals[key]
            return False
            
        return True
    
    def clear_expired(self) -> None:
        """Remove expired signals from the cache."""
        now = datetime.now().timestamp()
        expired_keys = [
            key for key, data in self.signals.items()
            if (now - data['timestamp']) > self.signal_ttl
        ]
        for key in expired_keys:
            del self.signals[key]


# Global cache instances
data_cache = DataCache()
signal_cache = SignalCache()
signal_tracker = SignalTracker()


def get_cached_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str
) -> Optional[OHLCV]:
    """Get cached OHLCV data if available and not expired."""
    return data_cache.get(exchange, symbol, timeframe)


def cache_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
    data: OHLCV
) -> None:
    """Cache OHLCV data."""
    data_cache.set(exchange, symbol, timeframe, data)


def clear_expired_cache() -> None:
    """Clear expired entries from all caches."""
    data_cache.clear_expired()
    signal_cache.clear_expired()


# Example usage
if __name__ == "__main__":
    # Example of caching OHLCV data
    from market_data import OHLCV
    import numpy as np
    
    # Create a sample OHLCV data
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1h')
    data = OHLCV(
        exchange='binance',
        symbol='BTC/USDT',
        timeframe='1h',
        timestamp=timestamps,
        open=pd.Series(np.random.random(100) * 50000 + 20000, index=timestamps),
        high=pd.Series(np.random.random(100) * 1000 + 50000, index=timestamps),
        low=pd.Series(np.random.random(100) * 1000 + 19000, index=timestamps),
        close=pd.Series(np.random.random(100) * 1000 + 25000, index=timestamps),
        volume=pd.Series(np.random.random(100) * 1000, index=timestamps)
    )
    
    # Cache the data
    cache_ohlcv('binance', 'BTC/USDT', '1h', data)
    
    # Retrieve from cache
    cached_data = get_cached_ohlcv('binance', 'BTC/USDT', '1h')
    print(f"Cached data: {cached_data is not None}")
    
    # Example of signal caching
    signal = {
        'exchange': 'binance',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'side': 'buy',
        'entry': 45000.5,
        'stop_loss': 44000.0,
        'take_profit': [46000.0, 47000.0, 48000.0],
        'timestamp': datetime.now().timestamp()
    }
    
    signal_cache.add_signal(signal)
    has_signal = signal_cache.has_active_signal('binance', 'BTC/USDT', '1h')
    print(f"Has active signal: {has_signal}")
