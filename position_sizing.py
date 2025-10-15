"""Position sizing and risk management calculations."""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class PositionSize:
    """Position size calculation result."""
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    risk_per_trade: float  # In percentage (e.g., 1.0 for 1%)
    account_balance: float
    
    # Calculated fields
    position_size: float  # In base currency (e.g., USDT)
    position_size_units: float  # In crypto units (e.g., BTC)
    risk_amount: float  # In base currency (e.g., USDT)
    reward_risk_ratio: float  # Reward/risk ratio
    leverage: int = 1  # Leverage (1 for no leverage)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_per_trade': self.risk_per_trade,
            'account_balance': self.account_balance,
            'position_size': self.position_size,
            'position_size_units': self.position_size_units,
            'risk_amount': self.risk_amount,
            'reward_risk_ratio': self.reward_risk_ratio,
            'leverage': self.leverage
        }


def calculate_position_size(
    entry_price: float,
    stop_loss: float,
    take_profit: List[float],
    risk_per_trade: float,
    account_balance: float,
    leverage: int = 1,
    max_leverage: int = 20,
    min_position_size: float = 5.0,  # Minimum position size in quote currency (e.g., USDT)
    max_position_size_pct: float = 20.0  # Max position size as % of account balance
) -> Optional[PositionSize]:
    """
    Calculate position size based on risk parameters.
    
    Args:
        entry_price: Entry price of the trade
        stop_loss: Stop loss price
        take_profit: List of take profit levels
        risk_per_trade: Risk per trade as a percentage (e.g., 1.0 for 1%)
        account_balance: Total account balance in quote currency
        leverage: Desired leverage (default: 1)
        max_leverage: Maximum allowed leverage
        min_position_size: Minimum position size in quote currency
        max_position_size_pct: Maximum position size as % of account balance
        
    Returns:
        PositionSize object with calculated values, or None if invalid parameters
    """
    # Validate inputs
    if (entry_price <= 0 or stop_loss <= 0 or not take_profit or 
            risk_per_trade <= 0 or account_balance <= 0 or 
            leverage < 1 or leverage > max_leverage):
        return None
    
    # Calculate risk amount in quote currency (e.g., USDT)
    risk_amount = (risk_per_trade / 100) * account_balance
    
    # Determine if it's a long or short trade
    is_long = entry_price > stop_loss
    
    # Calculate price difference for risk calculation
    if is_long:
        price_diff = abs(entry_price - stop_loss)
        # Use the first take profit for reward/risk calculation
        reward = abs(take_profit[0] - entry_price) if take_profit else 0
    else:
        price_diff = abs(stop_loss - entry_price)
        # Use the first take profit for reward/risk calculation
        reward = abs(entry_price - take_profit[0]) if take_profit else 0
    
    # Calculate position size in quote currency
    position_size = (risk_amount * entry_price) / price_diff
    
    # Apply leverage
    position_size *= leverage
    
    # Calculate position size in units of base currency
    position_size_units = position_size / entry_price
    
    # Calculate reward/risk ratio
    reward_risk_ratio = reward / price_diff if price_diff > 0 else 0
    
    # Apply position size limits
    max_position_size = (max_position_size_pct / 100) * account_balance * leverage
    position_size = min(position_size, max_position_size)
    
    if position_size < min_position_size:
        return None
    
    # Recalculate units based on final position size
    position_size_units = position_size / entry_price
    
    return PositionSize(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_per_trade=risk_per_trade,
        account_balance=account_balance,
        position_size=position_size,
        position_size_units=position_size_units,
        risk_amount=risk_amount,
        reward_risk_ratio=reward_risk_ratio,
        leverage=leverage
    )


def calculate_step_sizes(
    position_size: PositionSize,
    num_steps: int = 5,
    step_increment: float = 0.2  # 20% increase per step
) -> List[Dict[str, float]]:
    """
    Calculate position sizes for each step of a scaled entry/exit.
    
    Args:
        position_size: PositionSize object with base calculations
        num_steps: Number of steps/levels
        step_increment: Size increment per step as a decimal (e.g., 0.2 for 20%)
        
    Returns:
        List of dicts with step details
    """
    if num_steps < 1:
        return []
    
    steps = []
    remaining_size = position_size.position_size_units
    remaining_value = position_size.position_size
    
    for i in range(1, num_steps + 1):
        if i < num_steps:
            # For all but the last step, calculate the step size
            step_pct = step_increment * (1 + step_increment) ** (i - 1)
            step_size_pct = min(step_pct, 1.0)  # Cap at 100% of remaining
            
            step_units = position_size.position_size_units * step_size_pct
            step_value = position_size.position_size * step_size_pct
            
            # Adjust for floating point precision
            step_units = min(step_units, remaining_size)
            step_value = min(step_value, remaining_value)
            
            remaining_size -= step_units
            remaining_value -= step_value
        else:
            # Last step takes remaining position
            step_units = remaining_size
            step_value = remaining_value
        
        # Calculate price for this step (weighted average)
        if i == 1:
            step_price = position_size.entry_price
        else:
            # For subsequent steps, adjust price based on trend
            # This is a simplified example - you might want to use ATR or other indicators
            price_diff = (position_size.entry_price - position_size.stop_loss) * 0.5
            if position_size.entry_price > position_size.stop_loss:  # Long
                step_price = position_size.entry_price + (price_diff * (i - 1) / num_steps)
            else:  # Short
                step_price = position_size.entry_price - (price_diff * (i - 1) / num_steps)
        
        steps.append({
            'step': i,
            'units': step_units,
            'value': step_value,
            'price': step_price,
            'pct_of_position': (step_value / position_size.position_size) * 100
        })
    
    return steps


def calculate_take_profit_levels(
    entry_price: float,
    stop_loss: float,
    risk_reward_ratios: List[float] = None,
    num_levels: int = 5
) -> List[float]:
    """
    Calculate take profit levels based on risk/reward ratios.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_reward_ratios: List of R:R ratios for each level
        num_levels: Number of take profit levels if risk_reward_ratios not provided
        
    Returns:
        List of take profit prices
    """
    if risk_reward_ratios is None:
        # Default R:R ratios if not provided
        risk_reward_ratios = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    if entry_price <= 0 or stop_loss <= 0 or not risk_reward_ratios:
        return []
    
    is_long = entry_price > stop_loss
    risk_amount = abs(entry_price - stop_loss)
    
    take_profits = []
    
    for i, rr in enumerate(risk_reward_ratios[:num_levels]):
        if is_long:
            tp = entry_price + (risk_amount * rr)
        else:
            tp = entry_price - (risk_amount * rr)
        take_profits.append(round(tp, 8))  # Round to 8 decimal places for crypto
    
    return take_profits


# Example usage
if __name__ == "__main__":
    # Example for a long trade
    entry = 50000.0
    stop = 48000.0
    take_profits = calculate_take_profit_levels(entry, stop)
    
    position = calculate_position_size(
        entry_price=entry,
        stop_loss=stop,
        take_profit=take_profits,
        risk_per_trade=1.0,  # 1% risk per trade
        account_balance=10000.0,
        leverage=5
    )
    
    if position:
        print(f"Position size: {position.position_size:.2f} USDT")
        print(f"Units: {position.position_size_units:.6f} BTC")
        print(f"Risk amount: {position.risk_amount:.2f} USDT")
        print(f"Reward/Risk ratio: {position.reward_risk_ratio:.2f}")
        
        print("\nStep sizes:")
        steps = calculate_step_sizes(position)
        for step in steps:
            print(f"Step {step['step']}: {step['units']:.6f} BTC (${step['value']:.2f}) "
                  f"at ${step['price']:.2f} - {step['pct_of_position']:.1f}%")
    else:
        print("Invalid position parameters")
