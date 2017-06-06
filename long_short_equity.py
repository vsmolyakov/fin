import numpy as np
import pandas as pd

from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.factors import CustomFactor, SimpleMovingAverage, AverageDollarVolume
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters import Q1500US

# Constraint Parameters
NUM_LONG_POSITIONS = 5
NUM_SHORT_POSITIONS = 5

class Momentum(CustomFactor):

    inputs = [USEquityPricing.close]
    window_length = 252

    def compute(self, today, assets, out, prices):
        out[:] = ((prices[-21] - prices[-252])/prices[-252] -
                  (prices[-1] - prices[-21])/prices[-21])

def make_pipeline():
    
    # define alpha factors
    momentum = Momentum()
    growth = morningstar.operation_ratios.revenue_growth.latest
    pe_ratio = morningstar.valuation_ratios.pe_ratio.latest
        
    # Screen out non-desirable securities by defining our universe. 
    mkt_cap_filter = morningstar.valuation.market_cap.latest >= 500000000    
    price_filter = USEquityPricing.close.latest >= 5
    universe = Q1500US() & price_filter & mkt_cap_filter & \
               momentum.notnull() & growth.notnull() & pe_ratio.notnull()

    combined_rank = (
        momentum.rank(mask=universe).zscore() +
        growth.rank(mask=universe).zscore() +
        pe_ratio.rank(mask=universe).zscore()
    )

    longs = combined_rank.top(NUM_LONG_POSITIONS)
    shorts = combined_rank.bottom(NUM_SHORT_POSITIONS)

    long_short_screen = (longs | shorts)        

    # Create pipeline
    pipe = Pipeline(columns = {
        'longs':longs,
        'shorts':shorts,
        'combined_rank':combined_rank,
        'momentum':momentum,
        'growth':growth,            
        'pe_ratio':pe_ratio
    },
    screen = long_short_screen)
    return pipe

def initialize(context):

    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1))        

    attach_pipeline(make_pipeline(), 'long_short_factors')

    # Schedule my rebalance function
    schedule_function(func=rebalance,
                      date_rule=date_rules.month_start(),
                      time_rule=time_rules.market_open(hours=1,minutes=30),
                      half_days=True)
    
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)

def before_trading_start(context, data):
    # Call pipeline_output to get the output
    context.output = pipeline_output('long_short_factors')
    
    context.longs = context.output[context.output['longs']].index.tolist()
    context.shorts = context.output[context.output['shorts']].index.tolist()

    context.long_weight, context.short_weight = assign_weights(context)
    
    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index
   

def assign_weights(context):
    """
    Assign weights to securities that we want to order.
    """
    long_weight = 0.5 / len(context.longs)
    short_weight = -0.5 / len(context.shorts)
        
    return long_weight, short_weight
 
def rebalance(context, data):
    
    for security in context.portfolio.positions:
        if security not in context.longs and \
        security not in context.shorts and data.can_trade(security):
            order_target_percent(security, 0)

    for security in context.longs:
        if data.can_trade(security):
            order_target_percent(security, context.long_weight)

    for security in context.shorts:
        if data.can_trade(security):
            order_target_percent(security, context.short_weight)        
    
def recording_statements(context, data):
    # Check how many long and short positions we have.
    longs = shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
        elif position.amount < 0:
            shorts += 1

    # Record our variables.
    record(leverage=context.account.leverage, long_count=longs, short_count=shorts)

