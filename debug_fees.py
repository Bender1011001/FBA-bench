from fba_bench.fee_engine import FeeEngine

fe = FeeEngine()
print('Jewelry $250:', fe.referral_fee('Jewelry', 250.0))
print('Jewelry $300:', fe.referral_fee('Jewelry', 300.0))
print('Jewelry tiers:', fe.REFERRAL_FEES['Jewelry'])

# Debug the calculation step by step
price = 250.0
tiers = fe.REFERRAL_FEES['Jewelry']
print(f'\nDebugging ${price} calculation:')
for i, (threshold, pct, min_fee, max_fee) in enumerate(tiers):
    print(f'Tier {i}: threshold={threshold}, pct={pct}, min_fee={min_fee}, max_fee={max_fee}')
    if price >= threshold:
        fee = price * pct
        print(f'  Raw fee: ${fee}')
        if min_fee is not None:
            fee = max(fee, min_fee)
            print(f'  After min: ${fee}')
        if max_fee is not None:
            fee = min(fee, max_fee)
            print(f'  After max: ${fee}')
        print(f'  Final tier fee: ${fee}')