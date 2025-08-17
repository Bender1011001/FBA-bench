import pytest

from money import Money
from services.double_entry_ledger_service import DoubleEntryLedgerService
from services.dispute_service import DisputeService


@pytest.mark.asyncio
async def test_create_and_resolve_refund_with_inventory_return():
    ledger = DoubleEntryLedgerService(config={})
    svc = DisputeService(ledger)

    # Create dispute: 2 units * $12 price, $5 cost per unit, with inventory return
    rec = svc.create_dispute(
        asin="B00TEST123",
        units=2,
        unit_price=Money.from_dollars("12.00"),
        cost_per_unit=Money.from_dollars("5.00"),
        reason="Customer dissatisfaction",
        return_to_inventory=True,
    )

    # Resolve with refund; posts balanced adjusting entry
    updated = await svc.resolve_refund(rec.dispute_id)

    assert updated.status == "resolved_refund"
    assert updated.resolved_at is not None

    # Expected ledger impacts:
    # - sales_revenue debited $24.00 => revenue credit-normal -> balance becomes -$24.00
    # - cash credited $24.00 => asset debit-normal -> balance becomes -$24.00 (cash out)
    # - inventory debited $10.00 => asset debit-normal -> +$10.00
    # - cost_of_goods_sold credited $10.00 => expense debit-normal -> -$10.00 (expense reduced)
    assert ledger.get_account_balance("sales_revenue").cents == -2400
    assert ledger.get_account_balance("cash").cents == -2400
    assert ledger.get_account_balance("inventory").cents == 1000
    assert ledger.get_account_balance("cost_of_goods_sold").cents == -1000

    # Trial balance must remain balanced
    assert ledger.is_trial_balance_balanced() is True


@pytest.mark.asyncio
async def test_resolve_reject_has_no_ledger_impact():
    ledger = DoubleEntryLedgerService(config={})
    svc = DisputeService(ledger)

    rec = svc.create_dispute(
        asin="B00TESTREJ",
        units=1,
        unit_price=Money.from_dollars("10.00"),
        cost_per_unit=Money.from_dollars("4.00"),
        reason="Invalid claim",
        return_to_inventory=False,
    )

    updated = await svc.resolve_reject(rec.dispute_id)

    assert updated.status == "resolved_reject"
    assert updated.resolved_at is not None

    # No ledger impact expected; balances remain zero
    for acct in ("sales_revenue", "cash", "inventory", "cost_of_goods_sold", "other_expenses"):
        assert ledger.get_account_balance(acct).cents == 0

    assert ledger.is_trial_balance_balanced() is True


@pytest.mark.asyncio
async def test_write_off_recognizes_revenue_reduction():
    ledger = DoubleEntryLedgerService(config={})
    svc = DisputeService(ledger)

    rec = svc.create_dispute(
        asin="B00WRITEOFF",
        units=1,
        unit_price=Money.from_dollars("9.99"),
        cost_per_unit=Money.from_dollars("3.50"),
        reason="Operational adjustment",
        return_to_inventory=False,
    )

    updated = await svc.write_off(rec.dispute_id, Money.from_dollars("6.00"))

    assert updated.status == "written_off"
    assert updated.resolved_at is not None

    # Expected: sales_revenue debited $6 => -$6 on a credit-normal account
    #           other_expenses credited $6 => -$6 on a debit-normal expense account (reduces expense)
    assert ledger.get_account_balance("sales_revenue").cents == -600
    assert ledger.get_account_balance("other_expenses").cents == -600

    assert ledger.is_trial_balance_balanced() is True


@pytest.mark.asyncio
async def test_invalid_ids_and_state_transitions():
    ledger = DoubleEntryLedgerService(config={})
    svc = DisputeService(ledger)

    rec = svc.create_dispute(
        asin="B00STATE",
        units=1,
        unit_price=Money.from_dollars("8.00"),
        cost_per_unit=Money.from_dollars("3.00"),
        reason="State transition test",
        return_to_inventory=True,
    )

    # Unknown dispute_id should raise KeyError
    with pytest.raises(KeyError):
        await svc.resolve_refund("unknown_id")

    with pytest.raises(KeyError):
        await svc.write_off("unknown_id", Money.from_dollars("1.00"))

    # Resolve refund once
    await svc.resolve_refund(rec.dispute_id)

    # Second resolution attempts should fail due to invalid state
    with pytest.raises(ValueError):
        await svc.resolve_refund(rec.dispute_id)

    with pytest.raises(ValueError):
        await svc.resolve_reject(rec.dispute_id)

    with pytest.raises(ValueError):
        await svc.write_off(rec.dispute_id, Money.from_dollars("1.00"))