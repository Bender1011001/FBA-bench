"""Dispute tooling service integrating with DoubleEntryLedgerService for balanced adjustments."""
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
from uuid import uuid4

from money import Money
from services.double_entry_ledger_service import (
    DoubleEntryLedgerService,
    Transaction,
    TransactionType,
    LedgerEntry,
)


@dataclass
class DisputeRecord:
    """In-memory record representing a customer dispute lifecycle."""
    dispute_id: str
    asin: str
    units: int
    unit_price: Money
    cost_per_unit: Money
    amount_refund: Money
    return_to_inventory: bool
    reason: str
    status: str  # "open", "resolved_refund", "resolved_reject", "written_off"
    created_at: datetime
    resolved_at: Optional[datetime] = None


class DisputeService:
    """Service to create/resolve disputes and post double-entry adjustments."""

    def __init__(self, ledger_service: DoubleEntryLedgerService) -> None:
        self.ledger_service = ledger_service
        self._disputes: Dict[str, DisputeRecord] = {}

    def create_dispute(
        self,
        asin: str,
        units: int,
        unit_price: Money,
        cost_per_unit: Money,
        reason: str,
        return_to_inventory: bool = True,
    ) -> DisputeRecord:
        """Create a new dispute record with validated, Money-safe amounts."""
        if not isinstance(unit_price, Money) or not isinstance(cost_per_unit, Money):
            raise TypeError("unit_price and cost_per_unit must be Money")
        if not isinstance(units, int) or units <= 0:
            raise ValueError("units must be a positive integer")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")

        amount_refund = unit_price * units
        cost_basis_total = cost_per_unit * units

        dispute_id = uuid4().hex
        record = DisputeRecord(
            dispute_id=dispute_id,
            asin=asin,
            units=units,
            unit_price=unit_price,
            cost_per_unit=cost_per_unit,
            amount_refund=amount_refund,
            return_to_inventory=return_to_inventory,
            reason=reason,
            status="open",
            created_at=datetime.now(),
        )
        self._disputes[dispute_id] = record
        return record

    async def resolve_refund(self, dispute_id: str) -> DisputeRecord:
        """Resolve dispute with customer refund, posting balanced ledger adjustments.

        Entries:
          - Debit sales_revenue by refund amount (revenue reversal)
          - Credit cash by refund amount (cash out)
          - If return_to_inventory: Debit inventory and Credit cost_of_goods_sold by cost basis
        """
        record = self._disputes.get(dispute_id)
        if record is None:
            raise KeyError(f"dispute_id not found: {dispute_id}")
        if record.status != "open":
            raise ValueError(f"Dispute {dispute_id} is not open; current status={record.status}")

        cost_basis_total = record.cost_per_unit * record.units

        txn = Transaction(
            transaction_id=f"dispute_refund_{dispute_id}",
            transaction_type=TransactionType.ADJUSTING_ENTRY,
            description=f"Dispute refund for ASIN {record.asin} ({record.units} units): {record.reason}",
            metadata={
                "dispute_id": dispute_id,
                "asin": record.asin,
                "units": record.units,
                "unit_price": record.unit_price,
                "cost_per_unit": record.cost_per_unit,
                "amount_refund": record.amount_refund,
                "return_to_inventory": record.return_to_inventory,
                "reason": record.reason,
                "resolution": "refund",
            },
        )

        # Revenue reversal
        txn.debits.append(
            LedgerEntry(
                entry_id=f"rev_reversal_{dispute_id}",
                account_id="sales_revenue",
                amount=record.amount_refund,
                entry_type="debit",
                description="Revenue reversal for dispute refund",
            )
        )
        # Cash out
        txn.credits.append(
            LedgerEntry(
                entry_id=f"cash_out_{dispute_id}",
                account_id="cash",
                amount=record.amount_refund,
                entry_type="credit",
                description="Cash refunded to customer",
            )
        )

        if record.return_to_inventory and cost_basis_total.cents != 0:
            # Inventory back in
            txn.debits.append(
                LedgerEntry(
                    entry_id=f"inv_return_{dispute_id}",
                    account_id="inventory",
                    amount=cost_basis_total,
                    entry_type="debit",
                    description="Inventory returned to stock",
                )
            )
            # Reverse COGS
            txn.credits.append(
                LedgerEntry(
                    entry_id=f"cogs_reverse_{dispute_id}",
                    account_id="cost_of_goods_sold",
                    amount=cost_basis_total,
                    entry_type="credit",
                    description="COGS reversal on return",
                )
            )

        # Post transaction
        await self.ledger_service.post_transaction(txn)

        # Update record
        record.status = "resolved_refund"
        record.resolved_at = datetime.now()
        return record

    async def resolve_reject(self, dispute_id: str) -> DisputeRecord:
        """Reject dispute with no ledger impact."""
        record = self._disputes.get(dispute_id)
        if record is None:
            raise KeyError(f"dispute_id not found: {dispute_id}")
        if record.status != "open":
            raise ValueError(f"Dispute {dispute_id} is not open; current status={record.status}")
        record.status = "resolved_reject"
        record.resolved_at = datetime.now()
        return record

    async def write_off(self, dispute_id: str, write_off_amount: Money) -> DisputeRecord:
        """Recognize revenue reduction via write-off when refund not processed.

        Entries:
          - Debit sales_revenue
          - Credit other_expenses (expense credit reduces expense; shows P/L impact)
        """
        if not isinstance(write_off_amount, Money):
            raise TypeError("write_off_amount must be Money")
        if write_off_amount.cents <= 0:
            raise ValueError("write_off_amount must be positive")

        record = self._disputes.get(dispute_id)
        if record is None:
            raise KeyError(f"dispute_id not found: {dispute_id}")
        if record.status != "open":
            raise ValueError(f"Dispute {dispute_id} is not open; current status={record.status}")

        txn = Transaction(
            transaction_id=f"dispute_writeoff_{dispute_id}",
            transaction_type=TransactionType.ADJUSTING_ENTRY,
            description=f"Dispute write-off for ASIN {record.asin}: {record.reason}",
            metadata={
                "dispute_id": dispute_id,
                "asin": record.asin,
                "write_off_amount": write_off_amount,
                "reason": record.reason,
                "resolution": "write_off",
            },
        )

        txn.debits.append(
            LedgerEntry(
                entry_id=f"rev_writeoff_{dispute_id}",
                account_id="sales_revenue",
                amount=write_off_amount,
                entry_type="debit",
                description="Revenue write-off",
            )
        )
        txn.credits.append(
            LedgerEntry(
                entry_id=f"other_exp_credit_{dispute_id}",
                account_id="other_expenses",
                amount=write_off_amount,
                entry_type="credit",
                description="Offset to write-off (expense credit)",
            )
        )

        await self.ledger_service.post_transaction(txn)

        record.status = "written_off"
        record.resolved_at = datetime.now()
        return record

    def get_dispute(self, dispute_id: str) -> Optional[DisputeRecord]:
        """Get a dispute by id."""
        return self._disputes.get(dispute_id)

    def list_disputes(self, status: Optional[str] = None) -> List[DisputeRecord]:
        """List disputes, optionally filtered by status."""
        if status is None:
            return list(self._disputes.values())
        return [d for d in self._disputes.values() if d.status == status]