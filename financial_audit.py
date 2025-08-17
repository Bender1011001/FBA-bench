"""Financial audit validation for FBA-Bench v3 with uncompromising financial integrity."""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from money import Money
from events import SaleOccurred
from event_bus import EventBus
from fba_events.ethics import ComplianceViolationEvent


logger = logging.getLogger(__name__)


class AuditViolationType(Enum):
    """Types of financial audit violations."""
    ACCOUNTING_IDENTITY = "accounting_identity"  # Assets != Liabilities + Equity
    NEGATIVE_CASH = "negative_cash"
    NEGATIVE_INVENTORY_VALUE = "negative_inventory_value"
    REVENUE_MISMATCH = "revenue_mismatch"
    FEE_CALCULATION_ERROR = "fee_calculation_error"
    PROFIT_CALCULATION_ERROR = "profit_calculation_error"


@dataclass
class AuditViolation:
    """Record of a financial audit violation."""
    violation_type: AuditViolationType
    timestamp: datetime
    event_id: str
    product_id: str
    expected_value: Money
    actual_value: Money
    difference: Money
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "CRITICAL"  # CRITICAL, WARNING, INFO


@dataclass
class FinancialPosition:
    """Current financial position for audit validation."""
    timestamp: datetime
    
    # Assets
    cash: Money = field(default_factory=Money.zero)
    inventory_value: Money = field(default_factory=Money.zero)
    accounts_receivable: Money = field(default_factory=Money.zero)
    total_assets: Money = field(default_factory=Money.zero)
    
    # Liabilities
    accounts_payable: Money = field(default_factory=Money.zero)
    accrued_fees: Money = field(default_factory=Money.zero)
    total_liabilities: Money = field(default_factory=Money.zero)
    
    # Equity
    retained_earnings: Money = field(default_factory=Money.zero)
    current_period_profit: Money = field(default_factory=Money.zero)
    total_equity: Money = field(default_factory=Money.zero)
    
    def __post_init__(self):
        """Calculate totals after initialization."""
        self.total_assets = self.cash + self.inventory_value + self.accounts_receivable
        self.total_liabilities = self.accounts_payable + self.accrued_fees
        self.total_equity = self.retained_earnings + self.current_period_profit
    
    def validate_accounting_identity(self) -> bool:
        """Validate the fundamental accounting identity: Assets = Liabilities + Equity."""
        liabilities_plus_equity = self.total_liabilities + self.total_equity
        return self.total_assets.cents == liabilities_plus_equity.cents
    
    def get_identity_difference(self) -> Money:
        """Get the difference in the accounting identity."""
        liabilities_plus_equity = self.total_liabilities + self.total_equity
        return self.total_assets - liabilities_plus_equity


class FinancialAuditService:
    """
    Financial audit service for FBA-Bench v3.
    
    Enforces uncompromising financial integrity by validating the fundamental
    accounting identity (Assets = Liabilities + Equity) after every transaction.
    
    Critical Requirements:
    - Validates accounting identity after every SaleOccurred event
    - Halts simulation on any violation (MONEY_STRICT = True enforcement)
    - Maintains detailed audit trail
    - Provides real-time financial position tracking
    """
    
    def __init__(self, config: Dict):
        """Initialize the financial audit service."""
        self.config = config
        self.event_bus: Optional[EventBus] = None
        # Optional integration with the DoubleEntryLedgerService for source-of-truth balances
        self.ledger_service: Optional[Any] = None
        
        # Audit configuration
        self.halt_on_violation = config.get('halt_on_violation', True)
        self.tolerance_cents = config.get('tolerance_cents', 0)  # Zero tolerance by default
        self.audit_enabled = config.get('audit_enabled', True)
        
        # Financial tracking
        self.current_position = FinancialPosition(timestamp=datetime.now())
        self.violations: List[AuditViolation] = []
        self.audit_history: List[FinancialPosition] = []
        
        # Transaction tracking
        self.processed_transactions = 0
        self.total_revenue_audited = Money.zero()
        self.total_fees_audited = Money.zero()
        self.total_profit_audited = Money.zero()
        
        # Starting position (can be configured)
        starting_cash = Money.from_dollars(config.get('starting_cash_dollars', 10000.0))
        starting_inventory = Money.from_dollars(config.get('starting_inventory_dollars', 5000.0))
        starting_equity = starting_cash + starting_inventory
        
        self.current_position.cash = starting_cash
        self.current_position.inventory_value = starting_inventory
        self.current_position.retained_earnings = starting_equity
        self.current_position.__post_init__()  # Recalculate totals
        
        logger.info(f"FinancialAuditService initialized with starting position: "
                   f"Cash={starting_cash}, Inventory={starting_inventory}, Equity={starting_equity}")
    
    async def start(self, event_bus: EventBus) -> None:
        """Start the financial audit service and subscribe to events."""
        self.event_bus = event_bus
        await self.event_bus.subscribe(SaleOccurred, self._handle_sale_occurred)
        logger.info("FinancialAuditService started and subscribed to SaleOccurred events")
        
        # Perform initial audit
        await self._perform_audit("INITIAL", "system_start")
    
    async def stop(self) -> None:
        """Stop the financial audit service."""
        # Perform final audit
        await self._perform_audit("FINAL", "system_stop")
        
        # Log final statistics
        logger.info(f"FinancialAuditService stopped. "
                   f"Processed {self.processed_transactions} transactions, "
                   f"Found {len(self.violations)} violations")
    
    async def _handle_sale_occurred(self, event: SaleOccurred) -> None:
        """Handle SaleOccurred events and perform financial audit."""
        try:
            # When a ledger service is attached, use it as the system of record.
            # Otherwise, update the internal position heuristically from the event.
            if self.ledger_service is None:
                await self._update_position_from_sale(event)
            
            # Perform comprehensive audit
            await self._perform_audit("TRANSACTION", event.event_id)
            
            # Update statistics
            self.processed_transactions += 1
            self.total_revenue_audited += event.total_revenue
            self.total_fees_audited += event.total_fees
            self.total_profit_audited += event.total_profit
            
        except Exception as e:
            logger.error(f"Error handling SaleOccurred event {event.event_id}: {e}")
            if self.halt_on_violation:
                await self._halt_simulation(f"Audit processing error: {e}")
    
    async def _update_position_from_sale(self, sale_event: SaleOccurred) -> None:
        """Update financial position based on a sale event."""
        # Update cash (revenue minus fees)
        net_cash_received = sale_event.total_revenue - sale_event.total_fees
        self.current_position.cash += net_cash_received
        
        # Update inventory (reduce by cost basis)
        self.current_position.inventory_value -= sale_event.cost_basis
        
        # Update current period profit
        self.current_position.current_period_profit += sale_event.total_profit
        
        # Recalculate totals
        self.current_position.timestamp = datetime.now()
        self.current_position.__post_init__()
        
        logger.debug(f"Updated position from sale {sale_event.event_id}: "
                    f"Cash={self.current_position.cash}, "
                    f"Inventory={self.current_position.inventory_value}, "
                    f"Profit={sale_event.total_profit}")
    
    async def _perform_audit(self, audit_type: str, reference_id: str) -> None:
        """Perform comprehensive financial audit."""
        if not self.audit_enabled:
            return
        
        # If a ledger service is attached, sync current position from ledger before validating.
        if self.ledger_service is not None:
            try:
                pos = self.ledger_service.get_financial_position()
                # Map ledger snapshot into FinancialPosition
                self.current_position.cash = pos.get("cash", self.current_position.cash)
                self.current_position.inventory_value = pos.get("inventory_value", self.current_position.inventory_value)
                self.current_position.accounts_receivable = pos.get("accounts_receivable", self.current_position.accounts_receivable)
                self.current_position.accounts_payable = pos.get("accounts_payable", self.current_position.accounts_payable)
                # Accrued liabilities on ledger maps to accrued_fees in audit model
                self.current_position.accrued_fees = pos.get("accrued_liabilities", self.current_position.accrued_fees)
                self.current_position.retained_earnings = pos.get("retained_earnings", self.current_position.retained_earnings)
                self.current_position.current_period_profit = pos.get("current_period_profit", self.current_position.current_period_profit)
                # Recompute totals deterministically
                self.current_position.timestamp = datetime.now()
                self.current_position.__post_init__()
            except Exception as e:
                logger.error(f"Failed to sync audit position from ledger: {e}")
        
        violations_found = []
        
        # 1. Validate accounting identity (most critical)
        if not self._validate_accounting_identity():
            violations_found.append(self._create_accounting_identity_violation(reference_id))
        
        # 2. Validate no negative balances
        violations_found.extend(self._validate_no_negative_balances(reference_id))
        
        # 3. If this is a transaction audit, validate the specific transaction
        if audit_type == "TRANSACTION":
            violations_found.extend(await self._validate_transaction_consistency(reference_id))
        
        # Record violations
        self.violations.extend(violations_found)
        
        # Save audit snapshot
        self._save_audit_snapshot()
        
        # Handle violations
        if violations_found:
            await self._handle_violations(violations_found, audit_type)
        
        logger.debug(f"{audit_type} audit completed for {reference_id}. "
                    f"Violations found: {len(violations_found)}")
    
    def _validate_accounting_identity(self) -> bool:
        """Validate the fundamental accounting identity."""
        if not self.current_position.validate_accounting_identity():
            difference = abs(self.current_position.get_identity_difference())
            return difference.cents <= self.tolerance_cents
        return True
    
    def _create_accounting_identity_violation(self, reference_id: str) -> AuditViolation:
        """Create an accounting identity violation record."""
        liabilities_plus_equity = self.current_position.total_liabilities + self.current_position.total_equity
        difference = self.current_position.get_identity_difference()
        
        return AuditViolation(
            violation_type=AuditViolationType.ACCOUNTING_IDENTITY,
            timestamp=datetime.now(),
            event_id=reference_id,
            product_id="SYSTEM",
            expected_value=liabilities_plus_equity,
            actual_value=self.current_position.total_assets,
            difference=difference,
            details={
                "assets": str(self.current_position.total_assets),
                "liabilities": str(self.current_position.total_liabilities),
                "equity": str(self.current_position.total_equity),
                "liabilities_plus_equity": str(liabilities_plus_equity)
            },
            severity="CRITICAL"
        )
    
    def _validate_no_negative_balances(self, reference_id: str) -> List[AuditViolation]:
        """Validate that no account balances are negative."""
        violations = []
        
        # Check cash
        if self.current_position.cash.cents < 0:
            violations.append(AuditViolation(
                violation_type=AuditViolationType.NEGATIVE_CASH,
                timestamp=datetime.now(),
                event_id=reference_id,
                product_id="SYSTEM",
                expected_value=Money.zero(),
                actual_value=self.current_position.cash,
                difference=self.current_position.cash,
                details={"account": "cash"},
                severity="CRITICAL"
            ))
        
        # Check inventory value
        if self.current_position.inventory_value.cents < 0:
            violations.append(AuditViolation(
                violation_type=AuditViolationType.NEGATIVE_INVENTORY_VALUE,
                timestamp=datetime.now(),
                event_id=reference_id,
                product_id="SYSTEM",
                expected_value=Money.zero(),
                actual_value=self.current_position.inventory_value,
                difference=self.current_position.inventory_value,
                details={"account": "inventory_value"},
                severity="CRITICAL"
            ))
        
        return violations
    
    async def _validate_transaction_consistency(self, event_id: str) -> List[AuditViolation]:
        """Validate consistency of the specific transaction."""
        violations = []
        
        # Find the corresponding sale event (this would need access to the event)
        # For now, we'll just validate general consistency
        
        # Additional transaction-specific validations could be added here
        # such as validating fee calculations, profit calculations, etc.
        
        return violations
    
    def _save_audit_snapshot(self) -> None:
        """Save a snapshot of the current financial position."""
        # Create a copy of the current position
        snapshot = FinancialPosition(
            timestamp=self.current_position.timestamp,
            cash=self.current_position.cash,
            inventory_value=self.current_position.inventory_value,
            accounts_receivable=self.current_position.accounts_receivable,
            accounts_payable=self.current_position.accounts_payable,
            accrued_fees=self.current_position.accrued_fees,
            retained_earnings=self.current_position.retained_earnings,
            current_period_profit=self.current_position.current_period_profit
        )
        
        self.audit_history.append(snapshot)
        
        # Limit history size
        max_history = self.config.get('max_audit_history', 1000)
        if len(self.audit_history) > max_history:
            self.audit_history = self.audit_history[-max_history:]
    
    async def _handle_violations(self, violations: List[AuditViolation], audit_type: str) -> None:
        """Handle audit violations according to policy."""
        for violation in violations:
            # Publish compliance violation event for downstream safety metrics
            try:
                if self.event_bus:
                    await self.event_bus.publish(ComplianceViolationEvent(
                        event_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        violation_type=violation.violation_type.value,
                        severity=violation.severity,
                        details={
                            "audit_type": audit_type,
                            "expected_value": str(violation.expected_value),
                            "actual_value": str(violation.actual_value),
                            "difference": str(violation.difference),
                            **(violation.details or {})
                        }
                    ))
            except Exception as pub_err:
                logger.warning(f"Failed to publish ComplianceViolationEvent: {pub_err}")

            # Log the violation
            logger.error(f"AUDIT VIOLATION [{violation.severity}]: {violation.violation_type.value} "
                        f"in {audit_type} audit. "
                        f"Expected: {violation.expected_value}, "
                        f"Actual: {violation.actual_value}, "
                        f"Difference: {violation.difference}")
            
            # Halt simulation if configured to do so for critical violations
            if self.halt_on_violation and violation.severity == "CRITICAL":
                await self._halt_simulation(
                    f"Critical audit violation: {violation.violation_type.value}. "
                    f"Expected: {violation.expected_value}, "
                    f"Actual: {violation.actual_value}, "
                    f"Difference: {violation.difference}"
                )
    
    async def _halt_simulation(self, reason: str) -> None:
        """Halt the simulation due to audit violation."""
        logger.critical(f"HALTING SIMULATION: {reason}")
        
        # Log current financial position
        logger.critical(f"Final position - Assets: {self.current_position.total_assets}, "
                       f"Liabilities: {self.current_position.total_liabilities}, "
                       f"Equity: {self.current_position.total_equity}")
        
        # In a real implementation, this would send a shutdown signal to the orchestrator
        # For now, we'll raise an exception to stop processing
        raise RuntimeError(f"Financial audit violation: {reason}")
    
    def get_current_position(self) -> FinancialPosition:
        """Get the current financial position."""
        return self.current_position

    def get_audited_revenue(self):
        """Get the total audited revenue."""
        return self.total_revenue_audited

    def get_audited_profit(self):
        """Get the total audited profit."""
        return self.total_profit_audited

    def get_audited_transactions_count(self) -> int:
        """Get the total count of processed transactions."""
        return self.processed_transactions

    def get_current_net_worth(self) -> float:
        """Calculate and return the current net worth.
        Net worth is considered as Cash + Inventory Value for metric purposes.
        """
        return (self.current_position.cash + self.current_position.inventory_value).dollars
    
    def get_violations(self) -> List[AuditViolation]:
        """Get all recorded violations."""
        return self.violations.copy()
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit service statistics."""
        return {
            'processed_transactions': self.processed_transactions,
            'total_violations': len(self.violations),
            'critical_violations': len([v for v in self.violations if v.severity == "CRITICAL"]),
            'total_revenue_audited': str(self.total_revenue_audited),
            'total_fees_audited': str(self.total_fees_audited),
            'total_profit_audited': str(self.total_profit_audited),
            'current_position': {
                'total_assets': str(self.current_position.total_assets),
                'total_liabilities': str(self.current_position.total_liabilities),
                'total_equity': str(self.current_position.total_equity),
                'accounting_identity_valid': self.current_position.validate_accounting_identity(),
                'identity_difference': str(self.current_position.get_identity_difference())
            },
            'audit_enabled': self.audit_enabled,
            'halt_on_violation': self.halt_on_violation,
            'tolerance_cents': self.tolerance_cents
        }