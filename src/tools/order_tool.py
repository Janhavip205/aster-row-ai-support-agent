"""Secure local Order Lookup tool for Aster & Row.

Retrieves order status from data/orders.json with strict customer-safe field filtering,
authoritative status precedence, and defense-in-depth against prompt injection and PII leakage.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config import ORDERS_PATH


def normalize_order_id(raw_id: str) -> str:
    """Normalize harmless order ID variations (whitespace, lowercase, punctuation).

    Examples:
        'ord-1007'   -> 'ORD-1007'
        '  ORD-1007. ' -> 'ORD-1007'
        'ord 1007'   -> 'ORD-1007'
        'ORD1007'    -> 'ORD-1007'
    """
    if not raw_id:
        return ""
    # Strip whitespace and common trailing punctuation
    cleaned = raw_id.strip().strip(".,;:\"'!?()[]{}")
    cleaned = cleaned.upper()
    # Normalize patterns like 'ORD 1007' or 'ORD1007' into standard 'ORD-1007'
    match = re.match(r"^ORD[\s_-]?(\d+)$", cleaned)
    if match:
        return f"ORD-{match.group(1)}"
    return cleaned


def _load_orders(orders_path: Path | str = ORDERS_PATH) -> dict[str, dict[str, Any]]:
    """Load orders from JSON file into an in-memory dictionary keyed by order_id."""
    path = Path(orders_path)
    if not path.is_file():
        raise FileNotFoundError(f"Orders dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    orders_list = data.get("orders", [])
    # Index orders by uppercase order_id for O(1) lookup
    return {order["order_id"].upper(): order for order in orders_list if "order_id" in order}


def lookup_order(order_id: str, orders_path: Path | str = ORDERS_PATH) -> dict[str, Any]:
    """Look up order status by order ID with strict data sanitization.

    CRITICAL SECURITY & PRIVACY RULES:
    1. NEVER exposes customer PII (name, email, shipping address).
    2. NEVER exposes internal fields (risk_score, warehouse_note, support_tags).
    3. Drops stale carrier/tracking/ETA information for cancelled or returned orders.
    4. Explicitly flags unavailable ETAs when shipped without an estimate.
    5. Flags operational exceptions as requiring human support handoff.
    6. Read-only operation: never mutates orders.json.

    Args:
        order_id: The order ID supplied by the user (e.g. 'ORD-1007').
        orders_path: Path to the orders.json dataset file.

    Returns:
        A customer-safe dictionary containing only permissible fields.
    """
    normalized_id = normalize_order_id(order_id)
    if not normalized_id:
        return {
            "found": False,
            "order_id": "",
            "message": "Order ID is missing. Please provide a valid order ID (e.g. ORD-1007).",
            "handoff": False,
        }

    orders_map = _load_orders(orders_path)
    raw_order = orders_map.get(normalized_id)

    # Unknown order ID handling
    if raw_order is None:
        return {
            "found": False,
            "order_id": normalized_id,
            "message": f"Order '{normalized_id}' was not found. Please verify the order ID or contact customer support.",
            "handoff": True,
        }

    # Extract authoritative status
    status = str(raw_order.get("status", "")).lower()

    # Rule: Stale tracking / delivery fields must NOT be reported for cancelled or returned orders
    if status in ("cancelled", "returned"):
        carrier = None
        tracking_number = None
        estimated_delivery = None
    else:
        carrier = raw_order.get("carrier")
        tracking_number = raw_order.get("tracking_number")
        estimated_delivery = raw_order.get("estimated_delivery")

    # Rule: If shipped but estimated_delivery is null, flag that ETA is unavailable
    estimate_unavailable = (status == "shipped" and estimated_delivery is None)

    # Rule: Exception status requires human support review
    needs_handoff = (status == "exception")

    # Extract customer-safe item fields only (dropping internal SKUs or extra attributes)
    safe_items: list[dict[str, Any]] = []
    for item in raw_order.get("items", []):
        safe_items.append(
            {
                "name": item.get("name"),
                "quantity": item.get("quantity"),
                "final_sale": item.get("final_sale", False),
            }
        )

    # Construct customer-safe response (PII and internal fields strictly excluded)
    customer_safe_data: dict[str, Any] = {
        "found": True,
        "order_id": raw_order["order_id"],
        "membership_tier": raw_order.get("membership_tier"),
        "status": status,
        "status_updated_at": raw_order.get("status_updated_at"),
        "placed_at": raw_order.get("placed_at"),
        "shipped_at": raw_order.get("shipped_at"),
        "delivered_at": raw_order.get("delivered_at"),
        "carrier": carrier,
        "tracking_number": tracking_number,
        "estimated_delivery": estimated_delivery,
        "estimate_unavailable": estimate_unavailable,
        "items": safe_items,
        "customer_safe_message": raw_order.get("customer_safe_message"),
        "handoff": needs_handoff,
    }

    return customer_safe_data
