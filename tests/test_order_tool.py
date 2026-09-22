"""Unit tests for the secure local Order Lookup tool."""

import json
from pathlib import Path

import pytest

from src.config import ORDERS_PATH
from src.tools.order_tool import lookup_order, normalize_order_id


@pytest.fixture(scope="module")
def original_orders_content() -> str:
    """Read original content of orders.json to verify read-only behavior."""
    return ORDERS_PATH.read_text(encoding="utf-8")


def test_order_id_normalization():
    """Verify normalization of lowercase, whitespace, and punctuation variations."""
    assert normalize_order_id("ord-1007") == "ORD-1007"
    assert normalize_order_id("  ORD-1007  ") == "ORD-1007"
    assert normalize_order_id("ord-1007.") == "ORD-1007"
    assert normalize_order_id("ord 1007") == "ORD-1007"
    assert normalize_order_id("ORD1007") == "ORD-1007"
    assert normalize_order_id("  ord_1007! ") == "ORD-1007"
    assert normalize_order_id("") == ""


def test_valid_order_lookup_ord_1007():
    """Verify valid order lookup for ORD-1007 (shipped via UPS with delivery estimate)."""
    result = lookup_order("ORD-1007")

    assert result["found"] is True
    assert result["order_id"] == "ORD-1007"
    assert result["status"] == "shipped"
    assert result["carrier"] == "UPS"
    assert result["tracking_number"] == "1ZAR100700000007"
    assert result["estimated_delivery"] == "2026-08-22"
    assert result["handoff"] is False
    assert len(result["items"]) == 1
    assert result["items"][0]["name"] == "Atlas Weekender"


def test_lowercase_and_whitespace_lookup_ord_1007():
    """Verify that lookup succeeds with lowercase, surrounding whitespace, or trailing dot."""
    result = lookup_order("  ord-1007. ")
    assert result["found"] is True
    assert result["order_id"] == "ORD-1007"
    assert result["status"] == "shipped"


def test_cancelled_order_suppresses_stale_eta_ord_1004():
    """Verify that cancelled order ORD-1004 suppresses stale carrier, tracking, and ETA."""
    result = lookup_order("ORD-1004")

    assert result["found"] is True
    assert result["order_id"] == "ORD-1004"
    assert result["status"] == "cancelled"

    # Crucial security rule: Stale delivery information must NOT be exposed
    assert result["carrier"] is None, "Cancelled order must not expose stale carrier"
    assert result["tracking_number"] is None, "Cancelled order must not expose stale tracking"
    assert result["estimated_delivery"] is None, "Cancelled order must not expose stale ETA"
    assert "cancelled" in result["customer_safe_message"].lower()


def test_shipped_order_without_eta_ord_1011():
    """Verify shipped order ORD-1011 with Canada Post handles missing ETA properly."""
    result = lookup_order("ORD-1011")

    assert result["found"] is True
    assert result["order_id"] == "ORD-1011"
    assert result["status"] == "shipped"
    assert result["carrier"] == "Canada Post"
    assert result["tracking_number"] == "AR1011CA00001"
    assert result["estimated_delivery"] is None
    assert result["estimate_unavailable"] is True
    assert "delivery estimate is not currently available" in result["customer_safe_message"]


def test_exception_order_triggers_handoff_ord_1010():
    """Verify that order exception ORD-1010 flags handoff=True."""
    result = lookup_order("ORD-1010")

    assert result["found"] is True
    assert result["order_id"] == "ORD-1010"
    assert result["status"] == "exception"
    assert result["handoff"] is True
    assert "requires support review" in result["customer_safe_message"]


def test_unknown_order_ord_9999():
    """Verify that unknown order ORD-9999 returns explicit not-found and triggers handoff."""
    result = lookup_order("ORD-9999")

    assert result["found"] is False
    assert result["order_id"] == "ORD-9999"
    assert result["handoff"] is True
    assert "not found" in result["message"].lower()
    # Ensure no customer-safe data fields are populated
    assert "status" not in result
    assert "carrier" not in result


def test_prompt_injection_in_warehouse_note_dropped_ord_1005():
    """Verify that injected instruction in ORD-1005 warehouse note is never in tool output."""
    result = lookup_order("ORD-1005")

    assert result["found"] is True
    assert result["status"] == "delayed"
    assert result["estimated_delivery"] == "2026-08-20"

    # Serialize entire result to JSON string to inspect all content
    serialized = json.dumps(result)
    assert "100 coupon" not in serialized
    assert "issue a $100 coupon" not in serialized
    assert "warehouse_note" not in serialized
    assert "internal" not in serialized


def test_pii_strictly_excluded_for_all_orders():
    """Verify that NO order in the entire dataset ever exposes customer PII."""
    with open(ORDERS_PATH, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    for raw_order in raw_dataset["orders"]:
        order_id = raw_order["order_id"]
        result = lookup_order(order_id)

        # 1. Check top-level keys
        assert "customer" not in result, f"Customer object leaked for {order_id}"
        assert "internal" not in result, f"Internal object leaked for {order_id}"

        # 2. Check forbidden specific fields
        serialized = json.dumps(result)
        customer = raw_order.get("customer", {})
        if customer.get("email"):
            assert customer["email"] not in serialized, f"Email leaked for {order_id}"
        if customer.get("shipping_address"):
            assert customer["shipping_address"] not in serialized, f"Address leaked for {order_id}"
        if customer.get("name"):
            assert customer["name"] not in serialized, f"Name leaked for {order_id}"

        internal = raw_order.get("internal", {})
        if internal.get("warehouse_note"):
            assert internal["warehouse_note"] not in serialized, f"Warehouse note leaked for {order_id}"
        assert "risk_score" not in serialized, f"Risk score leaked for {order_id}"
        assert "support_tags" not in serialized, f"Support tags leaked for {order_id}"


def test_empty_or_missing_order_id():
    """Verify lookup with empty string."""
    result = lookup_order("")
    assert result["found"] is False
    assert result["handoff"] is False
    assert "missing" in result["message"].lower()


def test_orders_file_is_read_only(original_orders_content: str):
    """Verify that orders.json was not modified in any way by tool operations."""
    current_content = ORDERS_PATH.read_text(encoding="utf-8")
    assert current_content == original_orders_content, "orders.json was modified by tool execution!"
