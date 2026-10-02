"""
Tests for the ARAG mock API.
Run from the project root with:  pytest -v
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mock_api.main import app

client = TestClient(app)
KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
CATALOGS = [
    "parts_catalog_brakes.md",
    "parts_catalog_electrical.md",
    "parts_catalog_filters_fluids.md",
    "accessories_catalog.md",
]


# ---------------------------------------------------------------- system
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "parts": 29, "orders": 10}


# ---------------------------------------------------------------- data integrity
@pytest.mark.skipif(not KB_DIR.exists(), reason="knowledge_base folder not found")
def test_every_catalog_part_exists_in_inventory():
    """The mock inventory must match the knowledge base catalogs exactly."""
    catalog_ids = set()
    for name in CATALOGS:
        text = (KB_DIR / name).read_text(encoding="utf-8")
        catalog_ids |= set(re.findall(r"^\|\s*([A-Z]{3}-\d{4})\s*\|", text, re.M))
    inventory_ids = {p["part_id"] for p in client.get("/inventory").json()["parts"]}
    assert catalog_ids == inventory_ids


# ---------------------------------------------------------------- inventory: happy paths
def test_get_part_in_stock():
    r = client.get("/inventory/BRK-1001")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "in_stock"
    assert body["total_quantity"] == sum(body["quantities_by_warehouse"].values())
    assert body["estimated_restock_date"] is None


def test_get_part_low_stock():
    body = client.get("/inventory/BRK-1020").json()
    assert body["status"] == "low_stock"
    assert "WH-SOUTH" not in body["warehouses_with_stock"]


def test_get_part_backordered_has_restock_date():
    body = client.get("/inventory/ELC-3030").json()
    assert body["status"] == "backordered"
    assert body["total_quantity"] == 0
    assert body["special_order"] is True
    assert body["estimated_restock_date"] == "2026-10-21"


def test_part_id_is_case_and_space_insensitive():
    r = client.get("/inventory/%20brk-1020%20")
    assert r.status_code == 200
    assert r.json()["part_id"] == "BRK-1020"


def test_filter_by_warehouse():
    body = client.get("/inventory/BRK-1040", params={"warehouse": "wh-north"}).json()
    assert body["quantities_by_warehouse"] == {"WH-NORTH": 14}
    body = client.get("/inventory/BRK-1040", params={"warehouse": "WH-EAST"}).json()
    assert body["status"] == "backordered"


def test_list_inventory_and_filters():
    assert client.get("/inventory").json()["count"] == 29
    brakes = client.get("/inventory", params={"category": "brakes"}).json()
    assert brakes["count"] == 8
    backordered = client.get("/inventory", params={"status": "backordered"}).json()
    assert {p["part_id"] for p in backordered["parts"]} == {"ELC-3021", "ELC-3030"}


# ---------------------------------------------------------------- inventory: errors
def test_unknown_part_returns_404():
    r = client.get("/inventory/BRK-9999")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "PART_NOT_FOUND"


@pytest.mark.parametrize("bad_id", ["BRK1020", "BRAKE-1020", "BRK-10", "1020"])
def test_malformed_part_id_returns_422(bad_id):
    r = client.get(f"/inventory/{bad_id}")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_PART_ID"


def test_unknown_warehouse_returns_422():
    r = client.get("/inventory/BRK-1001", params={"warehouse": "WH-MARS"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_WAREHOUSE"


def test_invalid_category_returns_422():
    r = client.get("/inventory", params={"category": "tyres"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_REQUEST"


# ---------------------------------------------------------------- orders
def test_get_shipped_order():
    body = client.get("/orders/ORD-10482").json()
    assert body["status"] == "Shipped"
    assert body["tracking_number"]
    assert body["can_cancel_free_of_charge"] is False


def test_placed_order_can_be_cancelled_free():
    assert client.get("/orders/ORD-10503").json()["can_cancel_free_of_charge"] is True


def test_delayed_order_has_revised_date_and_reason():
    body = client.get("/orders/ORD-10388").json()
    assert body["is_delayed"] is True
    assert body["revised_delivery"] == "2026-10-21"
    assert body["delay_reason"]


def test_vehicle_order_items_use_model_not_part_id():
    item = client.get("/orders/ORD-10290").json()["items"][0]
    assert item["model"] == "SUV Touring" and item["part_id"] is None


def test_unknown_order_returns_404():
    r = client.get("/orders/ORD-99999")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "ORDER_NOT_FOUND"


@pytest.mark.parametrize("bad_id", ["10482", "ORD-123", "ORDER-10482"])
def test_malformed_order_id_returns_422(bad_id):
    r = client.get(f"/orders/{bad_id}")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "INVALID_ORDER_ID"


# ---------------------------------------------------------------- failure simulation
@pytest.mark.parametrize("path", ["/inventory/BRK-1001", "/orders/ORD-10482", "/inventory"])
def test_simulated_outage_returns_503(path):
    r = client.get(path, params={"simulate": "error"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


def test_simulated_slow_response(monkeypatch):
    import time
    monkeypatch.setenv("MOCK_SLOW_SECONDS", "0.3")
    start = time.perf_counter()
    r = client.get("/inventory/BRK-1001", params={"simulate": "slow"})
    assert r.status_code == 200
    assert time.perf_counter() - start >= 0.3
