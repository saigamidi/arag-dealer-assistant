"""
ARAG Mock API
=============
Simulates the two live systems the assistant's real-time path depends on:

  * Inventory system  -> GET /inventory, GET /inventory/{part_id}
  * Order system      -> GET /orders/{order_id}

All data is synthetic and loaded from JSON files in mock_api/data/.

Failure simulation (for testing the assistant's fallback behaviour):
  * ?simulate=error  -> returns HTTP 503, as if the source system is down
  * ?simulate=slow   -> waits MOCK_SLOW_SECONDS (default 8s) before answering,
                        longer than the assistant's 5s latency target

Run locally:
    uvicorn mock_api.main:app --reload --port 8000
Then open /docs for the interactive Swagger page.
"""

import asyncio
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"

with open(DATA_DIR / "inventory.json", encoding="utf-8") as f:
    INVENTORY = json.load(f)

with open(DATA_DIR / "orders.json", encoding="utf-8") as f:
    ORDERS = json.load(f)

WAREHOUSES: list[str] = INVENTORY["warehouses"]
LOW_STOCK_THRESHOLD: int = INVENTORY["low_stock_threshold"]

# ID formats. These are what the assistant's entity extraction must produce.
PART_ID_PATTERN = re.compile(r"^[A-Z]{3}-\d{4}$")    # e.g. BRK-1020
ORDER_ID_PATTERN = re.compile(r"^ORD-\d{5}$")        # e.g. ORD-10482


# --------------------------------------------------------------------------
# Response models (these also document the API on the /docs page)
# --------------------------------------------------------------------------
class StockStatus(str, Enum):
    in_stock = "in_stock"
    low_stock = "low_stock"
    backordered = "backordered"


class PartStock(BaseModel):
    part_id: str
    description: str
    category: str
    special_order: bool
    status: StockStatus
    total_quantity: int
    quantities_by_warehouse: dict[str, int]
    warehouses_with_stock: list[str]
    estimated_restock_date: Optional[str] = None
    data_as_of: str


class PartSummary(BaseModel):
    part_id: str
    description: str
    category: str
    status: StockStatus
    total_quantity: int


class InventoryList(BaseModel):
    count: int
    parts: list[PartSummary]
    data_as_of: str


class OrderItem(BaseModel):
    part_id: Optional[str] = None
    model: Optional[str] = None
    quantity: int


class Order(BaseModel):
    order_id: str
    dealer_id: str
    order_type: Literal["in_stock_parts", "special_order_parts", "vehicle_factory_order"]
    status: Literal["Placed", "Processing", "Shipped", "Delivered"]
    placed_date: str
    expected_delivery: str
    delivered_date: Optional[str] = None
    tracking_number: Optional[str] = None
    is_delayed: bool
    revised_delivery: Optional[str] = None
    delay_reason: Optional[str] = None
    can_cancel_free_of_charge: bool
    items: list[OrderItem]
    data_as_of: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


# --------------------------------------------------------------------------
# App and consistent error handling
# --------------------------------------------------------------------------
app = FastAPI(
    title="ARAG Mock API",
    description="Synthetic inventory and order systems for the ARAG dealer support assistant.",
    version="1.0.0",
)


class APIError(Exception):
    """Raised by endpoints; converted to a consistent JSON error shape."""

    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    field = ".".join(str(p) for p in first.get("loc", []) if p not in ("query", "path"))
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "INVALID_REQUEST", "message": f"{field}: {first.get('msg')}"}},
    )


ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "ID is well-formed but not found"},
    422: {"model": ErrorResponse, "description": "ID or parameter is malformed"},
    503: {"model": ErrorResponse, "description": "Simulated outage (?simulate=error)"},
}


# --------------------------------------------------------------------------
# Failure simulation (shared by all data endpoints)
# --------------------------------------------------------------------------
async def simulate_failure(
    simulate: Optional[Literal["slow", "error"]] = Query(
        None, description="Testing only: 'error' returns 503, 'slow' delays the response"
    ),
):
    if simulate == "error":
        raise APIError(503, "SERVICE_UNAVAILABLE",
                       "The source system is temporarily unavailable. Try again or check it directly.")
    if simulate == "slow":
        await asyncio.sleep(float(os.getenv("MOCK_SLOW_SECONDS", "8")))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def normalise_id(raw: str) -> str:
    """Accept lower case and stray spaces, e.g. ' brk-1020 ' -> 'BRK-1020'."""
    return raw.strip().upper()


def stock_status(total: int) -> StockStatus:
    if total == 0:
        return StockStatus.backordered
    if total <= LOW_STOCK_THRESHOLD:
        return StockStatus.low_stock
    return StockStatus.in_stock


def build_part_stock(record: dict, warehouse: Optional[str] = None) -> PartStock:
    quantities = record["quantities"]
    if warehouse:
        quantities = {warehouse: quantities[warehouse]}
    total = sum(quantities.values())
    return PartStock(
        part_id=record["part_id"],
        description=record["description"],
        category=record["category"],
        special_order=record["special_order"],
        status=stock_status(total),
        total_quantity=total,
        quantities_by_warehouse=quantities,
        warehouses_with_stock=[w for w, q in quantities.items() if q > 0],
        estimated_restock_date=record.get("estimated_restock_date") if total == 0 else None,
        data_as_of=INVENTORY["data_as_of"],
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "parts": len(INVENTORY["parts"]), "orders": len(ORDERS["orders"])}


@app.get("/inventory", response_model=InventoryList, responses=ERROR_RESPONSES,
         tags=["inventory"], dependencies=[Depends(simulate_failure)])
def list_inventory(
    category: Optional[Literal["brakes", "electrical", "filters_fluids", "accessories"]] = None,
    status: Optional[StockStatus] = None,
):
    """List all parts, optionally filtered by category or stock status."""
    parts = []
    for rec in INVENTORY["parts"].values():
        stock = build_part_stock(rec)
        if category and rec["category"] != category:
            continue
        if status and stock.status != status:
            continue
        parts.append(PartSummary(**stock.model_dump(include=set(PartSummary.model_fields))))
    return InventoryList(count=len(parts), parts=parts, data_as_of=INVENTORY["data_as_of"])


@app.get("/inventory/{part_id}", response_model=PartStock, responses=ERROR_RESPONSES,
         tags=["inventory"], dependencies=[Depends(simulate_failure)])
def get_part_stock(
    part_id: str,
    warehouse: Optional[str] = Query(None, description="Limit to one warehouse, e.g. WH-NORTH"),
):
    """Current stock for one part across all warehouses (or one warehouse)."""
    pid = normalise_id(part_id)
    if not PART_ID_PATTERN.match(pid):
        raise APIError(422, "INVALID_PART_ID",
                       f"'{part_id}' is not a valid part ID. Expected format: ABC-1234 (e.g. BRK-1020).")
    if pid not in INVENTORY["parts"]:
        raise APIError(404, "PART_NOT_FOUND", f"No part with ID {pid} exists in the inventory system.")
    if warehouse is not None:
        warehouse = normalise_id(warehouse)
        if warehouse not in WAREHOUSES:
            raise APIError(422, "INVALID_WAREHOUSE",
                           f"Unknown warehouse '{warehouse}'. Valid options: {', '.join(WAREHOUSES)}.")
    return build_part_stock(INVENTORY["parts"][pid], warehouse)


@app.get("/orders/{order_id}", response_model=Order, responses=ERROR_RESPONSES,
         tags=["orders"], dependencies=[Depends(simulate_failure)])
def get_order(order_id: str):
    """Current status of one order: Placed -> Processing -> Shipped -> Delivered."""
    oid = normalise_id(order_id)
    if not ORDER_ID_PATTERN.match(oid):
        raise APIError(422, "INVALID_ORDER_ID",
                       f"'{order_id}' is not a valid order ID. Expected format: ORD-12345.")
    if oid not in ORDERS["orders"]:
        raise APIError(404, "ORDER_NOT_FOUND", f"No order with ID {oid} exists in the order system.")
    rec = ORDERS["orders"][oid]
    # Per order_status_faq.md: free cancellation only before "Processing".
    return Order(**rec, can_cancel_free_of_charge=rec["status"] == "Placed",
                 data_as_of=ORDERS["data_as_of"])
