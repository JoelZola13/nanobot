"""Finance tools: invoice_generator."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


class InvoiceGeneratorTool(Tool):
    """Generate markdown-formatted invoices."""

    @property
    def name(self) -> str:
        return "invoice_generator"

    @property
    def description(self) -> str:
        return (
            "Generate a formatted invoice from structured data. "
            "Provide client name, line items (JSON array of {description, quantity, unit_price}), "
            "and optional due date. Returns a markdown invoice."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "client": {"type": "string", "description": "Client name"},
                "items": {
                    "type": "string",
                    "description": 'JSON array of line items: [{"description": "...", "quantity": 1, "unit_price": 100.00}]',
                },
                "due_date": {"type": "string", "description": "Invoice due date (YYYY-MM-DD)"},
                "notes": {"type": "string", "description": "Optional notes or payment instructions"},
                "invoice_number": {"type": "string", "description": "Optional invoice number"},
            },
            "required": ["client", "items"],
        }

    async def execute(self, **kwargs: Any) -> str:
        client = kwargs.get("client", "").strip() if isinstance(kwargs.get("client"), str) else str(kwargs.get("client", ""))
        raw_items = kwargs.get("items", "")
        due_date = kwargs.get("due_date", "")
        due_date = due_date.strip() if isinstance(due_date, str) else str(due_date)
        notes = kwargs.get("notes", "")
        notes = notes.strip() if isinstance(notes, str) else str(notes)
        invoice_number = kwargs.get("invoice_number", "")
        invoice_number = invoice_number.strip() if isinstance(invoice_number, str) else str(invoice_number)

        if not client:
            return "Error: No client name provided."
        if not raw_items:
            return "Error: No line items provided."

        # Accept items as a list (from LLM tool calls) or JSON string
        if isinstance(raw_items, list):
            items = raw_items
        elif isinstance(raw_items, str):
            try:
                items = json.loads(raw_items.strip())
            except json.JSONDecodeError as e:
                return f"Error: Invalid JSON for items: {e}"
        else:
            return "Error: Items must be a JSON array or JSON string."

        if not isinstance(items, list):
            return "Error: Items must be a JSON array."

        # Validate and calculate
        line_items = []
        subtotal = 0.0
        for i, item in enumerate(items, 1):
            if not isinstance(item, dict):
                return f"Error: Item {i} must be an object with 'description', 'quantity', 'unit_price'."
            desc = item.get("description", f"Item {i}")
            qty = float(item.get("quantity", 1))
            price = float(item.get("unit_price", 0))
            total = qty * price
            subtotal += total
            line_items.append((desc, qty, price, total))

        today = date.today().isoformat()
        inv_num = invoice_number or f"INV-{datetime.now().strftime('%Y%m%d%H%M')}"

        # Build markdown invoice
        lines = [
            "# INVOICE",
            "",
            f"**Invoice #:** {inv_num}",
            f"**Date:** {today}",
        ]
        if due_date:
            lines.append(f"**Due Date:** {due_date}")
        lines += [
            "",
            f"**Bill To:** {client}",
            "",
            "---",
            "",
            "| # | Description | Qty | Unit Price | Total |",
            "|---|-------------|-----|-----------|-------|",
        ]

        for i, (desc, qty, price, total) in enumerate(line_items, 1):
            qty_str = f"{qty:g}"
            lines.append(f"| {i} | {desc} | {qty_str} | ${price:,.2f} | ${total:,.2f} |")

        lines += [
            "",
            f"**Subtotal:** ${subtotal:,.2f}",
            f"**Total Due:** ${subtotal:,.2f}",
        ]

        if notes:
            lines += ["", "---", "", f"**Notes:** {notes}"]

        return "\n".join(lines)
