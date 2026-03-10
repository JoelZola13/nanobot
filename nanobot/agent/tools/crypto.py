"""Cryptocurrency tools using CoinGecko free API."""

import json
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Common symbol → CoinGecko ID mappings
_SYMBOL_MAP = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "ada": "cardano",
    "dot": "polkadot",
    "matic": "matic-network",
    "avax": "avalanche-2",
    "link": "chainlink",
    "doge": "dogecoin",
    "shib": "shiba-inu",
    "xrp": "ripple",
    "bnb": "binancecoin",
    "ltc": "litecoin",
    "uni": "uniswap",
    "atom": "cosmos",
    "near": "near",
    "apt": "aptos",
    "arb": "arbitrum",
    "op": "optimism",
    "sui": "sui",
}


def _resolve_coin_id(symbol_or_id: str) -> str:
    """Resolve a symbol or name to a CoinGecko coin ID."""
    s = symbol_or_id.lower().strip()
    return _SYMBOL_MAP.get(s, s)


class CryptoApiTool(Tool):
    """Query cryptocurrency data via CoinGecko."""

    @property
    def name(self) -> str:
        return "crypto_api"

    @property
    def description(self) -> str:
        return (
            "Query cryptocurrency prices, market data, and coin info. "
            "Actions: 'price' (current price), 'info' (coin details), 'market' (top coins by market cap). "
            "Use common symbols like BTC, ETH, SOL."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "API action: 'price', 'info', or 'market'",
                },
                "symbol": {
                    "type": "string",
                    "description": "Cryptocurrency symbol (e.g., BTC, ETH) or CoinGecko ID",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "").strip().lower()
        symbol = kwargs.get("symbol", "").strip()

        if not action:
            return "Error: No action specified. Use 'price', 'info', or 'market'."

        try:
            if action == "price":
                return await self._get_price(symbol)
            elif action == "info":
                return await self._get_info(symbol)
            elif action == "market":
                return await self._get_market()
            else:
                return f"Error: Unknown action '{action}'. Use 'price', 'info', or 'market'."
        except httpx.HTTPStatusError as e:
            return f"Error: CoinGecko API returned HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Error: Request failed: {e}"

    async def _get_price(self, symbol: str) -> str:
        if not symbol:
            return "Error: 'symbol' is required for price action."
        coin_id = _resolve_coin_id(symbol)
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{COINGECKO_BASE}/simple/price",
                params={
                    "ids": coin_id,
                    "vs_currencies": "usd",
                    "include_24hr_change": "true",
                    "include_market_cap": "true",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if coin_id not in data:
            return f"Error: Coin '{symbol}' not found. Try using the full name (e.g., 'bitcoin')."

        info = data[coin_id]
        price = info.get("usd", "N/A")
        change = info.get("usd_24h_change")
        mcap = info.get("usd_market_cap")

        lines = [f"{symbol.upper()} ({coin_id})"]
        lines.append(f"  Price: ${price:,.2f}" if isinstance(price, (int, float)) else f"  Price: {price}")
        if change is not None:
            arrow = "+" if change >= 0 else ""
            lines.append(f"  24h Change: {arrow}{change:.2f}%")
        if mcap:
            lines.append(f"  Market Cap: ${mcap:,.0f}")
        return "\n".join(lines)

    async def _get_info(self, symbol: str) -> str:
        if not symbol:
            return "Error: 'symbol' is required for info action."
        coin_id = _resolve_coin_id(symbol)
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{COINGECKO_BASE}/coins/{coin_id}",
                params={"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false"},
            )
            resp.raise_for_status()
            data = resp.json()

        desc = data.get("description", {}).get("en", "")
        if desc:
            # Strip HTML from description
            import re
            desc = re.sub(r'<[^>]+>', '', desc)[:500]

        market = data.get("market_data", {})
        lines = [
            f"{data.get('name', symbol.upper())} ({data.get('symbol', '').upper()})",
            f"  Rank: #{data.get('market_cap_rank', 'N/A')}",
        ]
        if market.get("current_price", {}).get("usd"):
            lines.append(f"  Price: ${market['current_price']['usd']:,.2f}")
        if market.get("market_cap", {}).get("usd"):
            lines.append(f"  Market Cap: ${market['market_cap']['usd']:,.0f}")
        if market.get("total_volume", {}).get("usd"):
            lines.append(f"  24h Volume: ${market['total_volume']['usd']:,.0f}")
        if market.get("price_change_percentage_24h") is not None:
            lines.append(f"  24h Change: {market['price_change_percentage_24h']:.2f}%")
        if market.get("ath", {}).get("usd"):
            lines.append(f"  All-Time High: ${market['ath']['usd']:,.2f}")
        if desc:
            lines.append(f"\n{desc}")
        return "\n".join(lines)

    async def _get_market(self) -> str:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{COINGECKO_BASE}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 10,
                    "page": 1,
                },
            )
            resp.raise_for_status()
            coins = resp.json()

        if not coins:
            return "No market data available."

        lines = ["Top 10 Cryptocurrencies by Market Cap:\n"]
        for c in coins:
            name = c.get("name", "?")
            sym = c.get("symbol", "?").upper()
            price = c.get("current_price", 0)
            change = c.get("price_change_percentage_24h", 0)
            mcap = c.get("market_cap", 0)
            arrow = "+" if (change or 0) >= 0 else ""
            lines.append(
                f"  {c.get('market_cap_rank', '?'):>2}. {name} ({sym}): "
                f"${price:,.2f}  {arrow}{change:.1f}%  MCap: ${mcap:,.0f}"
            )
        return "\n".join(lines)
