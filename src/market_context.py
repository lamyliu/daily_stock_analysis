# -*- coding: utf-8 -*-
"""
Market context detection for LLM prompts.

Detects the market (A-shares, HK, US, JP, KR) from a stock code and returns
market-specific role descriptions so prompts are not hardcoded to a
single market.

Fixes: https://github.com/ZhuLinsen/daily_stock_analysis/issues/644
"""

import re
from typing import Optional


def detect_market(stock_code: Optional[str]) -> str:
    """Detect market from stock code.

    Returns:
        One of 'cn', 'hk', 'us', 'jp', 'kr', or 'cn' as fallback.
    """
    if not stock_code:
        return "cn"

    code = stock_code.strip().upper()

    # JP stocks: 7203.T (Tokyo Stock Exchange)
    if code.endswith(".T") and code[:-2].isdigit():
        return "jp"

    # KR stocks: 005930.KS (KOSPI), 247540.KQ (KOSDAQ)
    if (code.endswith(".KS") or code.endswith(".KQ")) and code.split(".")[0].isdigit():
        return "kr"

    # HK stocks: HK00700, 00700.HK, or 5-digit pure numbers
    if code.startswith("HK") or code.endswith(".HK"):
        return "hk"
    lower = code.lower()
    if lower.endswith(".hk"):
        return "hk"
    # 5-digit pure numbers are HK (A-shares are 6-digit)
    if code.isdigit() and len(code) == 5:
        return "hk"

    # US stocks: 1-5 uppercase letters (AAPL, TSLA, GOOGL)
    # Also handles suffixed forms like BRK.B
    if re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$', code):
        return "us"

    # Default: A-shares (6-digit numbers like 600519, 000001)
    return "cn"


# -- Market-specific role descriptions --

_MARKET_ROLES = {
    "cn": {
        "zh": " A 股",
        "en": "China A-shares",
    },
    "hk": {
        "zh": "港股",
        "en": "Hong Kong stock",
    },
    "us": {
        "zh": "美股",
        "en": "US stock",
    },
    "jp": {
        "zh": "日股",
        "en": "Japanese stock",
    },
    "kr": {
        "zh": "韩股",
        "en": "Korean stock",
    },
}

_MARKET_GUIDELINES = {
    "cn": {
        "zh": (
            "- 本次分析对象为 **A 股**（中国沪深交易所上市股票）。\n"
            "- 请关注 A 股特有的涨跌停机制（±10%/±20%/±30%）、T+1 交易制度及相关政策因素。"
        ),
        "en": (
            "- This analysis covers a **China A-share** (listed on Shanghai/Shenzhen exchanges).\n"
            "- Consider A-share-specific rules: daily price limits (±10%/±20%/±30%), T+1 settlement, and PRC policy factors."
        ),
    },
    "hk": {
        "zh": (
            "- 本次分析对象为 **港股**（香港交易所上市股票）。\n"
            "- 港股无涨跌停限制，支持 T+0 交易，需关注港币汇率、南北向资金流及联交所特有规则。"
        ),
        "en": (
            "- This analysis covers a **Hong Kong stock** (listed on HKEX).\n"
            "- HK stocks have no daily price limits, allow T+0 trading. Consider HKD FX, Southbound/Northbound flows, and HKEX-specific rules."
        ),
    },
    "us": {
        "zh": (
            "- 本次分析对象为 **美股**（美国交易所上市股票）。\n"
            "- 美股无涨跌停限制（但有熔断机制），支持 T+0 交易和盘前盘后交易，需关注美元汇率、美联储政策及 SEC 监管动态。"
        ),
        "en": (
            "- This analysis covers a **US stock** (listed on NYSE/NASDAQ).\n"
            "- US stocks have no daily price limits (but have circuit breakers), allow T+0 and pre/after-market trading. Consider USD FX, Fed policy, and SEC regulations."
        ),
    },
    "jp": {
        "zh": (
            "- 本次分析对象为 **日股**（东京证券交易所上市股票）。\n"
            "- 日股有涨跌停限制（根据股价区间不同），支持 T+3 结算，需关注日元汇率、日银政策及企业治理改革。"
        ),
        "en": (
            "- This analysis covers a **Japanese stock** (listed on Tokyo Stock Exchange).\n"
            "- Japanese stocks have price limit rules (varies by price range), T+3 settlement. Consider JPY FX, BOJ policy, and corporate governance reforms."
        ),
    },
    "kr": {
        "zh": (
            "- 本次分析对象为 **韩股**（韩国交易所上市股票）。\n"
            "- 韩股涨跌停限制为 ±30%，支持 T+2 结算，需关注韩元汇率、韩国央行政策及半导体产业周期。"
        ),
        "en": (
            "- This analysis covers a **Korean stock** (listed on KRX).\n"
            "- Korean stocks have ±30% daily price limits, T+2 settlement. Consider KRW FX, BOK policy, and semiconductor industry cycles."
        ),
    },
}


def get_market_role(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific role description for LLM prompt."""
    market = detect_market(stock_code)
    lang_key = "en" if lang == "en" else "zh"
    return _MARKET_ROLES.get(market, _MARKET_ROLES["cn"])[lang_key]


def get_market_guidelines(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific analysis guidelines for LLM prompt."""
    market = detect_market(stock_code)
    lang_key = "en" if lang == "en" else "zh"
    return _MARKET_GUIDELINES.get(market, _MARKET_GUIDELINES["cn"])[lang_key]
