from unittest.mock import MagicMock, patch

from graph.node import SearchNode


def test_search_node_delegates_to_tool():
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = {
        "status": "ok",
        "summary": "fake result",
        "data": [],
        "debug_hint": None,
    }

    with patch("graph.node.search_stock_info", mock_tool):
        node = SearchNode(num_results=5, hl="zh-TW", gl="tw")
        out = node({"query": "AAPL growth", "debug": {}})

    mock_tool.invoke.assert_called_once_with({
        "query": "AAPL growth",
        "num_results": 5,
        "hl": "zh-TW",
        "gl": "tw",
    })
    assert out["search_results"]["status"] == "ok"
    assert out["debug"]["search_output"] == "fake result"


def test_search_node_propagates_error_status():
    mock_tool = MagicMock()
    mock_tool.invoke.return_value = {
        "status": "error",
        "summary": "Search failed: HTTP 500",
        "data": None,
        "debug_hint": None,
    }

    with patch("graph.node.search_stock_info", mock_tool):
        node = SearchNode()
        out = node({"query": "any query", "debug": {}})

    assert out["search_results"]["status"] == "error"
    assert "500" in out["debug"]["search_output"]
