"""Dashboard API access kept separate from model and database logic for testability."""
import requests


def load_dashboard_data(api_url, get=requests.get):
    try:
        summary_response = get(api_url + "/risk/summary", timeout=4)
        summary_response.raise_for_status()
        products_response = get(api_url + "/products", params={"limit": 200}, timeout=4)
        products_response.raise_for_status()
        metrics_response = get(api_url + "/metrics", timeout=4)
        metrics = metrics_response.json() if metrics_response.ok else {}
        return summary_response.json(), metrics, products_response.json().get("items", []), None
    except requests.RequestException as exc:
        return {}, {}, [], f"API unavailable: {exc}"
