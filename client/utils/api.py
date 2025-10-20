import requests

LOCAL_BACKEND = 'http://0.0.0.0:8000'

def send_get_request(endpoint, request):
    route = LOCAL_BACKEND + endpoint
    try:
        result = requests.get(route, params=request)
        result.raise_for_status()

        return result.json()
    except requests.RequestException as e:
        return f'request failed: Exception caught: {e}'