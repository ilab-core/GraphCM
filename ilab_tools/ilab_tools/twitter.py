import requests


class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["Authorization"] = f"Bearer {self.token}"
        r.headers["User-Agent"] = "v2FullArchiveSearchPython"
        return r


def twitter_api(params, bearer_token, url):
    """Generates an API request to Twitter.

    Args:
        params (dict): parameters for the API request. query, max_results etc.
        bearer_token (string): Twitter API bearer token
        search_url (string): url of the API endpoint

    returns: json formatted response
    """
    response = requests.request("GET", url, auth=BearerAuth(bearer_token).__call__, params=params)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()
