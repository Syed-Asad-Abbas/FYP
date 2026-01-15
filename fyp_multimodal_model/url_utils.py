"""
URL Utilities
Helper functions for URL handling and validation
"""

import requests
import urllib3

# Suppress insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def is_url_alive(url, timeout=5):
    """
    Check if a URL is reachable (alive).
    
    Args:
        url: URL to check
        timeout: Request timeout in seconds
        
    Returns:
        bool: True if reachable, False if dead (DNS error, timeout, 404)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        # Try HEAD request first (faster)
        response = requests.head(
            url, 
            headers=headers, 
            timeout=timeout, 
            verify=False,
            allow_redirects=True
        )
        
        # 404 is definitely dead
        if response.status_code == 404:
            return False
            
        # 405 Method Not Allowed -> Try GET
        if response.status_code == 405:
            response = requests.get(
                url, 
                headers=headers, 
                timeout=timeout, 
                verify=False,
                stream=True # Don't download content
            )
            response.close()
            if response.status_code == 404:
                return False
                
        return True
        
    except (requests.ConnectionError, requests.Timeout):
        return False
    except Exception as e:
        # For other exceptions (e.g. invalid URL schema), assume dead or invalid
        print(f"[URL Check] Error checking {url}: {e}")
        return False
