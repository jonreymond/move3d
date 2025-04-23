from bs4 import BeautifulSoup
from datetime import datetime


def read_time(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    # Passing the stored data inside
    Bs_data = BeautifulSoup(data, "xml")
    b_name = Bs_data.find('CreationDate')
    value = b_name['value']
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # fallback: return the raw string
        return value