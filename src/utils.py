from PIL import Image
import requests

def load_image(url):
    return Image.open(requests.get(url, stream=True).raw)

