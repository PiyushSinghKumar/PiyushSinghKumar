from dotenv import load_dotenv
import os

load_dotenv()
KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY")
KRAKEN_SECRET = os.getenv("KRAKEN_SECRET")
