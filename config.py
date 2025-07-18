import os

from dotenv import load_dotenv

load_dotenv()

DEVICE = 'cuda:0'
TRAIN_SIZES = [200, 400, 600, 800, 1000]
BERT_TRAIN_SIZE = 15000
TEST_SIZE = 2000
SEED = 1
BOT_TOKEN = os.getenv('BOT_TOKEN')
DB_URL = os.getenv('DB_URL') or 'sqlite:///quotes.db'
