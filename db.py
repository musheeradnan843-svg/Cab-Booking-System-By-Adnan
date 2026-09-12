import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()

def get_connection():
    connection = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        use_pure=True,
        connection_timeout=5
    )
    return connection
