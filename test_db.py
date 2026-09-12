from db import get_connection
import mysql.connector

try:
	conn = get_connection()
	print("Connection successful!")
	conn.close()
except mysql.connector.Error as error:
	print(f"Connection failed: {error}")