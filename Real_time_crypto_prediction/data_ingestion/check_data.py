import sqlite3

DB_PATH = "trades.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT * FROM trades LIMIT 10")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
