import pandas as pd 
import sqlite3
conn = sqlite3.connect('bulletin.db')
#cursor = conn.cursor()
#cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
#print(cursor.fetchall())

query = '''
SELECT SupplierName,
SUM(CASE WHEN ActualDeliveryDate <= ExpectedDeliveryDate THEN 1 ELSE 0 END) AS OnTimeCount,
COUNT(*) AS TotalDeliveries,
(SUM(CASE WHEN ActualDeliveryDate <= ExpectedDeliveryDate THEN 1 ELSE 0 END) * 100.0 /COUNT(*) ) AS OnTimeDelPct 
FROM fact_procurement_transactions 
GROUP BY SupplierName
ORDER BY OnTimeDelPct asc '''

result = pd.read_sql(query,conn)
print(result)

conn.close()