import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="Ram",
  password="holmes@00",
  database="Project"
)
mycursor=mydb.cursor()
#mycursor.execute("CREATE DATABASE Project")
#Before executing

#mycursor.execute("CREATE TABLE vehicle_count (id INT AUTO_INCREMENT PRIMARY KEY, Category VARCHAR(255), Count VARCHAR(255))")
#sql = "INSERT INTO vehicle_count (Category, Count) VALUES (%s, %s)"
#val = [("TotalCount", "0"),("UnIndetified", "0"),("Car", "0"),("Mortorbike", "0"),("Bus", "0"),("Truck", "0")]
#mycursor.executemany(sql, val)
#mydb.commit()
def upd(a,c):
    mycursor=mydb.cursor()
    sql = "UPDATE vehicle_count SET Count = %s WHERE id = %s"
    val=(a,c)
    mycursor.execute(sql,val)
    mydb.commit()

mycursor.execute("SELECT * FROM vehicle_count")
myresult = mycursor.fetchall()
for x in myresult:
  print(x)



#Before tomorrow session
#mycursor.execute("DROP TABLE vehicle_count")
#mycursor.execute("SHOW TABLES")
#for x in mycursor:
#  print(x)
