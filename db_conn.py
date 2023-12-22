# freshman 작업
import pymysql


from pymysql.constants.CLIENT import MULTI_STATEMENTS


def open_db(dbname="DS2023"):
    conn = pymysql.connect(
        host="localhost",
        user="root",
        passwd="Fghghh1515!",
        db="score",
        client_flag=MULTI_STATEMENTS,
        charset="utf8mb4",
    )

    cursor = conn.cursor(pymysql.cursors.DictCursor)

    return conn, cursor


def close_db(conn, cur):
    cur.close()
    conn.close()


if __name__ == "__main__":
    conn, cur = open_db()
    close_db(conn, cur)
