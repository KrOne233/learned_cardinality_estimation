import psycopg2

conn = psycopg2.connect(database="Master_thesis",
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432'
)


def gain_plan(sql_file, conn):
    conn.autocommit = True
    cursor = conn.cursor()
    with open(sql_file) as f:
        lines = f.readlines()
        for line in lines:
            sql = line.split(';')[0]
        cursor.execute(cursor.mogrify('explain analyze ' + sql + ";"))
        analyze_fetched = cursor.fetchall()
        print(analyze_fetched)

# sql = '''SELECT histogram_bounds FROM pg_stats WHERE tablename='black_friday_purchase' AND attname='purchase';'''
# sql = '''SELECT null_frac, n_distinct, most_common_vals, most_common_freqs FROM pg_stats WHERE tablename='black_friday_purchase' AND attname='gender';'''