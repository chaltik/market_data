import pandas as pd
import psycopg2
import psycopg2.extensions
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

SQL_LIST = lambda symbols: f"({','.join(map(repr, symbols))})"

@contextmanager
def get_connection(params):
    conn = None
    try:
        # Establish the connection
        conn = psycopg2.connect(**params)
        # Create a cursor that returns results as dictionaries
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        # Yield both the connection and cursor
        yield conn, cursor
        conn.commit()
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()
            print("Database connection closed.")

def generate_create_table_sql(df, table_name,schema):
    type_mapping = {
        'int64': 'BIGINT',
        'float64': 'NUMERIC',
        'object': 'TEXT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP',
    }
    
    columns = []
    for column, dtype in df.dtypes.items():
        sql_type = type_mapping.get(str(dtype), 'TEXT')
        columns.append(f'"{column}" {sql_type}')
    
    columns_sql = ',\n    '.join(columns)
    create_table_sql = f"""
CREATE TABLE {schema}.{table_name} (
    {columns_sql}
);
"""
    return create_table_sql

def bcp_to_postgres(csv_file_path, table_name, db_params, schema='public',delim=','):
    with get_connection(db_params) as (conn, cursor):
        try:
            # Open the CSV file
            with open(csv_file_path, 'r') as f:
                # Use COPY command to bulk insert, specifying the schema
                cursor.copy_expert(f"COPY {schema}.{table_name} FROM STDIN WITH CSV HEADER DELIMITER as '{delim}'", f)
            
            conn.commit()
            print(f"Data successfully copied to table '{schema}.{table_name}'")
        except Exception as error:
            print(f"Error: {error}")
            conn.rollback()

def bcp_from_postgres(output_file_path, table_or_view_name, db_params, schema='public', delim=','):
    """
    Export data from a PostgreSQL table or view to a file using COPY command.

    Args:
        output_file_path (str): Path to the output file.
        table_or_view_name (str): Name of the PostgreSQL table or view to export.
        db_params (dict): Database connection parameters (e.g., host, port, dbname, user, password).
        schema (str): Schema name containing the table or view (default: 'public').
        delim (str): Field delimiter for the output file (default: ',').
    """
    with get_connection(db_params) as (conn, cursor):
        try:
            # Open the output file in write mode
            with open(output_file_path, 'w') as f:
                # Use COPY (SELECT ...) to handle both tables and views
                query = f"SELECT * FROM {schema}.{table_or_view_name}"
                cursor.copy_expert(
                    f"COPY ({query}) TO STDOUT WITH CSV HEADER DELIMITER '{delim}'", 
                    f
                )
            
            print(f"Data successfully exported from '{schema}.{table_or_view_name}' to '{output_file_path}'")
        except Exception as error:
            print(f"Error: {error}")
        finally:
            cursor.close()
            conn.close()

def run_sql(query, conn_params, query_params=None):
    """
    Execute SQL query with connection and query parameters
    """
    
    # Configure psycopg2 to return floats instead of Decimal
    DEC2FLOAT = psycopg2.extensions.new_type(
        psycopg2.extensions.DECIMAL.values,
        'DEC2FLOAT',
        lambda value, curs: float(value) if value is not None else None)
    psycopg2.extensions.register_type(DEC2FLOAT)
    
    with psycopg2.connect(**conn_params) as conn:
        with conn.cursor() as curs:
            if query_params is None:
                curs.execute(query)
            else:
                curs.execute(query,query_params)
            # Fetch data and column names
            data = curs.fetchall()
            col_names = [desc[0] for desc in curs.description]
            
    # Create DataFrame
    df = pd.DataFrame(data, columns=col_names)
    return df

def get_db_connection(conn_params):
    return psycopg2.connect(**conn_params)

def run_pd_sql(query, conn_params, query_params=None):
    """
    Execute SQL query using pandas read_sql which is optimized for DataFrame creation
    """
    conn = get_db_connection(conn_params)
    try:
        df = pd.read_sql(query, conn, params=query_params)
        return df
    finally:
        conn.close()


# Function to create a connection to the Neo4j database
# from neo4j import GraphDatabase
# def create_driver(**params):
#     driver = GraphDatabase.driver(params['uri'], auth=(params['user'], params['password']))
#     return driver

# def run_cypher(query, conn_params, query_params=None):
#     driver = create_driver(**conn_params)
#     data = []
#     try:
#         with driver.session(database=conn_params['database']) as session:
#             # Run the query with parameters if provided
#             if query_params:
#                 results = session.run(query, **query_params)
#             else:
#                 results = session.run(query)
#             # Immediately collect the data before the session is closed or result is consumed elsewhere
#             data = [record.data() for record in results]
#     finally:
#         driver.close()
#     return data