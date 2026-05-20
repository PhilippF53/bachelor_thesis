import sqlite3

class DBService:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
    
    def get_database_schema(self) -> str:
        pass
    
    def execute_sql(self, sql_statement: str) -> None:
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            result = cursor.execute(sql_statement)
            
        return result