import sqlite3
import config
from typing import List, Tuple


class IndexingDB(object):
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()

    def create_tables(self):
        table_name = 'term_title_table'
        c = self.cursor
        c.execute(f"CREATE TABLE {table_name} ("
                  f"term TEXT NOT NULL, "
                  f"article_title TEXT NOT NULL, "
                  f"occurrence INTEGER NOT NULL, "
                  f"tf_idf_score REAL"
                  f");")
        c.execute(f"CREATE INDEX term_index ON {table_name}(term);")
        c.execute(f"CREATE INDEX article_title_index ON {table_name}(article_title);")
        c.execute(f"CREATE INDEX term_title_index ON {table_name}(term, article_title);")

        article_table_name = 'title_table'
        c.execute(f"CREATE TABLE {article_table_name} ("
                  f"article_title TEXT NOT NULL PRIMARY KEY, "
                  f"gram_number INTEGER NOT NULL);")
        self.conn.commit()

    def insert_item(self, term: str, article_title: str, occurrence: int):
        table_name = 'term_title_table'
        self.cursor.execute(f"INSERT INTO {table_name}(term, article_title, occurrence) VALUES (?, ?, ?)",
                            (term, article_title, occurrence))
        self.conn.commit()

    def insert_many_items(self, items: List[Tuple[str, str, int]]):
        table_name = 'term_title_table'
        self.cursor.executemany(f"INSERT INTO {table_name}(term, article_title, occurrence) VALUES (?, ?, ?)",
                                items)
        self.conn.commit()

    def insert_article(self, article_title: str, gram_number: int):
        table_name = 'title_table'
        self.cursor.execute(f"INSERT INTO {table_name}(article_title, gram_number) VALUES (?, ?)",
                            (article_title, gram_number))
        self.conn.commit()

    def insert_many_articles(self, items: List[Tuple[str, int]]):
        table_name = 'title_table'
        self.cursor.executemany(f"INSERT INTO {table_name}(article_title, gram_number) VALUES (?, ?)",
                                items)
        self.conn.commit()

    def assign_score(self, term: str, article_title: str,
                     score: float, score_field: str = 'tf_idf_score'):
        self.cursor.execute(f"UPDATE term_title_table SET {score_field}=? where term=? AND article_title=?"
                            , (score, term, article_title))
        self.conn.commit()

    def assign_many_scores(self, items: List[Tuple[float, str, str]], score_field: str = 'tf_idf_score'):
        self.cursor.executemany(f"UPDATE term_title_table SET {score_field}=? where term=? AND article_title=?"
                                , items)
        # for value in self.cursor:
        #     print(value)
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.commit()
        self.conn.close()