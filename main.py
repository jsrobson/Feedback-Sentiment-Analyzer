from utils import CSVLoader
from processor import Parser

TEST_PATH = "data/Apple Magic Mouse (Technology) - Amazon Product Reviews.csv"

def main():
    dl = CSVLoader(TEST_PATH)
    df = dl.load()
    p = Parser(df)
    p.run()

if __name__ == '__main__':
    main()
