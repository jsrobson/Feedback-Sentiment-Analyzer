from utils import CSVLoader
from processor import Parser
from user_interface import UserInterface

TEST_PATH = "data/Apple Magic Mouse (Technology) - Amazon Product Reviews.csv"

def main():
    ui = UserInterface()
    ui.run()

if __name__ == '__main__':
    main()
