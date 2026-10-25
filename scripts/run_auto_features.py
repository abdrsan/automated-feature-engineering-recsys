import sys
from scripts.run_study import main

if __name__ == "__main__":
    main(["--stage", "dev", *sys.argv[1:]])
