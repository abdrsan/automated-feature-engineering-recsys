import sys
from scripts.run_study import main

if __name__ == "__main__":
    main(["--stage", "baseline", *sys.argv[1:]])
