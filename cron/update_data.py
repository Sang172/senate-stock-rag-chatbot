import subprocess
import time

def main():
    # Run scraper
    print("Starting scraper...")
    subprocess.run(["python", "scrape.py"])
    
    # Wait for scraper to finish and run processor
    print("Starting processor...")
    subprocess.run(["python", "process.py"])

if __name__ == "__main__":
    main()