from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os

print("="*60)
print("DSCA Data Scraper - Starting...")
print("="*60)

# Setup Edge options
edge_options = Options()
# edge_options.add_argument('--headless')  # Uncomment later to run without showing browser
edge_options.add_argument('--start-maximized')  # Start browser maximized

# Path to the Edge driver
driver_path = os.path.join(os.getcwd(), 'msedgedriver.exe')

# Check if driver exists
if not os.path.exists(driver_path):
    print(f"\n✗ ERROR: Edge driver not found at: {driver_path}")
    print("\nPlease follow these steps:")
    print("1. Check your Edge version: Edge menu → Help and feedback → About Microsoft Edge")
    print("2. Download matching driver from: https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/")
    print("3. Extract msedgedriver.exe to your project folder")
    print("4. Run this script again")
    exit()

# Initialize the driver
print("\n[1/5] Initializing Edge browser...")
print(f"Using driver at: {driver_path}")
driver = webdriver.Edge(
    service=Service(driver_path), 
    options=edge_options
)

try:
    # Navigate to the DSCA website
    url = "https://samm.dsca.mil/appendix/appendix-1-all"
    print(f"[2/5] Opening URL: {url}")
    driver.get(url)
    
    # Wait for the table to load
    print("[3/5] Waiting for page to fully load...")
    time.sleep(5)  # Give extra time for JavaScript to render
    
    # Wait for table to be present
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.TAG_NAME, "table"))
    )
    print("     ✓ Table found!")
    
    # Find the table
    print("[4/5] Extracting data from table...")
    table = driver.find_element(By.TAG_NAME, "table")
    
    # Find all rows in the table body
    rows = table.find_elements(By.TAG_NAME, "tr")
    print(f"     ✓ Found {len(rows)} rows (including header)")
    
    # Extract data
    data = []
    
    # Skip the header row (first row) and process data rows
    for index, row in enumerate(rows[1:], start=1):  # Skip header
        try:
            cells = row.find_elements(By.TAG_NAME, "td")
            
            if len(cells) >= 11:  # Make sure row has enough columns
                row_data = {
                    'USML': cells[0].text.strip(),
                    'CATEGORY': cells[1].text.strip(),
                    'SYSTEM': cells[2].text.strip(),
                    'MODEL': cells[3].text.strip(),
                    'ITEM': cells[4].text.strip(),
                    'MASL': cells[5].text.strip(),
                    'A1_NC_CHARGE': cells[6].text.strip(),
                    'A1_PREVIOUS_NC_CHARGE': cells[7].text.strip() if len(cells) > 7 else '',
                    'EFFECTIVE_DATE': cells[8].text.strip() if len(cells) > 8 else '',
                    'EXPIRATION_DATE': cells[9].text.strip() if len(cells) > 9 else '',
                    'POLICY_MEMO': cells[10].text.strip() if len(cells) > 10 else '',
                    'NOTES': cells[11].text.strip() if len(cells) > 11 else ''
                }
                data.append(row_data)
                
                # Print progress every 100 rows
                if index % 100 == 0:
                    print(f"     Processed {index} rows...")
        
        except Exception as e:
            print(f"     Warning: Error processing row {index}: {e}")
            continue
    
    print(f"     ✓ Successfully extracted {len(data)} rows of data")
    
    # Convert to pandas DataFrame
    print("[5/5] Converting to DataFrame and saving to CSV...")
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = "dsca_data.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"     ✓ Data saved to: {output_file}")
    
    # Display summary
    print("\n" + "="*60)
    print("SCRAPING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total records extracted: {len(df)}")
    print(f"\nFirst few rows preview:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("Please check if the website is accessible and try again.")

finally:
    # Close the browser
    print("\nClosing browser...")
    driver.quit()
    print("Done!")