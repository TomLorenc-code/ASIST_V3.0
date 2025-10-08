from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os
import re

print("="*60)
print("Transportation Cost Tables Scraper - Starting...")
print("="*60)

# Direct URL to Appendix 2 page
base_url = "https://samm.dsca.mil/appendix/appendix-2"

# Setup Edge options
edge_options = Options()
edge_options.add_argument('--start-maximized')

# Path to the Edge driver
driver_path = os.path.join(os.getcwd(), 'msedgedriver.exe')

if not os.path.exists(driver_path):
    print(f"\n✗ ERROR: Edge driver not found at: {driver_path}")
    exit()

# Initialize the driver
print("\n[Phase 1/3] Initializing Edge browser...")
driver = webdriver.Edge(
    service=Service(driver_path), 
    options=edge_options
)

all_data = []

try:
    # Navigate to the main page
    print(f"[Phase 1/3] Opening Appendix 2 page...")
    driver.get(base_url)
    time.sleep(5)
    
    # Find all year links
    print("[Phase 1/3] Finding year links in AP2.2 section only...")
    
    all_links = driver.find_elements(By.TAG_NAME, "a")
    year_links = []
    
    for link in all_links:
        link_text = link.text.strip()
        href = link.get_attribute('href')
        
        # Only include links that match CY year pattern
        if re.match(r'CY\s*\d{4}', link_text) and href:
            # Skip procedures link and any AP2.1 related links
            if 'procedure' in href.lower() or 'ap2.1' in href.lower():
                print(f"       Skipping: {link_text} (filtered out)")
                continue
            
            year_links.append({
                'year': link_text,
                'url': href
            })
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_year_links = []
    for yl in year_links:
        if yl['url'] not in seen_urls:
            seen_urls.add(yl['url'])
            unique_year_links.append(yl)
    
    year_links = unique_year_links
    
    print(f"     ✓ Found {len(year_links)} unique year links (AP2.2 only)")
    for yl in year_links:
        print(f"       - {yl['year']}: {yl['url']}")
    
    # Phase 2: Loop through each year and extract tables
    print(f"\n[Phase 2/3] Extracting data from each year...")
    
    for idx, year_info in enumerate(year_links, start=1):
        year = year_info['year']
        url = year_info['url']
        
        print(f"\n  [{idx}/{len(year_links)}] Processing {year}...")
        
        try:
            # Navigate to year page
            driver.get(url)
            time.sleep(3)
            
            # Wait for table to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            
            # Find all tables on the page (skip the last one - it's footer links)
            tables = driver.find_elements(By.TAG_NAME, "table")[:-1]
            print(f"       ✓ Found {len(tables)} data table(s)")
            
            # Extract data from tables
            for table_idx, table in enumerate(tables):
                rows = table.find_elements(By.TAG_NAME, "tr")
                
                current_category = ""
                
                for row_idx, row in enumerate(rows):
                    # Get both td and th cells
                    td_cells = row.find_elements(By.TAG_NAME, "td")
                    th_cells = row.find_elements(By.TAG_NAME, "th")
                    
                    # Skip header row (has multiple th cells)
                    if len(th_cells) > 1:
                        continue
                    
                    # Check if this is a category header (single th cell)
                    if len(th_cells) == 1 and len(td_cells) == 0:
                        category_text = th_cells[0].text.strip()
                        if category_text:
                            current_category = category_text
                        continue
                    
                    # Data rows have td cells (5 cells: empty + NSN + Item + Code8 + Code9)
                    if len(td_cells) == 5:
                        try:
                            # Skip the first empty cell, get the rest
                            nsn = td_cells[1].text.strip()
                            item = td_cells[2].text.strip()
                            code8 = td_cells[3].text.strip()
                            code9 = td_cells[4].text.strip()
                            
                            # Only add if we have meaningful data
                            if nsn and item:
                                all_data.append({
                                    'Year': year,
                                    'Category': current_category,
                                    'NSN': nsn,
                                    'Item': item,
                                    'Code_8_Estimated_Actual_Total': code8,
                                    'Code_9_Estimated_Actual_Total': code9
                                })
                        except Exception as e:
                            print(f"         Warning: Error on row {row_idx}: {e}")
                            continue
                
                print(f"       ✓ Processed table {table_idx + 1}")
            
            print(f"       ✓ Total rows collected so far: {len(all_data)}")
            
        except Exception as e:
            print(f"       ✗ Error processing {year}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Phase 3: Save all data
    print(f"\n[Phase 3/3] Saving all data...")
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Save to CSV
        output_file = "AP2.2updated.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"     ✓ Data saved to: {output_file}")
        
        # Display summary
        print("\n" + "="*60)
        print("EXTRACTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total records extracted: {len(df)}")
        print(f"\nRecords by year:")
        year_counts = df['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}: {count} records")
        print(f"\nCategories found: {df['Category'].nunique()}")
        print(f"Sample categories: {list(df['Category'].unique()[:5])}")
        
    else:
        print("     ✗ No data was extracted")
    
    print("\nBrowser will close in 10 seconds...")
    time.sleep(10)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    driver.quit()
    print("Done!")