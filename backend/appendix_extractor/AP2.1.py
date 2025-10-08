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
print("Policy Document Scraper - Starting...")
print("="*60)

# Hardcoded URL
url = "https://samm.dsca.mil/appendix/appendix-2-procedures"
print(f"\nUsing URL: {url}")

# Setup Edge options
edge_options = Options()
edge_options.add_argument('--start-maximized')

# Path to the Edge driver
driver_path = os.path.join(os.getcwd(), 'msedgedriver.exe')

if not os.path.exists(driver_path):
    print(f"\n✗ ERROR: Edge driver not found at: {driver_path}")
    exit()

# Initialize the driver
print("\n[1/4] Initializing Edge browser...")
driver = webdriver.Edge(
    service=Service(driver_path), 
    options=edge_options
)

try:
    # Navigate to the webpage
    print(f"[2/4] Opening URL: {url}")
    driver.get(url)
    
    # Wait for page to load
    print("[3/4] Waiting for page to fully load...")
    time.sleep(5)
    
    # Wait for content to be present
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "PNumber"))
    )
    print("     ✓ Page content loaded!")
    
    # Extract sections
    print("[4/4] Extracting sections...")
    
    # Find all section numbers
    section_numbers = driver.find_elements(By.CLASS_NAME, "PNumber")
    print(f"     ✓ Found {len(section_numbers)} section numbers")
    
    data = []
    
    for idx, section_elem in enumerate(section_numbers, start=1):
        try:
            # Get section number
            section_number = section_elem.text.strip()
            
            # Get the parent element
            parent = section_elem.find_element(By.XPATH, "..")
            
            # Try to find PTitle in the same parent
            section_title = ""
            try:
                title_elem = parent.find_element(By.CLASS_NAME, "PTitle")
                section_title = title_elem.text.strip()
            except:
                # If no PTitle found, leave it empty
                pass
            
            # Get all text from parent
            full_text = parent.text.strip()
            
            # Remove section number and title from the text to get just content
            section_text = full_text
            if section_text.startswith(section_number):
                section_text = section_text[len(section_number):].strip()
            if section_title and section_text.startswith(section_title):
                section_text = section_text[len(section_title):].strip()
            
            # Clean up - remove leading periods or spaces
            section_text = section_text.lstrip('. ')
            
            # Only add if section has meaningful text (more than 10 characters)
            if len(section_text) > 10:
                data.append({
                    'Section_Number': section_number,
                    'Section_Title': section_title,
                    'Section_Text': section_text
                })
            
            # Print progress
            if idx % 10 == 0:
                print(f"     Processed {idx} sections...")
        
        except Exception as e:
            print(f"     Warning: Error processing section {idx}: {e}")
            continue
    
    print(f"     ✓ Initially extracted {len(data)} sections")
    
    # Remove duplicates - keep entry with longest text for each section number
    print("     Removing duplicates...")
    seen_sections = {}
    for item in data:
        section_num = item['Section_Number']
        if section_num not in seen_sections:
            seen_sections[section_num] = item
        else:
            # Keep the one with more text
            if len(item['Section_Text']) > len(seen_sections[section_num]['Section_Text']):
                seen_sections[section_num] = item
    
    # Convert back to list
    data = list(seen_sections.values())
    print(f"     ✓ After removing duplicates: {len(data)} unique sections")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = "policy_sections.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"     ✓ Data saved to: {output_file}")
    
    # Display summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total sections extracted: {len(df)}")
    print(f"\nFirst 10 sections preview:")
    print(df.head(10).to_string())
    
    # Keep browser open for 10 seconds
    print("\nBrowser will close in 10 seconds...")
    time.sleep(10)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    driver.quit()
    print("Done!")