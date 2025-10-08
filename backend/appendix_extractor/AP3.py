from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import time
import os

print("="*60)
print("Appendix 3 PDF Downloader - Starting...")
print("="*60)

# Create folder for PDFs
pdf_folder = "Appendix3_PDFs"
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)
    print(f"Created folder: {pdf_folder}")

# Setup Edge with download preferences
driver_path = os.path.join(os.getcwd(), 'msedgedriver.exe')
edge_options = Options()

# Set download directory
prefs = {
    "download.default_directory": os.path.abspath(pdf_folder),
    "download.prompt_for_download": False,
    "plugins.always_open_pdf_externally": True
}
edge_options.add_experimental_option("prefs", prefs)
edge_options.add_argument('--start-maximized')

driver = webdriver.Edge(service=Service(driver_path), options=edge_options)

try:
    url = "https://samm.dsca.mil/appendix/appendix-3"
    print(f"\n[1/3] Opening: {url}")
    driver.get(url)
    time.sleep(5)
    
    # Find ALL tables on the page
    print("[2/3] Extracting document information from all tables...")
    tables = driver.find_elements(By.TAG_NAME, "table")
    print(f"     Found {len(tables)} tables on the page")
    
    data = []
    
    # Process each table
    for table_idx, table in enumerate(tables, start=1):
        rows = table.find_elements(By.TAG_NAME, "tr")
        print(f"     Table {table_idx}: {len(rows)-1} rows")
        
        for idx, row in enumerate(rows[1:], start=1):  # Skip header
            cells = row.find_elements(By.TAG_NAME, "td")
            
            if len(cells) >= 2:
                section = cells[0].text.strip()
                
                # Get the link element
                try:
                    link_elem = cells[1].find_element(By.TAG_NAME, "a")
                    title = link_elem.text.strip()
                    pdf_url = link_elem.get_attribute('href')
                    
                    # Create clean filename
                    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
                    safe_title = safe_title[:80]
                    filename = f"{section.replace('.', '_')}_{safe_title}.pdf"
                    
                    data.append({
                        'Section': section,
                        'Title': title,
                        'PDF_URL': pdf_url,
                        'Filename': filename,
                        'Link_Element': link_elem
                    })
                    
                except:
                    continue
    
    print(f"     ✓ Extracted {len(data)} total document records from all tables")
    
    # Download PDFs by clicking links
    print(f"\n[3/3] Downloading {len(data)} PDFs via browser...")
    print("     (This may take several minutes...)")
    
    for idx, doc in enumerate(data, start=1):
        print(f"     [{idx}/{len(data)}] Downloading {doc['Section']}...")
        
        try:
            # Get current files in download folder
            files_before = set(os.listdir(pdf_folder))
            
            # Click the link to trigger download
            doc['Link_Element'].click()
            
            # Wait for download to complete
            max_wait = 30
            waited = 0
            while waited < max_wait:
                time.sleep(1)
                waited += 1
                files_after = set(os.listdir(pdf_folder))
                new_files = files_after - files_before
                
                # Check for complete files
                complete_files = [f for f in new_files if not f.endswith(('.crdownload', '.tmp', '.part'))]
                
                if complete_files:
                    downloaded_file = complete_files[0]
                    old_path = os.path.join(pdf_folder, downloaded_file)
                    new_path = os.path.join(pdf_folder, doc['Filename'])
                    
                    # Rename to clean filename
                    try:
                        if os.path.exists(old_path):
                            # If target already exists, remove it first
                            if os.path.exists(new_path):
                                os.remove(new_path)
                            os.rename(old_path, new_path)
                            doc['Local_File_Path'] = new_path
                            print(f"          ✓ Saved: {doc['Filename']}")
                        break
                    except Exception as e:
                        doc['Local_File_Path'] = old_path
                        print(f"          ✓ Saved: {downloaded_file}")
                        break
            else:
                print(f"          ✗ Timeout waiting for download")
                doc['Local_File_Path'] = "DOWNLOAD_TIMEOUT"
            
            time.sleep(2)
            
        except Exception as e:
            print(f"          ✗ Error: {e}")
            doc['Local_File_Path'] = "DOWNLOAD_FAILED"
    
    # Close browser
    driver.quit()
    
    # Save catalog to CSV
    df_data = [{k: v for k, v in doc.items() if k != 'Link_Element'} for doc in data]
    df = pd.DataFrame(df_data)
    catalog_file = "AP3.1updated.csv"
    df.to_csv(catalog_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETED!")
    print("="*60)
    print(f"Total documents: {len(df)}")
    print(f"PDFs saved to: {pdf_folder}/")
    print(f"Catalog saved to: {catalog_file}")
    
    successful = len([d for d in data if 'Local_File_Path' in d and d['Local_File_Path'] not in ["DOWNLOAD_FAILED", "DOWNLOAD_TIMEOUT"]])
    print(f"\nSuccessfully downloaded: {successful}/{len(data)}")
    
    if successful < len(data):
        print(f"Failed/Timeout downloads: {len(data) - successful}")
    
    print("\nAll documents:")
    print(df[['Section', 'Title']].to_string())

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    try:
        driver.quit()
    except:
        pass
    print("\nDone!")