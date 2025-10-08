from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import time
import os

print("="*60)
print("Appendix 5 - Congressional Reports Extractor")
print("="*60)

driver_path = os.path.join(os.getcwd(), 'msedgedriver.exe')
edge_options = Options()
edge_options.add_argument('--start-maximized')

driver = webdriver.Edge(service=Service(driver_path), options=edge_options)

try:
    url = "https://samm.dsca.mil/appendix/appendix-5"
    print(f"\n[1/3] Opening main page...")
    driver.get(url)
    time.sleep(5)
    
    print("[2/3] Finding section links...")
    tables = driver.find_elements(By.TAG_NAME, "table")
    
    section_links = []
    
    for table in tables:
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        for row in rows[1:]:  # Skip header
            cells = row.find_elements(By.TAG_NAME, "td")
            
            if len(cells) >= 2:
                section = cells[0].text.strip()
                
                try:
                    link_elem = cells[1].find_element(By.TAG_NAME, "a")
                    title = link_elem.text.strip()
                    link_url = link_elem.get_attribute('href')
                    
                    if section and link_url:
                        section_links.append({
                            'section': section,
                            'title': title,
                            'url': link_url
                        })
                except:
                    continue
    
    print(f"     ✓ Found {len(section_links)} sections")
    
    print(f"\n[3/3] Extracting data from each section...")
    
    sheets_data = {}
    notes_data = {}
    
    for idx, section_info in enumerate(section_links, start=1):
        section = section_info['section']
        title = section_info['title']
        link_url = section_info['url']
        
        print(f"\n  [{idx}/{len(section_links)}] {section} - {title}")
        
        try:
            driver.get(link_url)
            time.sleep(3)
            
            # Extract notes section first (if exists)
            notes_text = ""
            try:
                # Look for "Notes:" heading
                page_text = driver.find_element(By.TAG_NAME, "body").text
                if "Notes:" in page_text:
                    notes_start = page_text.index("Notes:")
                    notes_text = page_text[notes_start:notes_start+2000]  # Get next 2000 chars
                    notes_data[section] = notes_text
                    print(f"       ✓ Captured notes section")
            except:
                pass
            
            # Find all tables on the page
            detail_tables = driver.find_elements(By.TAG_NAME, "table")
            
            section_data = []
            
            for table in detail_tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                
                if len(rows) < 2:
                    continue
                
                # Get headers
                header_cells = rows[0].find_elements(By.TAG_NAME, "th")
                num_cols = len(header_cells)
                
                # Skip navigation tables
                if num_cols < 3:
                    continue
                
                print(f"       Found table with {num_cols} columns, {len(rows)-1} data rows")
                
                # Get header names
                headers = []
                for header in header_cells:
                    col_name = header.text.strip().replace('\n', ' ').replace('*', '').strip()
                    headers.append(col_name)
                
                # Extract data rows
                for row in rows[1:]:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    if len(cells) >= num_cols:
                        row_data = {}
                        
                        for i, cell in enumerate(cells):
                            col_name = headers[i] if i < len(headers) else f"Column_{i+1}"
                            
                            # Check if this cell has links
                            try:
                                links = cell.find_elements(By.TAG_NAME, "a")
                                if links:
                                    # Extract both text and URLs
                                    link_texts = []
                                    link_urls = []
                                    for link in links:
                                        link_text = link.text.strip()
                                        link_url = link.get_attribute('href')
                                        if link_text and link_url:
                                            link_texts.append(link_text)
                                            link_urls.append(link_url)
                                    
                                    # Store text in main column
                                    row_data[col_name] = cell.text.strip()
                                    
                                    # Store URLs in separate column
                                    if link_urls:
                                        row_data[f"{col_name}_URL"] = " | ".join(link_urls)
                                else:
                                    row_data[col_name] = cell.text.strip()
                            except:
                                row_data[col_name] = cell.text.strip()
                        
                        section_data.append(row_data)
            
            if section_data:
                df = pd.DataFrame(section_data)
                sheet_name = f"{section.replace('.', '_')}"[:31]
                sheets_data[sheet_name] = df
                print(f"       ✓ Extracted {len(df)} records")
        
        except Exception as e:
            print(f"       ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    driver.quit()
    
    # Save to Excel with multiple sheets
    output_file = "AP5.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Add data sheets
        for sheet_name, df in sheets_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✓ Added sheet: {sheet_name}")
        
        # Add notes sheet if we have notes
        if notes_data:
            notes_df = pd.DataFrame([
                {'Section': section, 'Notes': notes}
                for section, notes in notes_data.items()
            ])
            notes_df.to_excel(writer, sheet_name="Notes", index=False)
            print(f"  ✓ Added sheet: Notes")
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETED!")
    print(f"{'='*60}")
    print(f"Excel file created: {output_file}")
    print(f"Total data sheets: {len(sheets_data)}")
    if notes_data:
        print(f"Notes captured for {len(notes_data)} sections")

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