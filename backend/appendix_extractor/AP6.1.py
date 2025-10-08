from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import time
import os
import re

print("="*60)
print("Appendix 6 - LOA Notes DETAILED Extractor (FINAL)")
print("="*60)

driver_path = os.path.join(os.getcwd(), 'msedgedriver.exe')
edge_options = Options()
edge_options.add_argument('--start-maximized')

driver = webdriver.Edge(service=Service(driver_path), options=edge_options)

def extract_field_by_label(driver, label_text):
    """
    Extract field value by finding the label and getting its sibling field__item
    """
    try:
        # Find the div containing the label
        label_elem = driver.find_element(By.XPATH, 
            f"//div[@class='field__label' and contains(text(), '{label_text}')]")
        
        # Get the parent field container
        parent = label_elem.find_element(By.XPATH, "./..")
        
        # Find the field__item within the same parent
        value_elem = parent.find_element(By.CLASS_NAME, "field__item")
        
        return value_elem.text.strip()
    except Exception as e:
        return None

def extract_note_usage_instructions(driver):
    """
    Extract complete Note Usage Instructions with all paragraphs
    """
    try:
        # Find the LOA_Note_Usage div
        usage_div = driver.find_element(By.XPATH, 
            "//div[contains(@class, 'LOA_Note_Usage')]")
        
        # Get the field__item within it
        field_item = usage_div.find_element(By.CLASS_NAME, "field__item")
        
        # Get all paragraphs
        paragraphs = field_item.find_elements(By.TAG_NAME, "p")
        
        # Combine all paragraph text
        full_text = "\n\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
        
        return full_text
    except Exception as e:
        print(f"       Warning: Could not extract Usage Instructions: {e}")
        return None

def extract_note_text(driver):
    """
    Extract complete Note Text with all content
    """
    try:
        # Find the LOA_Note_Text div
        note_text_div = driver.find_element(By.XPATH, 
            "//div[contains(@class, 'LOA_Note_Text')]")
        
        # Get the field__item within it
        field_item = note_text_div.find_element(By.CLASS_NAME, "field__item")
        
        # Get all content - paragraphs, lists, etc.
        # Method 1: Try to get structured content
        try:
            content_parts = []
            
            # Get all direct children (p, ul, ol, div, etc.)
            children = field_item.find_elements(By.XPATH, "./*")
            
            for child in children:
                text = child.text.strip()
                if text:
                    content_parts.append(text)
            
            if content_parts:
                return "\n\n".join(content_parts)
        except:
            pass
        
        # Method 2: Fallback to getting all text
        return field_item.text.strip()
        
    except Exception as e:
        print(f"       Warning: Could not extract Note Text: {e}")
        return None

try:
    # Step 1: Get the main table with all note links
    url = "https://samm.dsca.mil/appendix/appendix-6"
    print(f"\n[1/3] Opening main page...")
    driver.get(url)
    time.sleep(5)
    
    print("[2/3] Extracting note links...")
    tables = driver.find_elements(By.TAG_NAME, "table")
    
    note_links = []
    
    for table in tables:
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        if len(rows) < 2:
            continue
        
        header_cells = rows[0].find_elements(By.TAG_NAME, "th")
        headers = [h.text.strip() for h in header_cells]
        
        if "NOTE TITLE" not in ' '.join(headers):
            continue
        
        print(f"     Found main table with {len(rows)-1} notes")
        
        # Extract note links
        for row in rows[1:]:
            cells = row.find_elements(By.TAG_NAME, "td")
            
            if len(cells) >= 1:
                try:
                    link = cells[0].find_element(By.TAG_NAME, "a")
                    title = link.text.strip()
                    note_url = link.get_attribute('href')
                    
                    if title and note_url:
                        note_links.append({
                            'title': title,
                            'url': note_url
                        })
                except:
                    continue
    
    print(f"     ✓ Found {len(note_links)} note links")
    
    # Step 2: Visit each note and extract detailed info
    print(f"\n[3/3] Extracting detailed information from each note...")
    
    all_sheets = {}
    index_data = []
    
    for idx, note_info in enumerate(note_links, start=1):
        title = note_info['title']
        note_url = note_info['url']
        
        print(f"\n  [{idx}/{len(note_links)}] {title}")
        
        try:
            driver.get(note_url)
            time.sleep(3)
            
            detail_data = []
            
            # Add title and URL
            detail_data.append({'Field': 'Title', 'Value': title})
            detail_data.append({'Field': 'URL', 'Value': note_url})
            
            # Extract all metadata fields using the new function
            print(f"       Extracting metadata...")
            
            # Foreign Military Sales
            fms = extract_field_by_label(driver, 'Foreign Military Sales')
            if fms:
                detail_data.append({'Field': 'Foreign Military Sales', 'Value': fms})
                print(f"       ✓ FMS: {fms}")
            
            # Building Partner Capacity
            bpc = extract_field_by_label(driver, 'Building Partner Capacity')
            if bpc:
                detail_data.append({'Field': 'Building Partner Capacity', 'Value': bpc})
                print(f"       ✓ BPC: {bpc}")
            
            # Note Input Responsibility
            resp = extract_field_by_label(driver, 'Note Input Responsibility')
            if resp:
                detail_data.append({'Field': 'Note Input Responsibility', 'Value': resp})
                print(f"       ✓ Responsibility: {resp}")
            
            # Date Range Of Use
            date_range = extract_field_by_label(driver, 'Date Range Of Use')
            if date_range:
                detail_data.append({'Field': 'Date Range Of Use', 'Value': date_range})
                print(f"       ✓ Date Range: {date_range}")
            
            # References
            refs = extract_field_by_label(driver, 'References')
            if refs:
                detail_data.append({'Field': 'References', 'Value': refs})
                print(f"       ✓ References: {refs[:50]}...")
            
            # Extract Note Usage Instructions
            print(f"       Extracting Usage Instructions...")
            usage = extract_note_usage_instructions(driver)
            if usage:
                detail_data.append({'Field': 'Usage Instructions', 'Value': usage})
                print(f"       ✓ Usage Instructions: {len(usage)} characters")
                # Show first line
                first_line = usage.split('\n')[0][:80]
                print(f"       Preview: {first_line}...")
            
            # Extract Note Text
            print(f"       Extracting Note Text...")
            note_text = extract_note_text(driver)
            if note_text:
                detail_data.append({'Field': 'Note Text', 'Value': note_text})
                print(f"       ✓ Note Text: {len(note_text)} characters")
                # Show first line
                first_line = note_text.split('\n')[0][:80]
                print(f"       Preview: {first_line}...")
            
            # Create DataFrame for this note
            df = pd.DataFrame(detail_data)
            
            # Create safe sheet name
            safe_title = re.sub(r'[^\w\s-]', '', title)[:31]
            safe_title = safe_title.replace(' ', '_')
            
            all_sheets[safe_title] = df
            
            # Add to index
            index_data.append({
                'Note_Title': title,
                'Sheet_Name': safe_title,
                'URL': note_url
            })
            
            print(f"       ✓✓ TOTAL: {len(detail_data)} fields extracted")
            
        except Exception as e:
            print(f"       ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    driver.quit()
    
    # Save everything to Excel
    output_file = "AP6.1Finaltry.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # First sheet: Index
        index_df = pd.DataFrame(index_data)
        index_df.to_excel(writer, sheet_name="Index", index=False)
        print(f"\n  ✓ Added Index sheet")
        
        # Individual note sheets
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\n{'='*60}")
    print("✓✓✓ EXTRACTION COMPLETED SUCCESSFULLY! ✓✓✓")
    print(f"{'='*60}")
    print(f"Excel file created: {output_file}")
    print(f"Total notes extracted: {len(all_sheets)}")
    print(f"Sheets: 1 Index + {len(all_sheets)} detail sheets")

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