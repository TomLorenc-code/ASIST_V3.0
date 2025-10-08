from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import time
import os

print("="*60)
print("Appendix 4 - Generic Codes Extractor")
print("="*60)

driver_path = os.path.join(os.getcwd(), 'msedgedriver.exe')
edge_options = Options()
edge_options.add_argument('--start-maximized')

driver = webdriver.Edge(service=Service(driver_path), options=edge_options)

try:
    url = "https://samm.dsca.mil/appendix/appendix-4"
    print(f"\n[1/3] Opening main page...")
    driver.get(url)
    time.sleep(5)
    
    print("[2/3] Finding all Budget Activity Codes...")
    tables = driver.find_elements(By.TAG_NAME, "table")
    
    activity_links = []
    
    for table in tables:
        rows = table.find_elements(By.TAG_NAME, "tr")
        header_cells = rows[0].find_elements(By.TAG_NAME, "th") if rows else []
        
        if any("Budget Activity Code" in cell.text for cell in header_cells):
            for row in rows[1:]:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 2:
                    code = cells[0].text.strip()
                    try:
                        link_elem = cells[1].find_element(By.TAG_NAME, "a")
                        activity = link_elem.text.strip()
                        link_url = link_elem.get_attribute('href')
                        if code and link_url:
                            activity_links.append({'code': code, 'activity': activity, 'url': link_url})
                    except:
                        continue
    
    print(f"     ✓ Found {len(activity_links)} activities")
    print(f"\n[3/3] Extracting details...")
    
    sheets_data = {}
    
    for idx, activity_info in enumerate(activity_links, start=1):
        code = activity_info['code']
        activity = activity_info['activity']
        link_url = activity_info['url']
        
        print(f"\n  [{idx}/{len(activity_links)}] {code} - {activity}")
        
        try:
            driver.get(link_url)
            time.sleep(3)
            
            detail_tables = driver.find_elements(By.TAG_NAME, "table")
            activity_data = []
            
            if detail_tables:
                table = detail_tables[0]
                rows = table.find_elements(By.TAG_NAME, "tr")
                
                # Track hierarchy
                level1_code = ""
                level1_desc = ""
                level2_code = ""
                level2_desc = ""
                
                for row in rows[1:]:  # Skip header
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    # Get all text content, replacing nbsp with empty string
                    cell_texts = []
                    for cell in cells:
                        text = cell.text.strip()
                        if text == '\xa0' or not text:  # nbsp or empty
                            cell_texts.append("")
                        else:
                            cell_texts.append(text)
                    
                    # Count non-empty cells
                    non_empty = [t for t in cell_texts if t]
                    
                    if len(non_empty) == 1:
                        # Single value
                        if non_empty[0].isalpha() and len(non_empty[0]) == 1:
                            level1_code = non_empty[0]
                            level2_code = ""
                            level2_desc = ""
                        elif non_empty[0].isdigit():
                            level2_code = non_empty[0]
                            level2_desc = ""
                    
                    elif len(non_empty) == 2:
                        # Code and description
                        if non_empty[0].isdigit():
                            level2_code = non_empty[0]
                            level2_desc = non_empty[1]
                        elif len(non_empty[0]) <= 2:
                            level1_code = non_empty[0]
                            level1_desc = non_empty[1]
                            level2_code = ""
                            level2_desc = ""
                    
                    elif len(non_empty) >= 3:
                        # Detail row - extract up to 6 columns
                        detail_code = non_empty[0] if len(non_empty) > 0 else ""
                        detail_desc = non_empty[1] if len(non_empty) > 1 else ""
                        fsc = non_empty[2] if len(non_empty) > 2 else ""
                        major_item = non_empty[3] if len(non_empty) > 3 else ""
                        dollar_line = non_empty[4] if len(non_empty) > 4 else ""
                        remarks = non_empty[5] if len(non_empty) > 5 else ""
                        
                        activity_data.append({
                            'Main_Category_Code': level1_code,
                            'Main_Category': level1_desc,
                            'Subcategory_Code': level2_code,
                            'Subcategory': level2_desc,
                            'Item_Code': detail_code,
                            'Item_Description': detail_desc,
                            'FSC': fsc,
                            'Major_Item': major_item,
                            'Dollar_Line': dollar_line,
                            'Remarks': remarks
                        })
                
                if activity_data:
                    df = pd.DataFrame(activity_data)
                    sheet_name = f"{code}_{activity}"[:31]
                    sheets_data[sheet_name] = df
                    print(f"       ✓ {len(df)} records")
            
        except Exception as e:
            print(f"       ✗ Error: {e}")
            continue
    
    driver.quit()
    
    output_file = "AP4.2V2.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in sheets_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETED!")
    print(f"{'='*60}")
    print(f"Excel file created: {output_file}")
    print(f"Total sheets: {len(sheets_data)}")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    try:
        driver.quit()
    except:
        pass