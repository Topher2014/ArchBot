import os
import requests
from bs4 import BeautifulSoup
import json
import time
import random
from urllib.parse import urljoin, unquote
import re

# Create directory to store scraped data
DATA_DIR = "arch_wiki_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Create a log file
LOG_FILE = os.path.join(DATA_DIR, "scrape_log.txt")

# Headers to make our scraper look like a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}

def log_message(message):
    """Write message to log file and print to console"""
    print(message)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{message}\n")

def get_all_wiki_pages():
    """Get a list of all pages on the Arch Wiki"""
    log_message("Getting list of all Arch Wiki pages...")
    
    # Start with the "All pages" special page
    all_pages = []
    next_page_url = "https://wiki.archlinux.org/title/Special:AllPages"
    
    while next_page_url:
        log_message(f"Fetching page list from: {next_page_url}")
        response = requests.get(next_page_url, headers=headers)
        
        if response.status_code != 200:
            log_message(f"Error fetching page list: {response.status_code}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract page links from the current page
        content = soup.find('div', {'class': 'mw-allpages-body'})
        if content:
            for link in content.find_all('a'):
                page_url = urljoin("https://wiki.archlinux.org", link.get('href'))
                # Skip special pages and other namespaces
                if '/title/' in page_url and not any(x in page_url for x in ['Special:', 'Talk:', 'User:', 'File:']):
                    all_pages.append(page_url)
        
        # Find the "next page" link
        next_page = None
        nav_links = soup.find('div', {'class': 'mw-allpages-nav'})
        if nav_links:
            next_links = [link for link in nav_links.find_all('a') if 'Next page' in link.text]
            if next_links:
                next_page = urljoin("https://wiki.archlinux.org", next_links[0].get('href'))
        
        next_page_url = next_page
        
        # Be respectful to the server
        time.sleep(1)
    
    log_message(f"Found {len(all_pages)} wiki pages")
    
    # Save the page list for future reference
    with open(os.path.join(DATA_DIR, "all_pages.json"), 'w', encoding='utf-8') as f:
        json.dump(all_pages, f, indent=2)
    
    return all_pages

def load_existing_page_list():
    """Load the page list if it exists"""
    page_list_file = os.path.join(DATA_DIR, "all_pages.json")
    if os.path.exists(page_list_file):
        with open(page_list_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def clean_wiki_text(text):
    """Clean up wiki text by removing edit links, etc."""
    # Remove [edit] links
    text = re.sub(r'\[edit\]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_content_from_wiki_page(soup, url):
    """Extract structured content from a wiki page"""
    page_title = unquote(url.split('/title/')[-1].replace('_', ' '))
    
    # Get the main content
    content_div = soup.find('div', {'id': 'mw-content-text'})
    if not content_div:
        return None
    
    # Extract sections
    sections = []
    current_section = {"title": "Introduction", "level": 1, "content": ""}
    
    # Find all headings and content elements
    main_divs = content_div.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'ul', 'ol', 'div', 'table'], recursive=True)
    
    for element in main_divs:
        # Check if it's a heading
        if element.name in ['h1', 'h2', 'h3', 'h4']:
            # Save the previous section if it has content
            if current_section["content"].strip():
                sections.append(current_section.copy())
            
            # Get heading level
            level = int(element.name[1])
            
            # Get section title
            section_title = clean_wiki_text(element.get_text())
            
            # Create a new section
            current_section = {
                "title": section_title,
                "level": level,
                "content": ""
            }
        else:
            # Process content based on element type
            if element.name == 'p':
                text = clean_wiki_text(element.get_text())
                if text:  # Only add non-empty paragraphs
                    current_section["content"] += text + "\n\n"
            elif element.name == 'pre':
                # Code blocks
                code = element.get_text().strip()
                if code:
                    current_section["content"] += f"```\n{code}\n```\n\n"
            elif element.name in ['ul', 'ol']:
                # Lists
                for li in element.find_all('li', recursive=False):
                    prefix = "- " if element.name == 'ul' else "1. "
                    li_text = clean_wiki_text(li.get_text())
                    current_section["content"] += f"{prefix}{li_text}\n"
                current_section["content"] += "\n"
            elif element.name == 'div' and element.get('class') and any(cls in element.get('class') for cls in ['archwiki-template-box', 'archwiki-template-message']):
                # Handle info/note/warning boxes
                box_title = element.find('b')
                box_text = ""
                if box_title:
                    box_text = f"**{box_title.get_text()}** "
                box_content = element.find('p')
                if box_content:
                    box_text += clean_wiki_text(box_content.get_text())
                if box_text.strip():
                    current_section["content"] += f"Note: {box_text}\n\n"
            elif element.name == 'table':
                # Simple table handling
                table_content = "Table content:\n"
                for row in element.find_all('tr'):
                    cells = row.find_all(['th', 'td'])
                    if cells:
                        row_content = " | ".join([clean_wiki_text(cell.get_text()) for cell in cells])
                        table_content += f"{row_content}\n"
                if len(table_content.strip().split('\n')) > 1:  # Only add non-empty tables
                    current_section["content"] += table_content + "\n"
    
    # Add the last section
    if current_section["content"].strip():
        sections.append(current_section)
    
    # Create the page data structure
    page_data = {
        "title": page_title,
        "url": url,
        "sections": sections
    }
    
    return page_data

def scrape_wiki_page(url):
    """Scrape an individual Arch Wiki page"""
    try:
        page_title = unquote(url.split('/title/')[-1].replace('_', ' '))
        safe_title = page_title.replace('/', '_').replace('\\', '_').replace(':', '_')
        output_file = os.path.join(DATA_DIR, f"{safe_title}.json")
        
        # Skip if already scraped
        if os.path.exists(output_file):
            return True
        
        log_message(f"Scraping: {page_title}")
        
        # Get the page content
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            log_message(f"Error fetching {url}: {response.status_code}")
            return False
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract the content
        page_data = extract_content_from_wiki_page(soup, url)
        if not page_data:
            log_message(f"Could not extract content from {url}")
            return False
        
        # Save the data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(page_data, f, indent=2, ensure_ascii=False)
        
        return True
    
    except Exception as e:
        log_message(f"Error scraping {url}: {str(e)}")
        return False

def process_all_pages(page_list, delay_min=1, delay_max=3):
    """Process all wiki pages with rate limiting"""
    total_pages = len(page_list)
    success_count = 0
    error_count = 0
    skip_count = 0
    
    log_message(f"Starting scraping of {total_pages} pages...")
    
    for i, url in enumerate(page_list):
        page_title = unquote(url.split('/title/')[-1].replace('_', ' '))
        safe_title = page_title.replace('/', '_').replace('\\', '_').replace(':', '_')
        output_file = os.path.join(DATA_DIR, f"{safe_title}.json")
        
        # Skip if already scraped
        if os.path.exists(output_file):
            skip_count += 1
            if i % 10 == 0:
                log_message(f"Progress: {i+1}/{total_pages} - Skipped: {skip_count}, Success: {success_count}, Error: {error_count}")
            continue
        
        # Scrape the page
        success = scrape_wiki_page(url)
        if success:
            success_count += 1
        else:
            error_count += 1
        
        # Log progress
        if i % 10 == 0 or i == total_pages - 1:
            log_message(f"Progress: {i+1}/{total_pages} - Skipped: {skip_count}, Success: {success_count}, Error: {error_count}")
        
        # Random delay to be nice to their servers
        delay = random.uniform(delay_min, delay_max)
        time.sleep(delay)
    
    log_message(f"Scraping complete! Total pages: {total_pages}, Skipped: {skip_count}, Success: {success_count}, Error: {error_count}")

def main():
    """Main function to control the scraping process"""
    # Initialize log
    log_message(f"Starting Arch Wiki scraper at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Try to load existing page list
    page_list = load_existing_page_list()
    
    # If no existing list, fetch all pages
    if not page_list:
        page_list = get_all_wiki_pages()
    
    # Process all pages
    process_all_pages(page_list)
    
    log_message(f"Scraping finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()             
