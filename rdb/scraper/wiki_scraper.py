"""
Arch Wiki scraper for RDB.
"""

import os
import requests
from bs4 import BeautifulSoup
import json
import time
import random
from urllib.parse import urljoin, unquote
from pathlib import Path
from typing import List, Optional
import logging

from ..config.settings import Config
from ..utils.logging import get_logger
from .content_parser import ContentParser


class WikiScraper:
    """Scraper for Arch Wiki documentation."""

    def __init__(self, config: Config):
       """Initialize wiki scraper with configuration."""
       self.config = config
       self.logger = get_logger(__name__)
       self.parser = ContentParser()
       
       # Headers to make scraper look like a browser
       self.headers = {
            'User-Agent': 'git/2.40.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       }
       
       # Base URL for Arch Wiki
       self.base_url = "https://wiki.archlinux.org"
       
    def get_all_pages(self) -> List[str]:
       """Get list of all pages on Arch Wiki."""
       self.logger.info("Getting list of all Arch Wiki pages...")
       
       all_pages = []
       next_page_url = f"{self.base_url}/title/Special:AllPages?hideredirects=1"
       
       while next_page_url:
           self.logger.debug(f"Fetching page list from: {next_page_url}")
           
           try:
               response = requests.get(next_page_url, headers=self.headers, timeout=30)
               response.raise_for_status()
           except requests.RequestException as e:
               self.logger.error(f"Error fetching page list: {e}")
               break
               
           soup = BeautifulSoup(response.text, 'html.parser')
           
           # Extract page links
           content = soup.find('div', {'class': 'mw-allpages-body'})
           if content:
               for link in content.find_all('a'):
                   href = link.get('href')
                   if href and '/title/' in href:
                       href = href.replace('/title//', '/title/')
                       page_url = urljoin(self.base_url, href)
                       # Skip special pages and other namespaces
                       if not any(x in page_url for x in ['Special:', 'Talk:', 'User:', 'File:', 'International_communities']):
                           all_pages.append(page_url)
           
           # Find next page link
           next_page_url = None
           nav_links = soup.find('div', {'class': 'mw-allpages-nav'})
           if nav_links:
               next_links = [link for link in nav_links.find_all('a') if 'Next page' in link.text]
               if next_links:
                   next_page_url = urljoin(self.base_url, next_links[0].get('href'))
           
           # Be respectful to the server
           time.sleep(1)
       
       self.logger.info(f"Found {len(all_pages)} wiki pages")
       return all_pages

    def scrape_page(self, url: str) -> Optional[dict]:
        """Scrape individual wiki page, handling redirects."""
        try:
            page_title = unquote(url.split('/title/')[-1].replace('_', ' '))
            self.logger.debug(f"Scraping: {page_title}")
            
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Check if this was a redirect
            canonical_url = response.url
            is_redirect = canonical_url != url
            
            if is_redirect:
                canonical_title = unquote(canonical_url.split('/title/')[-1].replace('_', ' '))
                self.logger.debug(f"Redirect detected: {page_title} -> {canonical_title}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            page_data = self.parser.extract_content(soup, canonical_url)
            
            if page_data:
                # Add redirect information
                page_data['canonical_url'] = canonical_url
                page_data['original_url'] = url
                page_data['is_redirect'] = is_redirect
                if is_redirect:
                    page_data['redirect_source'] = page_title
                return page_data
            else:
               error_msg = f"No content extracted from: {page_title}"
               self.logger.warning(error_msg)
               return None
               
        except requests.RequestException as e:
           error_msg = f"Network error scraping {url}: {e}"
           self.logger.error(error_msg)
           return None
        except Exception as e:
           error_msg = f"Error scraping {url}: {e}"
           self.logger.error(f"Error scraping {url}: {e}")
           return None

    def save_page(self, page_data: dict, output_dir: Path) -> bool:
       """Save page data to JSON file."""
       try:
           page_title = page_data.get('title', 'Unknown')
           safe_title = page_title.replace('/', '_').replace('\\', '_').replace(':', '_')
           output_file = output_dir / f"{safe_title}.json"
           
           with open(output_file, 'w', encoding='utf-8') as f:
               json.dump(page_data, f, indent=2, ensure_ascii=False)
           
           return True
           
       except Exception as e:
           self.logger.error(f"Error saving page data: {e}")
           return False

    def scrape_all(self, output_dir: Optional[str] = None) -> int:
        """Scrape all wiki pages."""
        if output_dir is None:
            output_dir = self.config.raw_data_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track canonical URLs to avoid scraping duplicates
        canonical_urls_seen = set()
        redirect_mappings = {}  # original_url -> canonical_url
        
        # Get or load page list
        page_list_file = output_dir / "page_list.json"
        
        if page_list_file.exists():
            self.logger.info("Loading existing page list...")
            with open(page_list_file, 'r', encoding='utf-8') as f:
                page_list = json.load(f)
        else:
            page_list = self.get_all_pages()
            # Save page list for future reference
            with open(page_list_file, 'w', encoding='utf-8') as f:
                json.dump(page_list, f, indent=2)
        
        # Process pages
        total_pages = len(page_list)
        success_count = 0
        error_count = 0
        skip_count = 0
        
        self.logger.info(f"Starting scraping of {total_pages} pages...")
        
        for i, url in enumerate(page_list):
            page_title = unquote(url.split('/title/')[-1].replace('_', ' '))
            safe_title = page_title.replace('/', '_').replace('\\', '_').replace(':', '_')
            output_file = output_dir / f"{safe_title}.json"
            
            # Skip if already exists
            if output_file.exists():
                skip_count += 1
                if i % 50 == 0:
                    self.logger.info(f"Progress: {i+1}/{total_pages} - "
                                   f"Skipped: {skip_count}, Success: {success_count}, Error: {error_count}")
                continue
            
            # Scrape page
            page_data = self.scrape_page(url)
            if page_data:
                canonical_url = page_data.get('canonical_url', url)
                
                # Check if we've already scraped this canonical URL
                if canonical_url in canonical_urls_seen:
                    # Store redirect mapping but skip saving
                    redirect_mappings[url] = canonical_url
                    skip_count += 1
                    self.logger.debug(f"Skipping duplicate canonical URL: {canonical_url}")
                else:
                    # Mark canonical URL as seen
                    canonical_urls_seen.add(canonical_url)
                    
                    # Save the page
                    if self.save_page(page_data, output_dir):
                        success_count += 1
                    else:
                        error_count += 1
            else:
                error_count += 1
            
            # Progress logging
            if i % 50 == 0 or i == total_pages - 1:
                self.logger.info(f"Progress: {i+1}/{total_pages} - "
                               f"Skipped: {skip_count}, Success: {success_count}, Error: {error_count}")
            
            # Rate limiting
            delay = random.uniform(self.config.scrape_delay_min, self.config.scrape_delay_max)
            time.sleep(delay)
        
        # Save redirect mappings for reference
        if redirect_mappings:
            redirect_file = output_dir / "redirects.json"
            with open(redirect_file, 'w', encoding='utf-8') as f:
                json.dump(redirect_mappings, f, indent=2)
            self.logger.info(f"Saved {len(redirect_mappings)} redirect mappings to {redirect_file}")
        
        self.logger.info(f"Scraping complete! Total: {total_pages}, "
                        f"Skipped: {skip_count}, Success: {success_count}, Error: {error_count}")
        
        return success_count
