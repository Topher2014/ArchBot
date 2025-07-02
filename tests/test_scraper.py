"""
Tests for the scraper module.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from rdb.config.settings import Config
from rdb.scraper.wiki_scraper import WikiScraper
from rdb.scraper.content_parser import ContentParser


class TestContentParser:
   """Test cases for ContentParser."""
   
   def setup_method(self):
       """Setup test fixtures."""
       self.parser = ContentParser()
   
   def test_clean_text(self):
       """Test text cleaning functionality."""
       # Test normal text
       assert self.parser.clean_text("Hello world") == "Hello world"
       
       # Test text with edit links
       assert self.parser.clean_text("Hello [edit] world") == "Hello  world"
       
       # Test text with multiple spaces
       assert self.parser.clean_text("Hello    world") == "Hello world"
       
       # Test empty text
       assert self.parser.clean_text("") == ""
       assert self.parser.clean_text(None) == ""
   
   def test_extract_content_no_content_div(self):
       """Test extract_content with missing content div."""
       from bs4 import BeautifulSoup
       
       html = "<html><body><p>No content div</p></body></html>"
       soup = BeautifulSoup(html, 'html.parser')
       
       result = self.parser.extract_content(soup, "http://example.com/title/Test")
       assert result is None
   
   def test_extract_content_with_sections(self):
       """Test extract_content with proper sections."""
       from bs4 import BeautifulSoup
       
       html = """
       <html>
       <body>
       <div id="mw-content-text">
           <p>Introduction paragraph</p>
           <h2>Section 1</h2>
           <p>Section 1 content</p>
           <h3>Subsection</h3>
           <p>Subsection content</p>
       </div>
       </body>
       </html>
       """
       
       soup = BeautifulSoup(html, 'html.parser')
       result = self.parser.extract_content(soup, "http://example.com/title/Test_Page")
       
       assert result is not None
       assert result['title'] == "Test Page"
       assert result['url'] == "http://example.com/title/Test_Page"
       assert len(result['sections']) >= 2
       
       # Check introduction section
       intro_section = result['sections'][0]
       assert intro_section['title'] == "Introduction"
       assert "Introduction paragraph" in intro_section['content']
       
       # Check first section
       section1 = result['sections'][1]
       assert section1['title'] == "Section 1"
       assert section1['level'] == 2


class TestWikiScraper:
   """Test cases for WikiScraper."""
   
   def setup_method(self):
       """Setup test fixtures."""
       self.config = Config()
       self.scraper = WikiScraper(self.config)
   
   @patch('requests.get')
   def test_scrape_page_success(self, mock_get):
       """Test successful page scraping."""
       # Mock response
       mock_response = Mock()
       mock_response.status_code = 200
       mock_response.text = """
       <html>
       <body>
       <div id="mw-content-text">
           <p>Test content</p>
       </div>
       </body>
       </html>
       """
       mock_get.return_value = mock_response
       
       result = self.scraper.scrape_page("http://example.com/title/Test")
       
       assert result is not None
       assert result['title'] == "Test"
       assert result['url'] == "http://example.com/title/Test"
   
   @patch('requests.get')
   def test_scrape_page_network_error(self, mock_get):
       """Test page scraping with network error."""
       import requests
       
       mock_get.side_effect = requests.RequestException("Network error")
       
       result = self.scraper.scrape_page("http://example.com/title/Test")
       assert result is None
   
   def test_save_page(self, tmp_path):
       """Test saving page data to file."""
       page_data = {
           'title': 'Test Page',
           'url': 'http://example.com/title/Test_Page',
           'sections': []
       }
       
       success = self.scraper.save_page(page_data, tmp_path)
       
       assert success is True
       
       # Check file was created
       expected_file = tmp_path / "Test_Page.json"
       assert expected_file.exists()
       
       # Check file contents
       with open(expected_file, 'r', encoding='utf-8') as f:
           loaded_data = json.load(f)
       
       assert loaded_data == page_data
   
   @patch('rdb.scraper.wiki_scraper.WikiScraper.get_all_pages')
   @patch('rdb.scraper.wiki_scraper.WikiScraper.scrape_page')
   @patch('rdb.scraper.wiki_scraper.WikiScraper.save_page')
   def test_scrape_all(self, mock_save, mock_scrape, mock_get_pages, tmp_path):
       """Test scraping all pages."""
       # Mock page list
       mock_get_pages.return_value = [
           "http://example.com/title/Page1",
           "http://example.com/title/Page2"
       ]
       
       # Mock scraping results
       mock_scrape.side_effect = [
           {'title': 'Page1', 'url': 'http://example.com/title/Page1', 'sections': []},
           {'title': 'Page2', 'url': 'http://example.com/title/Page2', 'sections': []}
       ]
       
       # Mock save success
       mock_save.return_value = True
       
       # Override config for testing
       self.scraper.config.scrape_delay_min = 0
       self.scraper.config.scrape_delay_max = 0
       
       result = self.scraper.scrape_all(str(tmp_path))
       
       assert result == 2
       assert mock_scrape.call_count == 2
       assert mock_save.call_count == 2


class TestScraperIntegration:
   """Integration tests for scraper components."""
   
   def test_full_scraping_workflow(self, tmp_path):
       """Test complete scraping workflow with real HTML parsing."""
       config = Config(data_dir=str(tmp_path))
       scraper = WikiScraper(config)
       
       # Create test HTML content
       test_html = """
       <html>
       <body>
       <div id="mw-content-text">
           <p>This is the introduction.</p>
           <h2>Installation</h2>
           <p>Install the package using pacman:</p>
           <pre>$ sudo pacman -S package-name</pre>
           <h3>Configuration</h3>
           <p>Edit the configuration file:</p>
           <ul>
               <li>Option 1: Enable feature A</li>
               <li>Option 2: Disable feature B</li>
           </ul>
           <h2>Troubleshooting</h2>
           <p>Common issues and solutions.</p>
       </div>
       </body>
       </html>
       """
       
       # Mock the HTTP request
       with patch('requests.get') as mock_get:
           mock_response = Mock()
           mock_response.status_code = 200
           mock_response.text = test_html
           mock_get.return_value = mock_response
           
           # Scrape the page
           result = scraper.scrape_page("http://example.com/title/Test_Package")
       
       # Verify the result
       assert result is not None
       assert result['title'] == "Test Package"
       assert len(result['sections']) >= 3
       
       # Check sections
       section_titles = [s['title'] for s in result['sections']]
       assert "Introduction" in section_titles
       assert "Installation" in section_titles
       assert "Configuration" in section_titles
       assert "Troubleshooting" in section_titles
       
       # Check content includes code blocks and lists
       installation_section = next(s for s in result['sections'] if s['title'] == "Installation")
       assert "pacman -S" in installation_section['content']
       assert "```" in installation_section['content']  # Code block formatting
       
       configuration_section = next(s for s in result['sections'] if s['title'] == "Configuration")
       assert "Option 1" in configuration_section['content']
       assert "Option 2" in configuration_section['content']
