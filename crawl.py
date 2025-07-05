import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning # Import XMLParsedAsHTMLWarning
from urllib.parse import urljoin, urlparse
import time
import re
import collections
import warnings # Import warnings module

# Filter out the XMLParsedAsHTMLWarning globally
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --- Configuration ---
base_url = "https://ubestream.com/"
sitemap_url = "https://ubestream.com/sitemap.xml"
CRAWL_DELAY = 3 # seconds, as per robots.txt

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br', # Let requests handle decompression
    'Connection': 'keep-alive'
}

# --- Global Data Structures ---
visited_urls = set() # Stores URLs that have already been crawled to avoid re-processing
all_extracted_text = [] # Stores all extracted text snippets
urls_to_visit = collections.deque() # Queue for URLs to be crawled (FIFO for BFS)

# --- Helper Functions ---

def get_urls_from_sitemap(sitemap_url):
    """
    Fetches a list of URLs from the sitemap.xml.
    This function specifically targets the initial sitemap.xml.
    Further sitemap indexes will be handled in deep_crawl.
    """
    print(f"Fetching URLs from sitemap: {sitemap_url}")
    urls = []
    try:
        response = requests.get(sitemap_url, headers=HEADERS, timeout=10)
        response.raise_for_status()

        # Explicitly use 'lxml-xml' for XML parsing for robustness
        soup = BeautifulSoup(response.content, 'lxml-xml') 
        
        # Find all <loc> tags which contain the URLs
        # This will get URLs from both <url> tags (standard sitemap)
        # and <sitemap> tags (sitemap index) for initial discovery.
        for loc_tag in soup.find_all('loc'):
            url = loc_tag.get_text(strip=True)
            if url.startswith(base_url): # Ensure the URL belongs to the target domain
                urls.append(url)
        
        print(f"Found {len(urls)} URLs in sitemap.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
    except Exception as e:
        print(f"Error parsing sitemap {sitemap_url}: {e}")
    return urls

def extract_all_page_content(soup_obj, current_url):
    """
    Extracts all visible text and content from specific attributes (alt, title, value)
    from a BeautifulSoup object.
    This function is designed to handle deeply nested text automatically via soup.get_text().
    """
    page_text_fragments = []

    # Remove script, style, header, footer, nav, and noscript tags to clean up main text
    for tag_name in ['script', 'style', 'header', 'footer', 'nav', 'noscript']:
        for element in soup_obj.find_all(tag_name):
            element.extract() # Remove the tag and its contents
    
    # Get text from the body or the entire document after cleaning
    # get_text() automatically handles deeply nested text.
    main_text = soup_obj.get_text(separator='\n', strip=True)
    if main_text:
        page_text_fragments.append(main_text)

    # Extract text from 'alt' attribute of images
    for img in soup_obj.find_all('img', alt=True):
        if img['alt'].strip():
            page_text_fragments.append(f"[ALT: {img['alt'].strip()}]")

    # Extract text from 'title' attribute of various elements
    for element_with_title in soup_obj.find_all(attrs={'title': True}):
        if element_with_title['title'].strip():
            page_text_fragments.append(f"[TITLE: {element_with_title['title'].strip()}]")

    # Extract text from 'value' attribute of input fields
    for input_tag in soup_obj.find_all('input', value=True):
        if input_tag['value'].strip():
            page_text_fragments.append(f"[INPUT_VALUE: {input_tag['value'].strip()}]")

    # Extract text content from <textarea> elements
    for textarea_tag in soup_obj.find_all('textarea'):
        if textarea_tag.string and textarea_tag.string.strip():
            page_text_fragments.append(f"[TEXTAREA: {textarea_tag.string.strip()}]")

    # Extract text content from <option> elements within <select> dropdowns
    for option_tag in soup_obj.find_all('option'):
        if option_tag.string and option_tag.string.strip():
            page_text_fragments.append(f"[OPTION: {option_tag.string.strip()}]")

    # Combine all text fragments and clean up redundant blank lines
    combined_text = "\n".join(filter(None, page_text_fragments)) # Filter out any empty fragments
    cleaned_text = re.sub(r'\n\s*\n', '\n', combined_text).strip()
    
    return cleaned_text

# --- Main Crawling Function ---

def deep_crawl():
    """
    Performs a deep crawl by discovering new links and visiting them in order (BFS).
    It starts from the sitemap and also discovers links found within crawled pages.
    """
    initial_sitemap_urls = get_urls_from_sitemap(sitemap_url)
    
    # Use a set to store all unique URLs discovered from sitemaps (including sub-sitemaps)
    all_sitemap_urls_discovered = set()
    sitemap_index_urls_to_check = collections.deque([sitemap_url]) # Start with the main sitemap

    # Loop to process sitemap index files and collect all URLs
    while sitemap_index_urls_to_check:
        current_sitemap_url_to_process = sitemap_index_urls_to_check.popleft()
        # Avoid re-processing a sitemap URL if it was already the main sitemap and handled
        if current_sitemap_url_to_process in all_sitemap_urls_discovered:
            continue

        try:
            response = requests.get(current_sitemap_url_to_process, headers=HEADERS, timeout=10)
            response.raise_for_status()
            
            # Parse as XML using lxml for all sitemap-related content
            sitemap_soup = BeautifulSoup(response.content, 'lxml-xml')

            # Check if it's a sitemap index (contains <sitemap> tags)
            if sitemap_soup.find('sitemap'):
                for sitemap_tag in sitemap_soup.find_all('sitemap'):
                    loc_tag = sitemap_tag.find('loc')
                    if loc_tag and loc_tag.get_text(strip=True).startswith(base_url):
                        sub_sitemap_url = loc_tag.get_text(strip=True)
                        if sub_sitemap_url not in all_sitemap_urls_discovered and sub_sitemap_url not in sitemap_index_urls_to_check:
                            sitemap_index_urls_to_check.append(sub_sitemap_url)
                            print(f"Discovered sub-sitemap: {sub_sitemap_url}")
            # If it's a regular sitemap (contains <url> tags)
            elif sitemap_soup.find('url'):
                for url_tag in sitemap_soup.find_all('url'):
                    loc_tag = url_tag.find('loc')
                    if loc_tag and loc_tag.get_text(strip=True).startswith(base_url):
                        all_sitemap_urls_discovered.add(loc_tag.get_text(strip=True))

            all_sitemap_urls_discovered.add(current_sitemap_url_to_process) # Mark this sitemap URL itself as processed
            time.sleep(CRAWL_DELAY) # Respect delay for sitemap fetching too

        except requests.exceptions.RequestException as e:
            print(f"Error fetching sitemap index/sub-sitemap {current_sitemap_url_to_process}: {e}")
        except Exception as e:
            print(f"Error parsing sitemap index/sub-sitemap {current_sitemap_url_to_process}: {e}")

    # Add all collected sitemap URLs to the main crawling queue
    for url in all_sitemap_urls_discovered:
        if url not in visited_urls and url not in urls_to_visit:
            urls_to_visit.append(url)
            
    # Ensure the base URL is always added if not already there, and prioritize it
    if base_url not in visited_urls and base_url not in urls_to_visit:
        urls_to_visit.appendleft(base_url) # Add to front to ensure it's crawled early
        print(f"Added {base_url} to the queue as a starting point (or prioritized).")

    # --- Main Crawling Loop ---
    while urls_to_visit:
        current_url = urls_to_visit.popleft() # Get the next URL from the front of the queue (FIFO - BFS)

        if current_url in visited_urls:
            continue # Skip if already visited
        
        # Ensure the URL is within the same domain to avoid crawling external sites
        if urlparse(current_url).netloc != urlparse(base_url).netloc:
            print(f"Skipping external URL: {current_url}")
            continue

        print(f"Crawling: {current_url}")
        visited_urls.add(current_url) # Mark as visited

        try:
            response = requests.get(current_url, headers=HEADERS, timeout=15) # Increased timeout
            response.raise_for_status() # Check for HTTP errors

            # Rely on requests.text for automatic decompression and encoding detection.
            # This is generally the most robust approach.
            html_content = response.text 
            
            # For HTML content, typically use 'html.parser' or 'lxml' (for HTML)
            soup = BeautifulSoup(html_content, 'html.parser') 

            # Extract all relevant text content from the current page
            page_content = extract_all_page_content(soup, current_url)
            all_extracted_text.append(f"--- Content from: {current_url} ---\n{page_content}\n")

            # Find all new internal links on this page and add them to the queue
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href) # Convert relative URLs to absolute

                # Normalize URL: remove fragments (#section) as they don't represent unique pages
                parsed_absolute_url = urlparse(absolute_url)
                clean_url = parsed_absolute_url._replace(fragment="").geturl()

                # Add to queue only if it's internal, not visited, and not already in the queue
                if clean_url.startswith(base_url) and clean_url not in visited_urls and clean_url not in urls_to_visit:
                    urls_to_visit.append(clean_url) # Add to the end of the queue

            # Pause according to the Crawl-delay to be respectful to the server
            time.sleep(CRAWL_DELAY) 

        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error crawling {current_url}: {e.response.status_code} - {e.response.reason}")
        except requests.exceptions.RequestException as e:
            print(f"Error during request for {current_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {current_url}: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting deep crawling process...")
    deep_crawl()

    # Save all extracted text to a file
    output_filename = "ubestream_all_deep_content.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(all_extracted_text))

    print(f"\nCrawling finished.")
    print(f"Total unique URLs visited: {len(visited_urls)}")
    print(f"All content saved to: {output_filename}")
