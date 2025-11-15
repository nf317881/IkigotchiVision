#%%
harvard_images = 'https://arboretum.harvard.edu/plants/image-search/'
usda_images = 'https://plants.usda.gov/'

test_species_common = 'Trembling Aspen'
test_species_scientific = 'Populus tremuloides'
#%%
# First, install selenium if you don't have it:
# pip install selenium

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
from pathlib import Path
import time
from urllib.parse import quote_plus

def scrape_harvard_arboretum_selenium(species_scientific, species_common, output_dir):
    """Scrape images from Harvard Arboretum using Selenium"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_downloaded = 0

    # Set up Chrome options
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Uncomment to run in background
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    # Try both scientific and common names
    for search_term in [species_scientific, species_common]:
        print(f"\nSearching Harvard Arboretum for: {search_term}")

        driver = None
        try:
            # Initialize the driver
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(30)

            # Navigate to the image search page
            print("Loading search page...")
            driver.get("https://arboretum.harvard.edu/plants/image-search/")
            
            # Wait a bit for page to fully load
            time.sleep(3)
            
            print("Page loaded, looking for search input...")

            # Try multiple possible selectors for the search input
            search_input = None
            selectors = [
                (By.ID, "keyword"),
                (By.NAME, "keyword"),
                (By.CSS_SELECTOR, "input[type='text']"),
                (By.CSS_SELECTOR, "input[name='keyword']"),
                (By.XPATH, "//input[@id='keyword']"),
                (By.XPATH, "//input[@placeholder]")
            ]
            
            for by, selector in selectors:
                try:
                    search_input = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((by, selector))
                    )
                    print(f"Found search input using: {by} = {selector}")
                    break
                except:
                    continue
            
            if not search_input:
                print("Could not find search input. Page source sample:")
                print(driver.page_source[:1000])
                driver.quit()
                continue

            # Clear and enter the search term
            print(f"Entering search term: {search_term}")
            search_input.clear()
            time.sleep(0.5)
            search_input.send_keys(search_term)
            time.sleep(1)

            # Try to submit - either with button click or Enter key
            try:
                search_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
                print("Clicking search button...")
                search_button.click()
            except:
                print("Pressing Enter key instead...")
                search_input.send_keys(Keys.RETURN)
            
            # Wait for navigation and results to load
            time.sleep(5)
            print("Waiting for search results...")
            print(f"Current URL: {driver.current_url}")

            # Try multiple selectors for the image results
            image_elements = []
            image_selectors = [
                (By.CSS_SELECTOR, "img"),
            ]
            
            for by, selector in image_selectors:
                try:
                    print(f"Trying selector: {by} = {selector}")
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((by, selector))
                    )
                    
                    # Get all images
                    image_elements = driver.find_elements(by, selector)
                    
                    if len(image_elements) > 0:
                        print(f"Found {len(image_elements)} image elements using: {by} = {selector}")
                        break
                except Exception as e:
                    continue
            
            if len(image_elements) == 0:
                print("No image elements found. Debugging info:")
                print(f"Current URL: {driver.current_url}")
                print(f"Page title: {driver.title}")
                print("Page source snippet:")
                print(driver.page_source[:2000])
                driver.quit()
                continue
            
            # Give a bit more time for all images to load
            time.sleep(2)
            
            print(f"Processing {len(image_elements)} images")
            
            # Extract image URLs
            image_urls = []
            for idx, img in enumerate(image_elements):
                # Try to get the full-size image URL from data attributes or src
                img_url = (img.get_attribute('data-src') or 
                          img.get_attribute('src') or 
                          img.get_attribute('data-lazy-src'))
                
                # Validate that it's actually an image URL
                if img_url and 'arboretum.harvard.edu' in img_url:
                    # Filter out non-image URLs (like navigation links with #)
                    if img_url.endswith('#') or '?' in img_url.split('/')[-1]:
                        print(f"Skipping non-image URL: {img_url}")
                        continue
                    
                    # Must have a valid image extension or be from the labs.arboretum CDN
                    if ('labs.arboretum.harvard.edu' in img_url or 
                        any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif'])):
                        # Replace thumbnail suffix with full-size image
                        img_url = img_url.replace('_sp.jpg', '.jpg').replace('_sp.png', '.png')
                        image_urls.append(img_url)
                        print(f"Valid image {len(image_urls)-1}: {img_url}")
            
            driver.quit()
            
            print(f"Found {len(image_urls)} valid image URLs to download")
            
            # Download the images
            for idx, img_url in enumerate(image_urls):
                try:
                    # Download image
                    img_response = requests.get(img_url, timeout=10, verify=False)
                    img_response.raise_for_status()

                    # Determine file extension from URL
                    ext = img_url.split('.')[-1].split('?')[0].lower()
                    if ext not in ['jpg', 'jpeg', 'png', 'gif']:
                        ext = 'jpg'

                    # Save with unique filename
                    filename = f"harvard_{search_term.replace(' ', '_')}_{idx}.{ext}"
                    filepath = output_path / filename

                    with open(filepath, 'wb') as f:
                        f.write(img_response.content)

                    print(f"Downloaded: {filename}")
                    images_downloaded += 1
                    time.sleep(0.5)  # Be polite to the server

                except Exception as e:
                    print(f"Error downloading image {idx}: {e}")
            
            # If we found images with this search term, don't try the other
            if images_downloaded > 0:
                return images_downloaded

        except Exception as e:
            print(f"Error with Selenium for {search_term}: {e}")
            if driver:
                driver.quit()

    return images_downloaded


def scrape_plantnet(species_scientific, species_common, output_dir):
    """Scrape images from PlantNet using Selenium"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_downloaded = 0

    # Set up Chrome options
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Uncomment to run in background
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)

    print(f"\nSearching PlantNet for: {species_scientific}")

    driver = None
    try:
        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)

        # Navigate to PlantNet species search
        print("Loading PlantNet search page...")
        driver.get("https://identify.plantnet.org/k-world-flora/species")
        time.sleep(3)

        # Find and fill search input
        print("Looking for search input...")
        search_input = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.NAME, "taxon-search-input"))
        )
        
        search_input.clear()
        search_input.send_keys(species_scientific)
        print(f"Entered search term: {species_scientific}")
        time.sleep(1)

        # Click search button or press Enter
        try:
            search_button = driver.find_element(By.CSS_SELECTOR, "button.btn.btn-primary.btn-sm")
            search_button.click()
            print("Clicked search button")
        except:
            search_input.send_keys(Keys.RETURN)
            print("Pressed Enter")

        # Wait for results
        time.sleep(5)
        print(f"Current URL: {driver.current_url}")

        # Find and click the first species result
        print("Looking for species link...")
        species_link = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, f"a[href*='/k-world-flora/species/'][href*='/data']"))
        )
        species_url = species_link.get_attribute('href')
        print(f"Found species page: {species_url}")
        species_link.click()
        
        # Wait for species page to load
        time.sleep(5)

        # Click on Galleries tab
        print("Clicking Galleries tab...")
        galleries_tab = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a[href='#galleries']"))
        )
        galleries_tab.click()
        time.sleep(3)

        # Scrape images from flower, leaf, and bark sections
        organ_types = {
            'flower': 20,
            'leaf': 20,
            'bark': 20
        }

        for organ, max_images in organ_types.items():
            print(f"\nScraping {organ} images (max {max_images})...")
            
            # Click on the organ section in the sidebar
            try:
                organ_link = driver.find_element(By.CSS_SELECTOR, f"a[href*='#galleries-{organ}']")
                organ_link.click()
                time.sleep(2)
            except Exception as e:
                print(f"Could not find {organ} section link: {e}, skipping...")
                continue

            # Find the section by ID - this gets the anchor div
            try:
                # First find the anchor div
                anchor = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, f"galleries-{organ}"))
                )
                
                # Get the parent section element
                section = anchor.find_element(By.XPATH, "..")
                print(f"Found section: galleries-{organ}")
                
                # Scroll to the section to trigger lazy loading
                driver.execute_script("arguments[0].scrollIntoView(true);", section)
                time.sleep(2)
                
                # Scroll down a bit more to load more images
                driver.execute_script("window.scrollBy(0, 500);")
                time.sleep(2)
                
                # Find gallery items within this section only
                gallery_items = section.find_elements(By.XPATH, ".//div[contains(@class, 'species-gallery-item')]")
                print(f"Found {len(gallery_items)} gallery items")
                
                # Extract images from gallery items
                image_elements = []
                for item in gallery_items:
                    imgs = item.find_elements(By.TAG_NAME, "img")
                    image_elements.extend(imgs)
                
                print(f"Found {len(image_elements)} {organ} images")
                
                # Limit to max_images
                image_elements = image_elements[:max_images]
                
                # Extract image URLs
                for idx, img in enumerate(image_elements):
                    # For lazy-loaded images, src might not be set yet, so scroll to each image
                    driver.execute_script("arguments[0].scrollIntoView(true);", img)
                    time.sleep(0.3)
                    
                    img_url = img.get_attribute('src')
                    
                    if img_url:
                        # Convert from small (/s/) to original (/o/) size
                        img_url_large = img_url.replace('/image/s/', '/image/o/')
                        
                        try:
                            # Download image
                            img_response = requests.get(img_url_large, timeout=10, verify=False)
                            img_response.raise_for_status()

                            # Determine file extension
                            ext = 'jpg'

                            # Save with unique filename
                            filename = f"plantnet_{organ}_{idx}.{ext}"
                            filepath = output_path / filename

                            with open(filepath, 'wb') as f:
                                f.write(img_response.content)

                            print(f"Downloaded: {filename}")
                            images_downloaded += 1
                            time.sleep(0.5)

                        except Exception as e:
                            print(f"Error downloading {organ} image {idx}: {e}")
                
            except Exception as e:
                print(f"Error finding {organ} section: {e}")

        driver.quit()
        
    except Exception as e:
        print(f"Error with PlantNet scraper: {e}")
        if driver:
            driver.quit()

    return images_downloaded


def scrape_usda(species_scientific, species_common, output_dir):
    """Scrape USDA plant database"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSearching USDA for: {species_scientific}")
    
    # USDA uses plant symbols (e.g., POTR5 for Populus tremuloides)
    symbol_map = {
        'Populus tremuloides': 'POTR5'
    }
    
    symbol = symbol_map.get(species_scientific)
    
    if not symbol:
        print(f"USDA symbol not found for {species_scientific}")
        return 0
    
    try:
        profile_url = f"https://plants.usda.gov/home/plantProfile?symbol={symbol}"
        print(f"USDA profile URL: {profile_url}")
        print("Note: USDA scraping requires implementation")
        
    except Exception as e:
        print(f"Error accessing USDA: {e}")
    
    return 0


def scrape_all_sources(species_scientific, species_common, output_dir):
    """Scrape all available sources for plant images"""
    
    print(f"Starting image collection for {species_common} ({species_scientific})")
    print(f"Output directory: {output_dir}")
    
    # Disable SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    total_images = 0
    
    # Try Harvard Arboretum with Selenium
    total_images += scrape_harvard_arboretum_selenium(species_scientific, species_common, output_dir)
    
    # Try PlantNet
    total_images += scrape_plantnet(species_scientific, species_common, output_dir)
    
    # Try USDA
    total_images += scrape_usda(species_scientific, species_common, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Total images downloaded: {total_images}")
    print(f"{'='*60}")
    
    return total_images
#%%
# Run the scraper for Trembling Aspen
output_directory = r'plant_data\Trembling_Aspen'

scrape_all_sources(
    species_scientific=test_species_scientific,
    species_common=test_species_common,
    output_dir=output_directory
)