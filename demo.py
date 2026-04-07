import json
import time
from firecrawl import Firecrawl
from deep_translator import GoogleTranslator

app = Firecrawl(api_key="fc-2e67d7cd0e1c4d16a5112a7d8713ebcf")
pdf_url = 'https://drugstorehasselt.be/wp-content/uploads/2022/12/2022_menu_binnenwerk_aanpassingen_v07_web.pdf'

print("📄 Scraping PDF Menu (Raw Mode)...")

try:
    # 1. Sirf Markdown format me file fetch karo, prompt ka jhanjhat hi khatam!
    scrape_result = app.scrape(
        pdf_url, 
        formats=['markdown']
    )
    
    dutch_markdown = scrape_result.markdown if hasattr(scrape_result, 'markdown') else ""
    
    if not dutch_markdown:
        print("❌ Kuch scrape nahi hua, shayad PDF empty hai ya fetch nahi hui.")
        exit()
        
    print("✅ PDF Scraped Successfully!")
    
    # 2. Text ko English me translate karna
    print("🔄 Translating Dutch Menu to English...")
    translator = GoogleTranslator(source='nl', target='en')
    
    # Bari PDF files ko chunks me divide kar ke translate karte hain
    if len(dutch_markdown) > 4000:
        chunks = [dutch_markdown[i:i+4000] for i in range(0, len(dutch_markdown), 4000)]
        translated_chunks = [translator.translate(chunk) for chunk in chunks]
        english_markdown = "".join(translated_chunks)
    else:
        english_markdown = translator.translate(dutch_markdown)
        
    print("✅ Translation Complete!")

    # 3. Save as a new document to merge with your website data later
    pdf_document = {
        "url": pdf_url,
        "markdown": english_markdown
    }
    
    # Ek alag file me save karte hain
    with open("pdf_menu_translated.json", "w", encoding="utf-8") as f:
        json.dump([pdf_document], f, indent=4, ensure_ascii=False)
        
    print("\n🎉 Success! Menu PDF translate ho kar 'pdf_menu_translated.json' me save ho gayi hai.")

except Exception as e:
    print(f"\n❌ Error aagaya bhai: {e}")

try:
    map_result = app.map(url="https://drugstorehasselt.be/", limit=15)
    
    # Sirf raw URL string nikalna zaroori hai
    raw_urls = []
    if map_result.links:
        for item in map_result.links:
            # Agar item direct string nahi hai aur object hai to usme se URL nikaal lo
            url_str = str(item.url) if hasattr(item, 'url') else str(item)
            raw_urls.append(url_str)
            
    print(f"Found {len(raw_urls)} pages. Starting scraping & translation...")

    all_translated_pages = []
    
    # Dutch se English translate karne ke liye tool
    translator = GoogleTranslator(source='nl', target='en')

    for url in raw_urls:
        print(f"\nProcessing: {url}")
        try:
            scrape_result = app.scrape(
                url, 
                formats=['markdown']
            )
            
            dutch_markdown = scrape_result.markdown if hasattr(scrape_result, 'markdown') else ""
            
            # Agar page khali nahi hai to usay translate karo
            english_markdown = ""
            if dutch_markdown:
                print("🔄 Translating to English...")
                # Bari files ko chunks me translate karte hain
                if len(dutch_markdown) > 4000:
                    chunks = [dutch_markdown[i:i+4000] for i in range(0, len(dutch_markdown), 4000)]
                    translated_chunks = [translator.translate(chunk) for chunk in chunks]
                    english_markdown = "".join(translated_chunks)
                else:
                    english_markdown = translator.translate(dutch_markdown)
            
            all_translated_pages.append({
                "url": url,
                "markdown": english_markdown
            })
            print("✅ Success!")
            
            # API ko rest dene ke liye break
            time.sleep(2) 
            
        except Exception as e:
            print(f"❌ Error on {url}: {e}")

    # JSON file me save karein
    with open("website_data.json", "w", encoding="utf-8") as f:
        json.dump(all_translated_pages, f, indent=4, ensure_ascii=False)

    print(f"\nAll pages translated. and saved to JSON. Total: {len(all_translated_pages)}")

except Exception as e:
    print(f"\n❌ Map process me error aaya: {e}")