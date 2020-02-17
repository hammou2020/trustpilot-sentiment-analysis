import scrapy


class SubcatSpider(scrapy.Spider):
    name = 'subcat'
    start_urls = [
        "https://www.trustpilot.com/categories"
    ]
    
    def parse(self, response):
        categories = self.get_categories(response)
        
        subcat_urls = []
        for cat in categories:
            subcat_urls += self.get_subcategories_urls(response, cat)
            
        yield {
            'urls': subcat_urls
        }
        
    def get_categories(self, response):
        categories = response.css("li.categoryListItemDesktop___2zR3C").css("a::attr(href)").getall()
        return [c.strip('#') for c in categories]
    
    def get_subcategories_urls(self, response, category):
        rel_urls = response.css(f"div#{category}").css("a::attr(href)").getall()
        return [response.urljoin(rel_url) for rel_url in rel_urls]
