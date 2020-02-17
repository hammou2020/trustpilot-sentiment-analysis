import scrapy
import json

class CompanySpider(scrapy.Spider):
    name = 'company'
   
    def start_requests(self):
        with open("subcats.json", 'r') as f:
            data = json.load(f)     
        for url in data[0]['urls']:
            url = url + '?numberofreviews=0&status=all&timeperiod=0'
            yield scrapy.Request(url=url, callback=self.parse)
         
    def parse(self, response):
        company_rel_urls = response.css(".businessUnitCardsContainer___1Ez9Z a.wrapper___26yB4::attr(href)").getall()
        for url in company_rel_urls:
            yield {
                'subcat_url': response.url,
                'company_url': response.urljoin(url),
            }
            
        # if 'next' button exists, parse next page
        next_url = response.css("a.paginationLinkNext___1LQ14::attr(href)").get()
        if next_url:
            yield scrapy.Request(url=response.urljoin(next_url), callback=self.parse)