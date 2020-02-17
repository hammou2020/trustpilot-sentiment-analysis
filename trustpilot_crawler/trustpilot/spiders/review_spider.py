import json
import re

import scrapy

from ..utils import url_without_query_string


class ReviewSpider(scrapy.Spider):
    name = "review"

    def start_requests(self):
        with open('companies.json', 'r') as f:
            data = json.load(f)
        # urls = [
        #     "https://www.trustpilot.com/review/dogwatch.com"
        # ]
        urls = set([d["company_url"] for d in data])  # Remove duplicates
        urls = sorted(urls)
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        review_cards = response.css('div.review-card')

        trustpilot_url = url_without_query_string(response.url)
        company_header = response.css("div.company-profile-header-wrapper")
        company_url = company_header.css(
            "span.badge-card__title::text").get().strip()
        company_name = company_header.css(
            "span.multi-size-header__big::text").get().strip()
        company_logo = "https:" + \
            company_header.css(
                "img.business-unit-profile-summary__image::attr(src)").get()

        for card in review_cards:
            data = {}
            data["comment"] = card.css(
                'p.review-content__text::text').get().strip()
            data["rating"] = int(
                re.search("\d", card.css('img::attr(alt)').get()).group(0))
            data["trustpilot_url"] = trustpilot_url
            data["company_url"] = company_url
            data["company_name"] = company_name
            data["company_logo"] = company_logo
            yield data

        next_url = response.css("a.next-page::attr(href)").get()
        if next_url:
            yield scrapy.Request(url=response.urljoin(next_url), callback=self.parse)
