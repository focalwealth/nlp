#TODO: add fuzzy wuzzy, include feature that identifies long numbers, add a dict to check area codes, check for emails
#download wet file from """https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2018-30/segments/1531676599291.24/wet/CC-MAIN-20180723164955-20180723184955-00638.warc.wet.gz"""
import tldextract
from mrjob.job import MRJob
import requests
from bs4 import BeautifulSoup
import unicodedata
import usaddress
import re
import warc
import gzip
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from difflib import SequenceMatcher



states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA",
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]


def search_domain(domain):
    record_list = []
    print "[*] Trying target domain: %s" % domain

    for index in index_list:
        print "[*] Trying index %s" % index
        cc_url = "http://index.commoncrawl.org/CC-MAIN-%s-index?" % index
        cc_url += "url=%s&matchType=domain&output=json" % domain

        response = requests.get(cc_url)

        if response.status_code == 200:
            records = response.content.splitlines()
            for record in records:
                record_list.append(json.loads(record))
            print "[*] Added %d results." % len(records)
    print "[*] Found a total of %d hits." % len(record_list)
    return record_list


def get_data(url):
    r = requests.get(url)
    content = r.content
    soup = BeautifulSoup(content, "html.parser")
    data= unicodedata.normalize('NFKD', soup.text).encode('ascii','ignore')
    #print data
    return data


def get_address(data):
    parsed=usaddress.parse(data)
    #print ("\n\n\nHere is the parsed\n\n\n", parsed)
    address=[]
    for item in parsed:
        if item[1] != 'Recipient':
            address.append(item[0])
    address_text=" ".join(address)
    if "\n" in address_text:
        address_text=address_text.replace("\n", "")
    regexp = "[0-9]{3,4} .{1,20} .{1,20} .{1,20}, [A-Z]{2}" #identifies format: "2533 Jackson Avenue Evanston, IL"
    address_main = re.findall(regexp, address_text)
    add_main=[]
    for add in address_main:
        if add[-2:] not in states:
            add_main.append(add)
    for delt in add_main:
        address_main.remove(delt)

    return address_main


def get_address_reversed(data):
    regexp = "[0-9]{3,4} .{1,20} .{1,20} .{1,20}, [A-Z]{2}(?: [0-9]{5})?"  # identifies format: "2533 Jackson Avenue Evanston, IL"
    addresses = re.findall(regexp, data)
    print addresses
    remover = []
    for address in addresses:
        #cleaning non-states
        if "\n" in address:
            address = address.replace("\n", "")
        if address[-2:] not in states:
            remover.append(address)
            pass
        else:
            parsed=usaddress.parse(address)
            address=[]
            for item in parsed:
                if item[1] != 'Recipient':
                    address.strip(item[0])
            address=" ".join(address)

    for delt in remover:
        if delt in addresses:
            addresses.remove(delt)

    return addresses


def get_email(data):
    #emails = re.findall(r'[\w\.-]+@[\w\.-]+', data)
    emails= re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}",data)
    return emails


def get_phno(data):
    ph = re.findall(r'(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?',
        data)
    final_list=[]
    for nos in ph:
        actual="".join(nos)
        if len(actual)==10:
                final_list.append(actual)
    if len(final_list)>0:
        for items in final_list:
            try:
                new_idx = data.index(items) + len(items)
                if (data[new_idx]) in "1234567890":
                    final_list.remove(items)
            except:
                pass
            try:
                new_idx=data.index(items) - 1
                if (data[new_idx]) in "1234567890":
                    final_list.remove(items)
            except:
                pass
    #include feature that identifies long numbers
    #add a dict to check area codes
            # sum_digits=sum(int(digit) for digit in str(items))
            # checker=int(items[0])*10
            # if sum_digits == checker:
            #     print(items in final_list)
            #     final_list.remove(items)
    return list(set(final_list))


def update_flat_dict(url,address,email_ids,ph_nums):
    flat_dict["URL"].append(url)
    flat_dict["address"].append(address)
    flat_dict["email_ids"].append(email_ids)
    flat_dict["ph_nums"].append(ph_nums)


def warc_scraping(wet_file,limit=False):
    with gzip.open(wet_file, mode='rb') as gzf:
        i=0
        for record in warc.WARCFile(fileobj=gzf):
            website= record.header.get('warc-target-uri', 'none')
            data = record.payload.read()
            details= [get_address(data), get_email(data), (get_phno(data))]
            if details!=[[],[],[]]:
                update_flat_dict(website, details[0],details[1],details[2])
                print(i,website, details[0],details[1],details[2])
            i =i +1
            if i ==limit:
                break
    print(len(flat_dict["URL"]))

def url_scraping(urls):
    for url in urls:
        data = get_data(url)
        details = [get_address(data), get_email(data), get_phno(data)]
        print details


if __name__ == "__main__":
    index_list=[]
    flat_dict={"URL":[],"address":[],"email_ids":[],"ph_nums":[]}
    #warc_scraping('/Users/Sheel.Saket/Downloads/CC-MAIN-20180723164955-20180723184955-00638.warc.wet.gz',3000)
    url_scraping(urls= ["http://bookkeeping-results.com/resources/tax-retention-guide/", "http://wildfirerestaurant.com/schaumburg/", "https://www.datacubes.com/company", "https://www.discover.com/credit-cards/help-center/contact-us/?ICMPGN=PUB_FTR_QL_CONTACT", "https://giordanos.com/locations/gurnee", "http://www.bitboost.com/ref/international-address-formats.html", "http://www.evanstonliving.com/808-816-forest-apartments/default.aspx?_yTrackUser=MzUyMzk3MTc5Izg5MjY0MjY1NQ==-5uqsn91gGOY=&_yTrackVisit=NjkxNTcwNDM5IzE5NDEzOTgxMDA%3d-5ThhaDM%2bpTA%3d&_yTrackReqDT=19010520182408"])