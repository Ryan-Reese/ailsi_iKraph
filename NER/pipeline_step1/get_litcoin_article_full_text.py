import argparse
import re
import json
import requests
from bs4 import BeautifulSoup

# function for getting pmids from the dataset
def get_pmids(dataset):
    pmids = []
    with open(dataset, "r", encoding="utf-8") as f:
        for line in f:
            pmid = json.loads(line.strip())["document_id"]
            if not pmid in pmids:
                pmids.append(pmid)
    print(f"Find {len(pmids)} PMIDs in the dataset")
    return pmids

# function for getting pmcids for a list of pmids
def get_pmcids(pmids):
    pmcids = []
    base_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    for i in range(0, len(pmids), 200):
        if i+200 <= len(pmids):
            url = f"{base_url}?ids={','.join(pmids[i:i+200])}&idtype=pmid&format=json"
        else:
            url = f"{base_url}?ids={','.join(pmids[i:])}&idtype=pmid&format=json"
        req = requests.get(url)
        for rec in req.json()['records']:
            if 'pmcid' in rec:
    #             print(rec['pmcid'])
                pmcids.append(rec['pmcid'])
    print(f"Find {len(pmcids)} PMCIDs for {len(pmids)} PMIDs")
    return pmcids

# function for getting full text
def get_full_text(pmcids):
    fulltext_data = {}
    base_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc"
    def get_xml_full_text(soup):
        for art in soup.find_all('article'):
            if art.find('body'):
                pmid = art.find('article-id', attrs={'pub-id-type':'pmid'}).text
                pmcid = art.find('article-id', attrs={'pub-id-type':'pmc'}).text
                title = art.find('article-title').text
                abstract = art.find('abstract').text
                body = art.find('body')
                if body.find('sec', attrs={'sec-type':'supplementary-material'}):
                    body.find('sec', attrs={'sec-type':'supplementary-material'}).decompose()
                for sec in body.find_all('sec'):
                    if sec.find('title'):
                        if 'Competing interests' in sec.find('title').text:
                            sec.decompose()
                            continue
                        if "Authors' contributions" in sec.find('title').text:
                            sec.decompose()
                            continue
                        if "Availability and requirements" in sec.find('title').text:
                            sec.decompose()
                            continue
                        if "Pre-publication history" in sec.find('title').text:
                            sec.decompose()
                            continue
                        if "Abbreviations" in sec.find('title').text:
                            sec.decompose()
                            continue
                    if sec.find_all('fig'):
                        for fig in sec.find_all('fig'):
                            fig.decompose()
                    if sec.find_all('table-wrap'):
                        for tbl in sec.find_all('table-wrap'):
                            tbl.decompose()
                    if sec.find_all('table'):
                        for tbl in sec.find_all('table'):
                            tbl.decompose()
                    if sec.find_all('title'):
                        for ttl in sec.find_all('title'):
                            ttl.decompose()
                fulltext = body.text #encode('ascii','ignore').decode('utf-8').strip()
                fulltext = re.sub(r'\n+', ' ', fulltext)
                fulltext = re.sub(r'\s+\[[0-9\,\-]+\]|\[[0-9\,\-]+\]', '', fulltext)
                fulltext = re.sub(r'\s+\([^\)]*?et al\.?[^\(]*?\)|\([^\)]*?et al\.?[^\(]*?\)', '', fulltext)
                fulltext = re.sub(r'\\documentclass.*?\$\$\\end\{document\}', '', fulltext)
                fulltext_data[pmid] = {'pmid':pmid, 'pmcid':f'PMC{pmcid}',
                                        'title':title, 'abstract':abstract, 'fulltext':fulltext.strip()}

    for i in range(0, len(pmcids), 200):
        if i+200 <= len(pmcids):
            pmcurl = f"{base_url}&id={','.join([i[3:] for i in pmcids[i:i+200]])}"
        else:
            pmcurl = f"{base_url}&id={','.join([i[3:] for i in pmcids[i:]])}"
        pmcreq = requests.get(pmcurl)
        soup = BeautifulSoup(pmcreq.text, 'xml')
        get_xml_full_text(soup)
    return fulltext_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get full text for articles in litcoin training and test data")
    parser.add_argument("--input_file", "-i", type=str, required=True, help="Path and name of the file of litcoin training or test data, in json format")
    parser.add_argument("--output_file", "-o", type=str, required=True, help="Path and name of the file for saving full text, in json format")

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    # get pmids of the dataset
    pmids = get_pmids(input_file)

    # get pmcids of the dataset
    pmcids = get_pmcids(pmids)

    # get full text for training data
    fulltext = get_full_text(pmcids)

    # print number of articles having full text
    print(f"Get full text for {len(fulltext)} articles")

    # save full text file
    json.dump(fulltext, open(output_file, 'w', encoding='utf-8'))
