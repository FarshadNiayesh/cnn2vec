import requests
import os
import tempfile
import zipfile
from utils import API_KEY


headers = {'Authorization': API_KEY}
base_0 = 'http://188.166.23.190/api/company_list'
base_1 = 'http://188.166.23.190/api/document_list'
base_2 = 'http://188.166.23.190/api/document'
base_3 = 'http://188.166.23.190/api/documents'
text_location = os.path.join(os.getcwd(), 'corpus')
print('Downloading documents to {}'.format(text_location))

r = requests.get(base_0, headers=headers)
company_list = r.json()

document_to_dl = []
for i, company in enumerate(company_list):
    s = 'Downloading document list for {}                         '
    print(s.format(company['name']), end="\r")
    params = dict(query=company['company_id'], query_by='company_id')
    r = requests.get(base_1, params=params, headers=headers)
    if r.status_code != 200:
        print('Error for {}'.format(company['company_id']))
        continue
    document_list = r.json()
    filename = os.path.join(text_location, 'doc_{}.txt')
    for document in document_list:
        if not os.path.isfile(filename.format(document['document_id'])):
            document_to_dl.append(document['document_id'])
print()

chunk_size = 1024
for i in range(0, len(document_to_dl), chunk_size):
    print('{:02}% downloaded'.format(i/len(document_to_dl)), end="\r")
    params = dict(ids=document_to_dl[i:i+chunk_size])
    r = requests.post(base_3, json=params, headers=headers, stream=True)
    hndl, tmp = tempfile.mkstemp(suffix=".zip")
    with open(tmp, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1000):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    zip_ref = zipfile.ZipFile(tmp, 'r')
    zip_ref.extractall(text_location)
    zip_ref.close()
    os.close(hndl)
    os.remove(tmp)
