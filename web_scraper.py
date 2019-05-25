from bs4 import BeautifulSoup
import requests
import re

for i in range(2, 210, 2):
    url = 'https://www.sparknotes.com/nofear/shakespeare/hamlet/page_{}/'.format(i)
    response = requests.get(url)
    content = BeautifulSoup(response.content, 'html.parser')

    # print(content)

    originals = content.findAll('td', attrs={'class':
        'noFear__cell noFear__cell--original'})
    moderns = content.findAll('td', attrs={'class':
        'noFear__cell noFear__cell--modern'})

    translations = list(zip(originals, moderns))

    for o, m in translations:
        os = []
        for x in o.findAll('div'):
            if not x.findAll('span'):
                os.append(x.text.strip())
            else: os.append(x.text[len(x.findAll('span')[0].text) + 1:])

        original = ' '.join(os)
        modern = ' '.join([' '.join(x.text.split()) for x in m.findAll('div')])
        print('{}\t{}'.format(original, modern))
