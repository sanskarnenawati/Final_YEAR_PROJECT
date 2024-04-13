from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

service = Service(executable_path='/home/abiggj/Projects/Python/chromedriver')
options = webdriver.ChromeOptions()
options.add_argument('--headless')

driver = webdriver.Chrome(service=service, options=options)

for year in range(2015, 2021):
    for month in range(1, 13):
        links = dict()
        driver.get(f"https://economictimes.indiatimes.com/archive/year-{year},month-{month}.cms")
        table = driver.find_element(By.TAG_NAME, "table")
        anchors = table.find_elements(By.TAG_NAME, 'a')
        for anchor in anchors:
            date = f"{anchor.text if anchor.text != '' else '1'}-{month}-{year}"
            links[date] = anchor.get_attribute('href')
        news = []
        for (date, news_link) in links.items():
            driver.get(news_link)
            page = driver.find_element(By.CLASS_NAME,'pagetext')
            table = page.find_elements(By.TAG_NAME, 'tbody')[1]
            anchors = table.find_elements(By.TAG_NAME, 'a')
            current_news = [element.text for element in anchors]
            current_news.insert(0, date+'\n')
            current_news.append('\n')
            news.extend(current_news)

        date = str(month)+'-'+str(year)
        print(date)
        with open('./news/'+date+'.txt', 'w+') as file:
                file.write(' '.join(news))
