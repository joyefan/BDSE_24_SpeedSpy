from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")         # 最大化視窗
options.add_argument("--incognito")               # 開啟無痕模式
options.add_argument("--disable-popup-blocking ") # 禁用彈出攔截

driver_exec_path = './chromedriver.exe'

driver = webdriver.Chrome(options = options,executable_path = driver_exec_path)

driver.get("https://tisvcloud.freeway.gov.tw/history/TDCS/M05A/")

# get 2020 all url
elem_list = driver.find_elements(By.CSS_SELECTOR , "tr > td > a")

url_list = [ elem_list[i].get_attribute("href") for i in range(len(elem_list)) ]

targz_url_list = [ x for x in url_list if "tar.gz" in x ]
targz_url_list = [ x for x in targz_url_list if x[-15:-11] == '2020' ]


# output to csv
df = pd.DataFrame({'url':targz_url_list})

df.to_csv("url_2020.csv",index=False)


# Download tar.gz files
import requests, pandas as pd
import tarfile
import glob
from time import time
from time import sleep

url_ls = pd.read_csv("url_2020.csv")['url']
url_2020 = list(url_ls)

start_time = time()
for url in url_2020:
    date = url[-15:-7]
    response = requests.get(url, stream=True)
    
    with open(f"databyday/{date}.tar.gz", "wb") as file:
        file.write(response.raw.read())
        
    print(f"{date} OK !!")
    sleep(40)
end_time = time()
print(f"Done !! Time Cost : {end_time - start_time}")


# Unzip and save csv
Zips = glob.glob('databyday/*.tar.gz', recursive=True)

start_time = time()
for z in Zips:
    date = z[-15:-7]
    with tarfile.open(z) as tf:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tf, path="databyday/")
    
    csv_ls = glob.glob(f"databyday/**/{date}/**/*.csv", recursive=True)
    
    data = pd.concat(map(lambda x: pd.read_csv(x, header=None), csv_ls))
    data.columns = ["Time","S-Station","E-Station","V-Type","Speed","Count"]
    data.to_csv(f"./databyday/{date}.csv")
    print(f"{date} OK !!!")
end_time = time()
print(f"Done !! Time Cost : {end_time - start_time}")


# concat all csv
csv_ls = glob.glob('databyday/*.csv', recursive=True)

data = pd.concat(map(lambda x: pd.read_csv(x, header=None), csv_ls))
data = data.drop(data[data.index == 0],axis=0)
data = data.drop([0],axis=1)
data.columns = ["Time","S-Station","E-Station","V-Type","Speed","Count"]
data.to_csv("./databyday/2020.csv")
print("2020 OK")