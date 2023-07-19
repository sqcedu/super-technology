#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 首发 lwebapp.com
import requests
# 谷歌浏览器驱动
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# sleep模块，让程序停止往下运行
from time import sleep

# 设置谷歌浏览器驱动
driver = webdriver.Chrome()

# 手动改为想要下载的视频所在网页地址
url = 'https://www.zxzj.fun/video/1529-1-1.html'

# 打开网页
driver.get(url)

try:
  # 通过元素选择器找到iframe
    iframe = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, '#playleft iframe'))
    )
except:
    driver.quit()

# 获取到网页title，便于直观看到当前下载的视频标题
title = driver.find_elements(By.TAG_NAME, 'title')[
    0].get_attribute('innerHTML')

# 切换到iframe
driver.switch_to.frame(iframe)

# 通过video标签获取视频地址
video = driver.find_elements(By.TAG_NAME, 'video')[0]
video_url = video.get_attribute('src')
print('video', video_url)

# 已经获取到视频地址，可以关闭浏览器
driver.quit()

# 设置请求头信息
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36 Edg/96.0.1054.62'
}

# 请求视频内容
video_content = requests.get(video_url, headers=headers, stream=True)

print("开始下载")

# 视频大小
contentLength = int(video_content.headers['content-length'])

line = '大小: %.2fMB'

# 大小换算
line = line % (contentLength/1024/1024)

# 打印视频总长度
print(line)

# 存储已经下载的长度
downSize = 0

print('video_name', title)

# 分片下载
with open(title+'.mp4', "wb") as mp4:
    for chunk in video_content.iter_content(chunk_size=1024 * 1024):
        if chunk:
            mp4.write(chunk)

            # 记录已下载视频长度，实时输出下载进度
            downSize += len(chunk)
            print('进度：{:.2%}'.format(downSize / contentLength), end='\r')

print("下载结束")