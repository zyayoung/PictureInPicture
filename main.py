import cv2,os,time
import numpy as np
import multiprocessing as mp

img_list = os.listdir('smallimgs/')
img_list.sort()
x_train = []
tot = len(img_list)
for cnt, img_name in enumerate(img_list):
    tags = img_name.split(' ')
    img = cv2.imread('smallimgs/'+img_name)
    if(img.shape[0]==300 and 210<=img.shape[1]<=213):
        img = cv2.resize(img,(53,75))
        x_train.append(img)
    if cnt % 1000 == 0:
        print(
            cnt,
            '/',
            tot,
            end='\r'
        )
x_train = np.array(x_train, dtype='int')

target_img = cv2.imread('bigimg.jpg')
target_img = cv2.resize(target_img, (53*20,75*20))
target_img = np.array(target_img, dtype='int')

def find_best(target_img):
    i=0
    ans = np.array([])
    target_img_x = np.array([target_img]).repeat(1000, axis=0)
    while i<len(x_train):
        ans = np.concatenate([ans,np.square(x_train[i:i+1000] - target_img_x[:min(1000,len(x_train)-i)]).mean((1,2,3))])
        i+=1000
    ans += np.random.randint(0,1000, ans.shape)
    return ans.argmin()

pool = mp.Pool()
m = pool.map(find_best,[target_img[j*75:(j+1)*75,i*53:(i+1)*53] for i in range(20) for j in range(20)])

target_img_y = np.zeros_like(target_img)
m = np.array(m).reshape(20,20)
for i in range(20):
    for j in range(20):
        target_img_y[j*75:(j+1)*75,i*53:(i+1)*53] = x_train[m[i,j]]

cv2.imwrite('o.jpg',np.uint8(target_img_y))
 