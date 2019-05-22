#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'onbaording2\ML-onboarding'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Problem 3

#%%
N = 5; M = 3
arr = [[10, 2, 5],
[7, 1, 0],
[9, 9, 9],
[1, 23, 12],
[6, 5, 9]]
k = 1


#%%
sorted(arr, key=lambda x: x[k])

#%% [markdown]
# # Problem 1

#%%
import re


#%%
cc_nos = ['4253625879615786', '4424424424442444', '5122-2368-7954-3214', '42536258796157867', '4424444424442444', 
         '5122-2368-7954 - 3214', '44244x4424442444', '0525362587961578']


#%%
for cc in cc_nos:
    if re.match(r'(^[456])(?!\1{3})\d{3}-?(\d)(?!\2{3})\d{3}-?(\d)(?!\3{3})\d{3}-?(\d)(?!\3{3})\d{3}$', cc):
        print('valid')
    else:
        print('Invalid')

#%% [markdown]
# # Problem 2

#%%
s = 'AABCAAADA'; k=3


#%%
i = 0
substring = ''
alpha_arr = [0]*26
while i < len(s):
    alpha_index = ord(s[i])-65
    if alpha_arr[alpha_index] > 0:
        pass
    else:
        substring += s[i]
        alpha_arr[alpha_index] += 1
    if (i+1)%k == 0:
        print(substring)
        substring = ''
        alpha_arr = [0]*26
    i += 1


