from sqlalchemy import create_engine
import pandas as pd
import os
import datetime
starttime = datetime.datetime.now()

pd.set_option('display.max_columns',20)  #设置最大显示列
pd.set_option('display.width',250)       #设置宽度
pd.set_option('display.max_rows',100)     #设置最大显示行


# 定义路径
path = r'C:\\Users\\tzxin\\Desktop\\黄维\\自然风压3度、7度+4个辅扇方案\\自然风压3度、7度+4个辅扇方案\\自然风压10度+4个辅扇方案'
# 首先打开文件
files = os.listdir(path)
num = 0
for i in files:
    sExcelFile = path + '\\'+i
    df = pd.read_excel(sExcelFile,header=20,usecols='C:BG')
    df = df.dropna(axis=0,how='all')
    df = df.dropna(axis=1,how='all')

    df['风路编号'] = pd.to_numeric(df['风路编号'],'coerce')

    df = df.dropna(subset = ['风路编号'])

    df = df.dropna(axis=1,how='all')
    df = df.reset_index(drop=True)
    df['filename'] = i
    df['dpath'] = sExcelFile
    nrows = df.shape[0]
    ncols = df.columns.size
    
    engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/test')
    # 上面这句，mysql是数据库类型，mysqlconnector是数据库驱动，root是用户名，123456是密码，localhost是地址，3306是端口，test是数据库名称
    df.to_sql(name='test', con=engine,if_exists="append",index=False,chunksize=100)

    print('导入'+str(nrows)+'行成功，并导入' + i + '成功')
    num = num +1

endtime = datetime.datetime.now()
x = endtime - starttime
print(x.seconds)
print('总共导入'+str(num)+'个文件')
