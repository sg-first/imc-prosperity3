# 框架
1. 跑框架（https://imc-prosperity.notion.site/Writing-an-Algorithm-in-Python-658e233a26e24510bfccf0b1df647858）的时候，要在目录下复制一个datamodel.py解决引用问题。
2. 将datamodel的import部分（from datamodel import OrderDepth,TradingState, Order
from typing import List）复制到框架文件开头。

# 回测器的使用方法
1. 看每一轮文件下的run-backtest.ipynb文件，将原来引进的模块改成现在算法文件的Trader。
2. 改Product、listings、position_limit、fair_calculations、
3. 改market_data、trade_history文件名，file_name以.log结尾
![输入图片说明](%E7%B1%BB.png)
![输入图片说明](%E4%BA%A7%E5%93%81.png)
![输入图片说明](%E4%BF%AE%E6%94%B9%E6%95%B0%E6%8D%AE.png)

# Dashboard
1. 打开dashboard文件夹下的dashapp.py，把522行的app.runserve(debug=True)改成app.run(debug=True)
2. 如运行成功，保持运行并打开网址即可
3. 修改路径为存有日志的文件地址，选择日志

