## dashboard：
1. 打开dashboard文件夹下的dashapp.py，把522行的app.runserve(debug=True)改成app.run(debug=True)
2. 如运行成功，保持运行并打开网址即可
## 框架：
1. 跑框架（https://imc-prosperity.notion.site/Writing-an-Algorithm-in-Python-658e233a26e24510bfccf0b1df647858）的时候，要在目录下复制一个datamodel.py解决引用问题。
2. 将datamodel的import部分复制到框架文件开头，并把TradingState, Order,Listing,Trade,Observation添加到对datamodel的引用（from datamodel import）中
3. 把result的定义部分
``` python
        result = {}       
```
加在result报错处之前

4. 点击denomination报错处，把两个对应的冒号改成等号。