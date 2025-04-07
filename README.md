## dashboard：
1. 打开dashboard文件夹下的dashapp.py，把522行的app.runserve(debug=True)改成app.run(debug=True)
2. 如运行成功，保持运行并打开网址即可
## 框架：
1. 跑框架（https://imc-prosperity.notion.site/Writing-an-Algorithm-in-Python-658e233a26e24510bfccf0b1df647858）的时候，要在目录下复制一个datamodel.py解决引用问题。
2. 将datamodel的import部分复制到框架文件开头，并把TradingState, Order,Listing,Trade,Observation添加到对datamodel的引用（from datamodel import）中
3. 把result的定义部分
``` python
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
                len(order_depth.sell_orders)))

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
```加在result报错处之前

4. 点击denomination报错处，把两个对应的冒号改成等号。