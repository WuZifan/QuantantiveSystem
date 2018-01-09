import datetime


price_str='30.14 29.85 26.36 32.82'
print(type(price_str))

if not isinstance(type(price_str),str):
    print('not a str')
else:
    print('str')

price_array=price_str.split(' ')
print(price_array)
price_array.append('19.11')
print(price_array)

date_base=datetime.datetime(2017,1,1).__format__('%Y-%m-%d')
print(date_base)

mydate=20170101

date_price=[(str(mydate+index),price) for index, price in enumerate(price_array)]
print(date_price)

from collections import namedtuple

stock_namedtuple=namedtuple('stock',('date','price'))
stock_namedtuple_list=[stock_namedtuple(date,price) for date,price in date_price]
print(stock_namedtuple_list)

stock_dict={date:price for date,price in date_price}
print(stock_dict)

# 仅仅会按照输入的顺序排序，还是尴尬。
from collections import OrderedDict
stock_order_dict=OrderedDict((date,price) for date,price in date_price)
print(stock_order_dict)

print(min(zip(price_array)))

# 自定义函数
def find_second_max(dict_array):
    stock_prices_sorted=sorted(zip(dict_array.values(),dict_array.keys()))
    # stock_prices_sorted=sorted(zip(dict_array.keys(),dict_array.values()))
    return stock_prices_sorted[-2]

print(find_second_max(stock_dict))

# lambda函数
find_second_max_lambda=lambda dict_array:sorted(zip(dict_array.values(),dict_array.keys()))[-2]
print(find_second_max_lambda(stock_dict))

# 高阶函数
price_float_array=[float(price_str) for price_str in stock_order_dict.values()]
pp_array=[(price1,price2) for price1,price2 in zip(price_float_array[:-1],price_float_array[1:])]
print(price_float_array)
print(pp_array)

increase_rate=[round((b-a)/a,3) for a,b in pp_array]
print(increase_rate)
increase_rate.insert(0,0)
print(increase_rate)