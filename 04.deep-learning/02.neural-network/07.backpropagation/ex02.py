# Multiply & Add Layer Test
import sys
import os
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Multiply, Add
except ImportError:
    raise ImportError("Library Module Can Not Found")

# apple = 100
# applecount = 3
# orange = 200
# orangecount = 5
# discount = 0.9

apple = 100
applecount = 2
orange = 150
orangecount = 3
discount = 1.1

multiply_appleprice = Multiply()
multiply_orangeprice = Multiply()
add_appleorangeprice = Add()
multiply_discountprice = Multiply()

# forward propagation
appleprice = multiply_appleprice.forward(apple, applecount)
print(f'appleprice = {appleprice}')

orangeprice = multiply_orangeprice.forward(orange, orangecount)
print(f'orangeprice = {orangeprice}')

appleorangeprice = add_appleorangeprice.forward(appleprice, orangeprice)
print(f'orangeprice = {appleorangeprice}')

discountprice = multiply_discountprice.forward(appleorangeprice, discount)
print(f'discountprice = {discountprice}')

# backward propagation
ddiscountprice = 1

dappleorangeprice, ddiscount = multiply_discountprice.backward(ddiscountprice)
print(f'dappleorangeprice = {dappleorangeprice}, ddiscount = {ddiscount}')

dappleprice, dorangeprice = add_appleorangeprice.backward(dappleorangeprice)
print(f'dappleprice = {dappleprice}, dorangeprice = {dorangeprice}')

dapple, dapplecount = multiply_appleprice.backward(dappleprice)
print(f'dapple = {dapple}, dapplecount = {dapplecount}')

dorange, dorangecount = multiply_orangeprice.backward(dappleprice)
print(f'dorange = {dorange}, dorangecount = {dorangecount}')


