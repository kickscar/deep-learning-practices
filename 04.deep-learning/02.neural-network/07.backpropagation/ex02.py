# Multiply & Add Layer Test
import sys
import os
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import MultiplyLayer, AddLayer
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

multiply_appleprice_layer = MultiplyLayer()
multiply_orangeprice_layer = MultiplyLayer()
add_appleorangeprice_layer = AddLayer()
multiply_discountprice_layer = MultiplyLayer()

# forward propagation
appleprice = multiply_appleprice_layer.forward(apple, applecount)
print(f'appleprice = {appleprice}')

orangeprice = multiply_orangeprice_layer.forward(orange, orangecount)
print(f'orangeprice = {orangeprice}')

appleorangeprice = add_appleorangeprice_layer.forward(appleprice, orangeprice)
print(f'orangeprice = {appleorangeprice}')

discountprice = multiply_discountprice_layer.forward(appleorangeprice, discount)
print(f'discountprice = {discountprice}')

# backward propagation
ddiscountprice = 1

dappleorangeprice, ddiscount = multiply_discountprice_layer.backward(ddiscountprice)
print(f'dappleorangeprice = {dappleorangeprice}, ddiscount = {ddiscount}')

dappleprice, dorangeprice = add_appleorangeprice_layer.backward(dappleorangeprice)
print(f'dappleprice = {dappleprice}, dorangeprice = {dorangeprice}')

dapple, dapplecount = multiply_appleprice_layer.backward(dappleprice)
print(f'dapple = {dapple}, dapplecount = {dapplecount}')

dorange, dorangecount = multiply_orangeprice_layer.backward(dappleprice)
print(f'dorange = {dorange}, dorangecount = {dorangecount}')


