# Multiply Layer Test
import sys
import os
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import Multiply
except ImportError:
    raise ImportError("Library Module Can Not Found")

apple = 100
applecount = 5
discount = 0.9

# layers
multiply_appleprice = Multiply()
multiply_discountprice = Multiply()

# foward propagation
appleprice = multiply_appleprice.forward(apple, applecount)
print(f'appleprice = {appleprice}')

discountprice = multiply_discountprice.forward(appleprice, discount)
print(f'discountprice = {discountprice}')

# backward propagation
ddiscountprice = 1

dappleprice, ddiscount = multiply_discountprice.backward(ddiscountprice)
print(f'dappleprice = {dappleprice}, ddiscount={ddiscount}')

dapple, dapplecount = multiply_appleprice.backward(dappleprice)
print(f'dapple = {dapple}, dapplecount={dapplecount}')


