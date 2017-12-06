# coding: utf-8
from layers import AddLayer, MulLayer

# ---------------------------------
# AddLayer動作確認
# ---------------------------------
apple      = 100
apple_num  = 2
orange     = 150
orange_num = 3
tax        = 1.1

# layer
mul_apple_layer        = MulLayer()
mul_orange_layer       = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer          = MulLayer()

# forword
apple_amount  = mul_apple_layer.forword(apple, apple_num)
orange_amount = mul_orange_layer.forword(orange, orange_num)
amount        = add_apple_orange_layer.forword(apple_amount, orange_amount)
total_amount  = mul_tax_layer.forword(amount, tax)
print('合計金額は' + str(total_amount))

# backword(りんごの値段、りんごの個数、みかんの値段、みかんの個数に対する微分)
d_amount, _                   = mul_tax_layer.backword(1)
d_apple_total, d_orange_total = add_apple_orange_layer.backword(d_amount)
d_apple_price, d_apple_num    = mul_apple_layer.backword(d_apple_total)
d_orange_price, d_orange_num  = mul_orange_layer.backword(d_orange_total)

print('りんごの値段に対する微分は' +  str(d_apple_price))
print('りんごの個数に対する微分は' +  str(d_apple_num))
print('みかんの値段に対する微分は' +  str(d_orange_price))
print('みかんの個数に対する微分は' +  str(d_orange_num))
