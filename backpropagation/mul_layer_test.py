# coding: utf-8
from layers import MulLayer

# ---------------------------------
# AddLayer動作確認
# ---------------------------------
apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer   = MulLayer()

# forword
apple_total = mul_apple_layer.forword(apple, apple_num)
total_price = mul_tax_layer.forword(apple_total, tax)
print('合計金額は' + str(total_price))

# backword(りんごの値段、りんごの個数の微分を求める)
# まずはりんごの合計金額 * 税率の計算から、りんごの合計金額に対する微分を求める。
d_apple_total, _           = mul_tax_layer.backword(1)
d_apple_price, d_apple_num = mul_apple_layer.backword(d_apple_total)

print('りんごの値段に対する微分は' + str(d_apple_price))
print('りんごの個数に対する微分は' + str(d_apple_num))
