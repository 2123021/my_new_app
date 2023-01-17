import streamlit as st

# ウィンドウのタイトルを設定
st.title("Number Comparison App")

# ユーザーに2つの整数を入力するように促す
num1 = st.number_input("Enter a number:")
num2 = st.number_input("Enter another number:")

# 2つの整数を比較し、結果を表示する
if num1 > num2:
    st.success("The first number is greater.")
elif num1 < num2:
    st.success("The second number is greater.")
else:
    st.success("The two numbers are equal.")
