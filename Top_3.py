import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mpdates
import streamlit as st

order_items = pd.read_csv('olist_order_items_dataset.csv', parse_dates=["shipping_limit_date"])
products = pd.read_csv('olist_products_dataset.csv')

df = pd.merge(order_items[['seller_id','product_id']],products[['product_id','product_category_name']],on='product_id').groupby(['product_category_name',
                                                                                                                            'seller_id']).count().sort_values('product_id',ascending=False)

df = df.reset_index(level='seller_id').sort_index()
df_top_categoria = pd.DataFrame()
for i in df.index.unique():
    df_top_categoria = pd.concat([df_top_categoria,df[df.index == i].sort_values(by='product_id',ascending=False).head(3)])
   
top=['beleza_saude', 'relogios_presentes', 'cama_mesa_banho', 'esporte_lazer', 'informatica_acessorios',
     'moveis_decoracao', 'cool_stuff', 'utilidades_domesticas']
genre = st.sidebar.radio(
     "Elige la categoría",
     top)  
fig, ax = plt.subplots()

ax.bar(df_top_categoria[df_top_categoria.index == genre]['seller_id'],df_top_categoria[df_top_categoria.index == genre]['product_id'], color=['red', 'orange', 'gold'], alpha=0.7)
ax.set_xlabel('ID de vendedor') 
ax.set_ylabel('productos vendidos')
fig.autofmt_xdate()
fig.tight_layout() 
st.header('Top 3 vendedores por categoría')
st.header(genre)  
st.write(fig) 




  

