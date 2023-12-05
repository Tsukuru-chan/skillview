
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from model import predict #これは、model.pyのこと
import webbrowser







st.set_option("deprecation.showfileUploaderEncoding", False)


#st.title('つくるちゃん')
#st.caption('ものづくりを学ぼう')
st.sidebar.title("つくるのＡＩ-ＷＥＢアプリ")
image = Image.open('TSUKURU.png')
#st.image(image,width=150)
st.sidebar.image(image,width=300)

st.sidebar.write("●私の名前は「つくる」です。\nあなたがアップロードする道具や工具の画像を見分けて、\
                 使い方などを説明します。")
st.sidebar.write("")

global img_source
img_source = st.sidebar.radio("画像のアップロード方法を選んでね。",
                              ("画像ファイルをアップロード","カメラ撮影でアップロード"))
if img_source == "画像ファイルをアップロード":
    img_file = st.sidebar.file_uploader("下の枠内に画像ファイルをドラッグ＆ドロップするか、ボタンを押して画像ファイルを選択してね。", type=["png", "jpg","jpeg"])
elif img_source == "カメラ撮影でアップロード":
    img_file = st.camera_input("カメラ撮影でアップロード")
   

st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write(" ")
st.sidebar.write("【プロフィール】")
st.sidebar.write("バナナ高等学校 工業科、総合ものづくりコースの２年生１７歳です。\
                 「つくる」という名前は、ものづくりが好きなパパが付けてくれました。\
                 \n好きな食べ物は熟す前のバナナ！！微妙に青臭いのがハオい。そして誕生日は８月７日…バナナの日！(なんてこったぁ～） \
                 \nパパに似て、ものづくりが好きで、特に木材加工や電子工作が大好き。\
                 最近は、パパのクルマの改造や裏の畑で野菜づくりのお手伝いもしています。\
                 運動音痴でスポーツが苦手（…ってかキライ）だけど、作業で動きやすい体育着でいるのが自分らしくておK。\
                 \n中学時代所属した吹奏楽部で出会ったサックスという楽器のおかげで、今は社会人に混じってバンド活動もめちゃ楽しんでます♪\
                 DTM（デスクトップミュージック）やプログラミング、釣りやお料理などなど、やりたいことがあり過ぎて困ったもんだぁ～。\
                 \n「多芸は無芸」ということわざ通り、どれも中途半端ですが、よろしくお願いします。\
                 ちなみに、同級生の「白根くん」は、ただのお友達。")
image = Image.open('SIRANE.png')
st.sidebar.image(image,width=150,caption="白根くん")

if img_file is not None:
    with st.spinner("推定中..."):
        img = Image.open(img_file)
        if img_source=="画像ファイルをアップロード":
            st.image(img, caption="アップロードされた画像", width=480)
            st.write("")

        # 予測
        results = predict(img)

        # 結果の表示
        #st.subheader("判定結果")        
        n_top = 1  # 確率が高い順に5位まで返す
        for result in results[:n_top]:
            #st.write(str(round(result[1]*100, 2)) + "%の確率で" + result[0] + "だよね？")
            st.subheader("これって、"+result[0]+"？だよねぇ")
            st.write(str(round(100-result[1]*100,2)) + "％まちがってるかも…てへぺろ")            
            
        # ---------- ボタン ----------
        #st.subheader("下のボタンで説明をはじめるよ")

        
        #if st.button("新しいタブで説明を見る"):
            #st.write("Good morinig!")
        #    webbrowser.open_new_tab('https://hibiki-press.tech/python/webbrowser_module/1884')
        
        url = "https://tsukuru-chan.github.io/test/"
        st.subheader(result[0]+"の説明をしますか？ [はい](%s)" % url)
        st.write("（新しいタブが開かれます。）")
        #st.markdown("check out this [link](%s)" % url)




