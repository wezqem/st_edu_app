import streamlit as st

st.title("Streamlit 超入門")

with st.expander('Streamlit 超入門について'):
    st.markdown("""
            このアプリケーションは、Streamlitアプリの構築を始めるにあたって、必要最低限の知識をつけるためにご用意しました。
            各ウィジェットの実装から、ウィジェットに対してデータを渡してデータ解析が可能なアプリケーションの構築の方法を紹介します。
    """)

calc = st.selectbox('項目を選択してください',['セットアップ方法','文字入力','スライダー','チャート','セレクトボックス','チェックボックス(ボタン)','アップローダー',
                                   '画像読み込み','デモアプリ作成①','デモアプリ作成②','デモアプリ作成③'])


st.sidebar.header(("Streamlitについて"))
st.sidebar.markdown((
    "[Streamlit](https://streamlit.io) は、Pythonを使用してインタラクティブでデータドリブンなWebアプリケーションを作成することができるPythonライブラリです。"
))

st.sidebar.header(("リソース"))
st.sidebar.markdown((
    """
- [Streamlit ドキュメント](https://docs.streamlit.io/)
- [チートシート](https://docs.streamlit.io/library/cheatsheet)
- [書籍](https://www.amazon.com/dp/180056550X) (データサイエンスのためのStreamlit入門)
- [ブログ](https://blog.streamlit.io/how-to-master-streamlit-for-data-science/) (データサイエンスのためのStreamlitの使いこなし方)
- [API reference](https://docs.streamlit.io/library/api-reference)(機能の詳細)
- [GitHub](https://github.com/wezqem/st_edu_app/blob/main/st_edu_app.py)(ソースコード)
"""
))


if calc == 'セットアップ方法':
    st.header('ここではStreamlitのセットアップ方法を紹介します。')
    st.subheader('準備')
    st.markdown("""まずはメインディレクトリに任意の名前のpythonファイルを作成してください。ここでは便宜上`app.py`とします。""")
    st.subheader('Streamlitライブラリのインストール')
    st.code('pip install streamlit')
    st.subheader('Streamlitアプリの起動')
    st.code('streamlit run app.py')
    st.info('起動した時点では何も入力していないため、何も表示されません。')
    st.write('Streamlitアプリを作成するにあたり、下記のコードでインポートしておきましょう。')
    st.code('import streamlit as st')
    st.write('また、今回のデモで使用するライブラリをインストールしておいてください。')
    st.code("pip install pandas numpy plotly scikit-learn opencv-python-headless")

    st.divider()
    st.subheader('おまけ')
    st.markdown("""
                Streamlitでは、アプリケーションの中身を書き換えた際、画面に反映させるためにはリロードし直す必要があります。
                Streamlitアプリケーションファイルと同ディレクトリ内に`.streamlit`というディレクトリを作成し、その中に`config.toml`という設定ファイルを作成します。
                このファイルに以下の内容を記載する。
                ```
                [server]
                runOnSave = true
                ```
                この設定を行うことで、アプリケーションのファイルを更新するたび、自動でリロードが入るため楽ができます。
                
                
                ディレクトリ関係は以下のようになります。
                ```
                (任意のディレクトリ)
                ├── app.py
                └── .streamlit/
                    └── config.toml 
                ```
                """)

if calc == '文字入力':
    st.header('ここでは文字入力のコードについて紹介します')
    st.markdown("""
    今回紹介するのは
    - st.title
    - st.header
    - st.subheader
    - st.write
    """)
    st.divider()
    st.subheader('st.title')
    st.code('st.title("タイトル")')
    st.title('タイトル')
    st.write('タイトルを表示できます。')
    st.divider()
    st.subheader('st.header')
    st.code('st.header("ヘッダー")')
    st.header('ヘッダー')
    st.write('ヘッダーを表示できます。')
    st.divider()
    st.subheader('st.subheader')
    st.code('st.subheader("サブヘッダー")')
    st.subheader('サブヘッダー')
    st.write('サブヘッダーを表示できます。')
    st.divider()
    st.subheader('st.write')
    st.code('st.write("hogehoge")')
    st.write('hogehoge')
    st.write('文字を表示できます。')
    st.divider()
    st.write('他にも文字を表示するための機能はたくさんあります。サイドバーに公式ドキュメントやチートシートを載せていますのでご確認下さい。')

if calc == 'スライダー':
    st.header('ここではスライダーのコードについて紹介します')
    st.subheader('st.slider')
    st.code('st.slider("スライダー", 0, 100, 50, 10)')
    st.slider('スライダー', 0, 100, 50, 10)
    st.write('引数にラベル、最小値、最大値、デフォルト値、ステップを指定しています。')

    st.subheader('範囲指定させることもできます。')
    st.code('st.slider("スライダー", 0, 100, (20, 50))')
    st.slider('スライダー', 0, 100, (20, 50))
    st.markdown("""
                ### スライダーでとった値を表示させる。
                """)
    code = '''
    slider_values = st.slider("スライダー", 0, 100, 50, 10)
    st.write("slider_values :", slider_values)
    '''
    st.code(code, language='python')
    slider_values = st.slider("スライダーの値を取得", 0, 100, 50, 10)
    st.write("slider_values :", slider_values)

    st.markdown("""
                このように、ウィジェットを変数に格納することでウィジェットで取得したデータを表示したり、他のウィジェットに渡すことが可能となる。
                
                日付のデータを取得し、期間をスライダーで指定することも可能です。
                詳しくはドキュメントを参考にしてください。
                """)

if calc == "チャート":
    st.header('ここではチャートのコードについて紹介します')
    st.markdown("""
                まずはデータを用意します。
                    
                    data = {
                
                    '2019/01/01': 100,
                
                    '2019/01/02': 200,
                
                    '2019/01/03': 500,
                
                    '2019/01/04': 250,
                
                    '2019/01/05': 400
                    }
            """)
    st.subheader('st.line_chart')
    st.code('st.line_chart(data)')
    data = {
            '2019/01/01': 100,
            '2019/01/02': 200,
            '2019/01/03': 500,
            '2019/01/04': 250,
            '2019/01/05': 400
            }
    
    st.line_chart(data)
    
    st.subheader('st.bar_chart')
    st.code('st.bar_chart(data)')
    st.bar_chart(data)
    
    st.write('matplotlibやplotlyで作成したグラフを表示することもできます。後半のデモアプリ作成の際に紹介します。')
    
if calc == "セレクトボックス":
    st.header('ここではセレクトボックスのコードについて紹介します')
    st.subheader('st.selectbox')
    st.code('st.selectbox("セレクトボックス", ["アメリカ", "イギリス", "フランス"])')
    st.selectbox("セレクトボックス", ["アメリカ", "イギリス", "フランス"])
    st.write('リストで渡したデータがセレクトボックスに格納されます。')

    st.subheader('st.multiselect')
    st.code('st.multiselect("マルチセレクト", ["アメリカ", "イギリス", "フランス"])')
    st.multiselect("マルチセレクト", ["アメリカ", "イギリス", "フランス"])
    st.write('リストで渡したデータがマルチセレクトに格納されます。複数のデータを同時に選択することができます。')

    st.subheader('選択したデータを表示させる')
    code = '''
    selected_option = st.selectbox("セレクトボックス", ["アメリカ", "イギリス", "フランス"])
    if selected_option == "アメリカ":
        st.write("首都はワシントンです。")
    if selected_option == "イギリス":
        st.write("首都はロンドンです。")
    if selected_option == "フランス":
        st.write("首都はパリです。")
    '''
    st.code(code, language='python')
    selected_option = st.selectbox("セレクトボックスのデータを表示", ["アメリカ", "イギリス", "フランス"])
    if selected_option == "アメリカ":
        st.write("首都はワシントンです。")
    if selected_option == "イギリス":
        st.write("首都はロンドンです。")
    if selected_option == "フランス":
        st.write("首都はパリです。")

if calc == "チェックボックス(ボタン)":
    st.header('ここではチェックボックスやボタンのコードについて紹介します')
    st.subheader('st.checkbox')
    st.code('st.checkbox("チェックボックス")')
    st.checkbox("チェックボックス")
    st.write('チェックボックスでは、チェックしたらTrue、されなかったらFalseが返ります。')

    st.subheader('st.radio')
    st.code('st.radio("ラジオボタン", ["アメリカ", "イギリス", "フランス"])')
    st.radio("ラジオボタン", ["アメリカ", "イギリス", "フランス"])

    st.subheader('st.button')
    st.code('st.button("ボタン")')
    st.button("ボタン")
    
    st.subheader('st.download_button')
    st.code('st.download_button("ダウンロードボタン", "hogehoge.txt")')
    st.download_button("ダウンロードボタン", "hogehoge.txt")

    st.subheader('st.link_button')
    st.code('st.link_button("Go to Docs", "https://docs.streamlit.io/library/api-reference/widgets/st.link_button")')
    st.link_button("Go to Docs", "https://docs.streamlit.io/library/api-reference/widgets/st.link_button")

    st.subheader('チェックしたデータを表示させる')
    code = '''
    checkbox_value = st.checkbox("チェックボックス")
    if checkbox_value:
        st.write("チェックボックスがチェックされました。")
    '''
    st.code(code, language='python')
    checkbox_value = st.checkbox("チェックボックスの結果を返す")
    if checkbox_value:
        st.write("チェックボックスがチェックされました。")

if calc == "アップローダー":
    st.header('ここではアップローダーのコードについて紹介します')
    st.subheader('st.file_uploader')
    code = '''
        st.file_uploader("アップローダー", type=['csv', 'txt', 'png'])
    '''
    st.code(code, language='python')
    st.file_uploader("アップローダー", type=['csv', 'txt', 'png'])
    st.write('200MBまでのファイルをアップロードすることができます。複数のファイルを読み込むことはできません。引数の`type`に読み込むファイルの拡張子をリストで渡すことでファイルの指定ができます。')

    st.write('アップロードされたファイルの内容の表示についてはデモアプリ作成で紹介します。')

if calc == '画像読み込み':
    st.header('ここでは画像の読み込みについて紹介します')
    st.subheader('st.image')
    code = '''
        st.image('https://cdn.pixabay.com/photo/2014/04/13/20/49/cat-323262_640.jpg', caption='猫')
        st.image('https://media.istockphoto.com/id/1082784928/ja/%E3%82%B9%E3%83%88%E3%83%83%E3%82%AF%E3%83%95%E3%82%A9%E3%83%88/%E5%AE%B6%E3%81%AE%E4%B8%AD%E3%81%AE%E5%B0%8F%E3%81%95%E3%81%AA%E7%81%B0%E8%89%B2%E3%81%AE%E3%83%81%E3%83%B3%E3%83%81%E3%83%A9.jpg?s=612x612&w=0&k=20&c=jlX6uMjLeytw5OexFwan_498smO3xo5J0QxpmdJQpfI=', caption='チンチラ', width=200)
    '''
    st.code(code, language='python')
    st.image('https://cdn.pixabay.com/photo/2014/04/13/20/49/cat-323262_640.jpg', caption='猫')
    st.image('https://media.istockphoto.com/id/1082784928/ja/%E3%82%B9%E3%83%88%E3%83%83%E3%82%AF%E3%83%95%E3%82%A9%E3%83%88/%E5%AE%B6%E3%81%AE%E4%B8%AD%E3%81%AE%E5%B0%8F%E3%81%95%E3%81%AA%E7%81%B0%E8%89%B2%E3%81%AE%E3%83%81%E3%83%B3%E3%83%81%E3%83%A9.jpg?s=612x612&w=0&k=20&c=jlX6uMjLeytw5OexFwan_498smO3xo5J0QxpmdJQpfI=', caption='チンチラ')
    
if calc == 'デモアプリ作成①':
    st.subheader('データを読み込んで、グラフで表示しよう！')
    st.write('アヤメのデータセットを読み込んで、データフレームとplotlyのグラフを表示する簡易的なアプリを作成します。')
    code = '''
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    iris_data = px.data.iris()
    st.dataframe(iris_data)
    
    fig = px.scatter(iris_data, x="sepal_length", y="sepal_width", color="species")
    st.plotly_chart(fig)
    '''
    st.code(code, language='python')

    import pandas as pd
    import plotly.express as px
    import streamlit as st

    iris_data = px.data.iris()
    st.dataframe(iris_data)
    
    fig = px.scatter(iris_data, x="sepal_length", y="sepal_width", color="species")
    st.plotly_chart(fig)

if calc == 'デモアプリ作成②':
    st.subheader('画像データを読み込んで画像処理を行おう！')
    st.write('画像データを読み込んで、OpenCVを用いた画像処理を行います。')
    code = '''
    import requests
    import cv2
    import streamlit as st
    import numpy as np

    url = 'https://cdn.pixabay.com/photo/2014/04/13/20/49/cat-323262_640.jpg'
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)
    '''
    st.code(code, language='python')
    
    import requests
    import cv2
    import streamlit as st
    import numpy as np

    url = 'https://cdn.pixabay.com/photo/2014/04/13/20/49/cat-323262_640.jpg'
    response = requests.get(url)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)

    st.write('上記のコードではURLから画像を読み込んだが、読み込んだ画像ファイルを表示させるコードも下記に示す。')

    code = '''
    import cv2
    import streamlit as st
    import numpy as np

    def load_image(image_file):
        image_bytes = image_file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    image_file = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'])
    if image_file:
        st.image(load_image(image_file))
    '''
    st.code(code, language='python')

    import cv2
    import streamlit as st
    import numpy as np

    def load_image(image_file):
        image_bytes = image_file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    image_file = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'])
    if image_file:
        st.image(load_image(image_file))

    st.markdown("""
                画像ファイルをバイト型で読み込み、デコードすることで画像を表示することができます。通常のOpenCVとは読み込み方が異なるので注意してください。
                
                また、OpenCVで画像を読み込んだ際には色空間はGBR値で読み込まれるため、色が反転しています。画像を表示する前には必ずGBR値→RGB値に変換しましょう。
                """)
    
    st.divider()
    st.subheader('コードの書き方')
    st.markdown("""
                コードを書く際には、上記の2例のように
                
                - Pythonの処理内容を書いて、結果をStreamlitのコンポーネントに渡す書き方
                - 関数を作成して、Streamlitのコンポーネント内で関数を走らせる書き方

                が一般的な書き方です。

                好みで使い分けて下さい。
                
                """)
    
if calc == 'デモアプリ作成③':
    st.subheader('機械学習アプリを作成しよう！')
    st.write('ここでは簡易的な機械学習アプリを作成します。用いるデータはアヤメのデータセットを使用します。')
    code = '''
    import pandas as pd 
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import streamlit as st

    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    st.dataframe(df)

    if st.checkbox('モデル作成'):
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        n = st.slider('n_neighbors', 1, 10, 5)
        model = KNeighborsClassifier(n_neighbors=5)

        if st.button('学習'):
            model.fit(X_train, y_train)
            st.write('学習が完了しました。')
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write('正解率:', accuracy)
    '''
    st.code(code, language='python')

    import pandas as pd 
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import streamlit as st

    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    st.dataframe(df)

    if st.checkbox('モデル作成'):
        X = data.data
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        n = st.slider('n_neighbors', 1, 10, 5)
        model = KNeighborsClassifier(n_neighbors=5)

        if st.button('学習'):
            model.fit(X_train, y_train)
            st.write('学習が完了しました。')
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write('正解率:', accuracy)
    
    
