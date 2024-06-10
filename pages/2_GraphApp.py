import streamlit as st
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout='wide')

st.title('グラフ可視化アプリ')

st.markdown('''
            
        ''')

st.code('''

        ''')

st.markdown('''
            このページでは読み込んだデータを用途に合ったグラフを選択することで、データを可視化するためのアプリケーションをハンズオン形式で学んでいきます。\n
            新しく`GraphApp.py`というファイルを作成し、アプリを起動してください。\n
            まずはページのレイアウトを設定し、グラフのタイトルを決めましょう。\n
            以下のコードを記載してください。
''')
st.code('''
        import streamlit as st

        st.set_page_config(layout="wide")
        
        st.title("グラフ可視化アプリ")
        ''')

st.markdown('''
        それでは、データを読み込むためのウィジェットを用意しましょう。\n
        今回はcsvファイルを読み込んで、それをデータとして取得し、Webアプリケーションに渡してみましょう。\n
        以下のコードを記載してください。
            ''')

st.code('''
        uploaded_file = st.file_uploader('csvファイルをアップロードしてください。', type=['csv'])
        ''')

st.markdown('''
        上記のコードにより、ローカルにあるファイルをアップロードすることができます。\n
        `type`にcsvを指定することでcsvファイルの読み込みに限定しています。\n
        複数ファイルのアップロードも可能ですが、今回は必要ないので興味がある方はドキュメントを参考にしてみてください。\n
        また、Streamlitでは読み込むファイルの最大サイズが200MBとなっていますが、`./.streamlit/config.toml`の設定でサイズの変更が可能なので、こちらも随時読み込むファイルのサイズに合わせて変更してみてください。\n
        それでは、読み込んだcsvファイルをデータフレームとして表示させてみましょう。\n
        上記のようにStreamlitで作成するコンポーネントは変数に格納することができ、その変数を使用することで様々な処理にデータを受け渡しすることが可能になります。\n
        この、ファイルが読み込まれたときのイベントを`if文`でコントロールしてみましょう。\n
        以下のコードを追記してください。        
        ''')

st.code('''
        import pandas as pd
        
        if uploaded_file :
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
        ''')

st.markdown('''
        `uploaded_file`がTrueのときにPandasのread_csvメソッドが実行されます。\n
        つまり、ウィジェットにファイルを読み込ませ、ファイルの種類がcsvであると認識されたときにコードが走ります。\n
        上記コードでは`st.dataframe`メソッドを使用してデータフレームを表示していますが、`st.write`メソッドや、単純に`df`のみでも表示することができます。\n
        随時コードを変更して確認してみてください。\n
        それでは、次に読み込んだデータのカラム名を取得し、選択できるようにしてみましょう。\n
        可視化したいデータをユーザー側で自由に選択し、ユーザビリティを高めたいと思います。\n
        以下のコードを`if文`の続きに追記してください。
        ''')

st.code('''
        if uploaded_file :
                
                (略)

                columns = df.columns
                column_list = columns.to_list()
                x_col = st.selectbox('x軸を選択してください。', column_list)
                y_col = st.selectbox('y軸を選択してください。', column_list)
        ''')

st.markdown('''
        上記コードの追記により、カラムをセレクトボックスで選択できるようになったと思います。\n
        セレクトボックス内の初期値を`None`に設定している理由としては、データを読み込むと同時にグラフを作成してしまい、余分な読み込みを防ぐために追加しています。\n
        選択したカラムは後でグラフを作成する際に使用するため、それぞれ変数に格納してあげましょう。\n
        それでは、選択したデータを可視化してみましょう。\n
        まずはグラフの種類を考えましょう。\n
        今回は、棒グラフ、散布図、箱ひげ図を選択できるようにしてみます。\n
        使用する可視化ライブラリは`Plotly`を使用してみましょう。\n
        以下のコードを追記してください。
        ''')

st.code('''
        plot_types = ['bar', 'scatter', 'vaioline']
        select_type = st.radio('グラフの種類を選択してください。', plot_types)
        ''')



uploaded_file = st.file_uploader('csvファイルをアップロードしてください。', type=['csv'])

if uploaded_file :
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        columns = df.columns
        column_list = columns.to_list()
        column_list = [None] + column_list
        x_col = st.selectbox('x軸を選択してください。', column_list)
        y_col = st.selectbox('y軸を選択してください。', column_list)

        plot_types = ['bar', 'scatter', 'vaioline']
        select_type = st.radio('グラフの種類を選択してください。', plot_types)