import streamlit as st
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(layout='center')

st.title('グラフ可視化アプリ')

st.markdown('''
            このページでは読み込んだデータを用途に合ったグラフを選択することで、データを可視化するためのアプリケーションをハンズオン形式で学んでいきます。\n
            新しく`GraphApp.py`というファイルを作成し、アプリを起動してください。\n
            まずはページのレイアウトを設定し、グラフのタイトルを決めましょう。\n
            以下のコードを記載してください。
''')
st.code('''
        import streamlit as st

        st.set_page_config(layout="center")
        
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
        plot_types = ['bar', 'scatter', 'vaiolin']
        select_type = st.radio('グラフの種類を選択してください。', plot_types)
        ''')

st.markdown('''
        リストで取得したグラフの種類がラジオボタンで生成されたと思います。\n
        このラジオボタンが選択されたときに、それぞれのグラフが出力されるような条件式を書いていきたいと思います。\n
        以下のコードを追記してください。
        ''')

st.code('''
        import plotly.express as px

        if uploaded_file :
                
                (略)

                if select_type == 'bar':
                        fig = px.bar(df, x_col, y_col)
                elif select_type == 'scatter':
                        fig = px.scatter(df, x_col, y_col)
                else:
                        fig = px.violin(df, x_col, y_col)
                st.plotly_chart(fig)
        ''')

st.markdown('''
        グラフが表示されたと思います。\n
        Plotlyの`scatter_3d`メソッドを使用することで散布図の3Dプロットも表示することができます。\n
        興味がある方は挑戦してみてください。\n
        ''')
st.info('''
        Hint💡 : z軸のデータを追加し、条件式を編集してみてください。\n
        答えは全体コードに記載しています。
        ''')

st.markdown('''
        お疲れ様でした。\n
        本ページではグラフを可視化するためのアプリケーションを作成しました。\n
        グラフの種類を追加したり、複数のデータを選択することでデータを比較したりなど他機能を追加してみてください。
        以下に全体のコードを記載しておきます。
        ''')

st.code('''
        import pandas as pd
        import plotly.express as px
        import streamlit as st

        st.set_page_config(layout="center")
        
        st.title("グラフ可視化アプリ")

        uploaded_file = st.file_uploader('csvファイルをアップロードしてください。', type=['csv'])

        if uploaded_file :
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())

                columns = df.columns
                column_list = columns.to_list()
                column_list = [None] + column_list
                x_col = st.selectbox('x軸を選択してください。', column_list)
                y_col = st.selectbox('y軸を選択してください。', column_list)
                z_col = st.selectbox('z軸を選択してください。', column_list)    # z軸の追加

                plot_types = ['bar', 'scatter', 'vaiolin']
                select_type = st.radio('グラフの種類を選択してください。', plot_types)

                if select_type == 'bar':
                        fig = px.bar(df, x_col, y_col)
                if select_type == 'scatter':                                            # 2Dプロットか3Dプロットの条件分け
                        if z_col :                                                      # ここではz軸が選択されたかされていないかで条件分岐
                                fig = px.scatter_3d(df, x_col, y_col, z_col)
                        else:
                                fig = px.scatter(df, x_col, y_col)
                if select_type == 'vaiolin':
                        fig = px.violin(df, x_col, y_col)
                st.plotly_chart(fig)

        ''')

