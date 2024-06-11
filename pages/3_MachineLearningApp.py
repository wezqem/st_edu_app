import streamlit as st

st.set_page_config(layout='centered')

st.title('機械学習アプリ')

st.markdown('''

            ''')

st.code('''

        ''')

st.markdown('''
            このページでは読み込んだデータを使用して機械学習を行うためのアプリケーションを構築するためのハンズオンを提供します。\n
            新しく`MachineLearningApp.py`というファイルを作成し、アプリを起動してください。\n
            まずはページのレイアウトを設定し、グラフのタイトルを決めましょう。\n
            以下のコードを記載してください。
''')
st.code('''
        import streamlit as st

        st.set_page_config(layout="centered")
        
        st.title("機械学習アプリ")
        ''')

st.markdown('''
        このアプリでは、**データ読み込み**と**機械学習**を行うためのページを作成し、それらを切り替えることでそれぞれの機能を使用できるような形式にします。\n
        以下のコードを追記してください。
            ''')

st.code('''
        pages = st.sidebar.radio('Pages', ['データ読み込み', '機械学習デモ'])
        ''')

st.markdown('''
        サイドバーにページ切り替えのラジオボタンを用意しました。\n
        サイドバーを作成する際には、`st`とコンポーネントの間に`sidebar`を入れることで簡単に実装することができます。\n
        このラジオボタンで選択されたとき、それぞれの機能を使用していくため条件式を書いていく必要があります。\n
        それでは、条件式を書き、さらにデータを読み込むためのウィジェットを用意しましょう。\n
        以下のコードを追記してください。
            ''')

# データ読み込み部分

st.code('''
        import pandas as pd

        if pages == 'データ読み込み':
            st.subheader('データ読み込み')
            st.write('このページではデータの読み込み、およびデータの概要を確認することを目的とします。')

            if 'data' not in st.session_state:
                st.session_state.data = None

            uploaded_file = st.file_uploader('csvファイルをアップロードしてください。', type = ['csv'])
        ''')

st.markdown('''
        `データ読み込み`ページを選択した際の機能として、まずはデータをアップロードするためのウィジェットを作成します。\n
        ここで、Streamlitの仕様を説明します。\n
        Streamlitでは、ボタンクリックなどのイベントが発生した場合、コード自体が1行目から再リロードされてしまいます。\n
        そのため、ページを切り替えた場合に読み込んだファイルが消去されることになります。\n
        これを防ぐために`session_state`というデータキャッシュの機能を用いることで、ページが切り替わった際にもデータを保持することができます。\n
        上記コードでは、セッションステートを初期化しているだけですが、データを読み込ませた後にセッションステートを使用してデータを保持します。これについては後述します。\n
        それでは、データを読み込んだ際の処理について以下のコードを追記してください。
            ''')

st.code('''
        if pages == 'データ読み込み':
        
            (略)    

            uploaded_file = st.file_uploader('csvファイルをアップロードしてください。', type = ['csv'])

            if uploaded_file :
                df = pd.read_csv(uploaded_file)
                st.write('データテーブル')
                st.dataframe(df)
                st.write('データの統計量')
                st.write(df.describe())

                col1, col2 = st.columns(2)

                with col1 :
                    st.write('欠損値の有無')
                    st.write(df.isnull().sum())

                with col2 :
                    st.write('データの型')
                    st.write(df.dtypes)
                
                st.session_state.data = df
            ''')

st.markdown('''
        `データ可視化アプリ`でも同様のコードを書きましたが、`if文`を用いてウィジェットにファイルがアップロードされた際の処理を書いていきます。\n
        ここでは、データフレームだけでなく、データの統計量も併せて表示させてみましょう。\n
        また、機械学習ではデータの欠損値や型が重要になってきますので、ここで一緒に表示させておきましょう。\n
        通常通り記載すると縦方向に並んで表示されることになりますが、カラム機能を用いることで横方向に並べて表示させることができます。\n
        Webアプリケーションを構築する際にはUIの部分にも注意して構築することでユーザビリティを高めることができますので覚えておきましょう。\n
        書き方としては、`st.columns()`の引数に生成したいカラムの数を渡します。\n
        その後、with構文を用いてそれぞれのカラムに表示させたいコードを記載します。\n
        そして最後にセッションステートを用いてデータをキャッシュしておきましょう。
            ''')

# 機械学習部分
st.divider()

st.markdown('''
        データ読み込み部分の機能の作成ができました。\n
        ここから機械学習の部分を実装していきます。\n
        今回は回帰モデルのみ実装します。分類モデルについては実装しませんが、コード内には記入していきますので、随時「未実装」と表示します。\n
        興味がある方は分類モデルの実装に挑戦してみてください。\n
        それでは、以下のコードを追記してください。
        ''')

st.code('''
        if pages == '機械学習デモ':

            st.subheader('機械学習デモ')
            st.write('このページでは、非データサイエンティスト向けの機械学習デモを行います。')
            st.write('機械学習モデルは、機械学習のためのライブラリを用意しています。')
            st.write('予測させたいターゲット、モデル、学習器を選択してください。')

            st.error('分類器モデルについては各モデル未実装です。')

            if st.session_state.data is not None:
                df = st.session_state.data

            else:
                st.warning('データが読み込まれていません。データ読み込みページで、CSVファイルをアップロードしてください。')

            target = st.selectbox('ターゲットを選択してください。', df.columns)

            X = df.drop(columns=[target])
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

            model_list = st.selectbox('モデルを選択してください。', ['Random Forest', 'XGBoost', 'LightGBM'])
        ''')

st.markdown('''
        まずは、セッションステートからキャッシュしたデータを呼び出します。\n
        呼び出したデータからカラム名を取得し、セレクトボックスを使用してターゲットを選択できるようにします。\n
        選択したターゲットをもとに特徴量を取得します。\n
        その後、機械学習用にデータを分割しておきます。\n
        使用するモデルも同様に選択できるようにセレクトボックスを作成します。\n
        今回はモデルとして`Rnadom Forest`、`XGBoost`、`LightGBM`を実装していきます。\n
        続けて、選択したモデルに対して学習器を選択できるようにし、それぞれの機能を実装していきます。\n
        今回は回帰器のみ実装していきます。\n
        以下のコードを追記して下さい。
        ''')

st.code('''
        if pages == '機械学習デモ' :
                
                (略)

                if model_list == 'Random Forest':

                task_rf = st.selectbox('学習器を選択してください。', ['回帰器', '分類器'])

                if task_rf == '分類器':
                    rf = RandomForestClassifier()
                    st.error('未実装です。')
                if task_rf == '回帰器':
                    rf = RandomForestRegressor()

                st.subheader('Random Forestのハイパーパラメータを決定してください。')
                n_estimators = st.slider('n_estimators', 100, 500, 100, 100)
                max_depth = st.slider('max_depth', 3, 9, 3, 2)
                min_samples_split = st.slider('min_samples_split', 2, 10, 2, 2)
                min_samples_leaf = st.slider('min_samples_leaf', 1, 5, 1)
        ''')

st.markdown('''
        まずは、モデルで`Random Forest`を選択した際の機能を実装していきます。\n
        学習器を選択できるようにセレクトボックスを作成し、選択した学習器の機能をそれぞれ条件分岐させます。\n
        今回は分類器の実装はしないので、未実装の表示をさせます。\n
        回帰器を選択した場合の機能として、ハイパーパラメータを設定できるようなコンポーネントを作成しましょう。\n
        使用頻度が高いハイパーパラメータをスライダーを用いてユーザー側で自由に設定できるようにします。\n
        随時データに応じて必要なハイパーパラメータを追加してください。\n
        以下のコードを追記してください。
        ''')

st.code('''
        import matplotlib.pyplot as plt 
        import japanize_matplotlib

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import xgboost as xgb
        import lightgbm as lgb
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
        from math import sqrt

        if pages == '機械学習デモ' :
                
                (略)

                if model_list == 'Random Forest':
        
                    (略)
        
                    if st.button('学習'):

                        # モデル学習
                        rf.set_params(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf)
                        rf.fit(X_train, y_train)

                        # 予測値の取得
                        y_pred = rf.predict(X_test)

                        # 学習が終了したかの確認
                        st.success('学習が完了しました。')

                        # 評価指標の計算と表示
                        if task_rf == '分類器':
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            st.write('評価指標（分類）:')
                            st.write('Accuracy:', accuracy)
                            st.write('Precision:', precision)
                            st.write('Recall:', recall)
                            st.write('F1-score:', f1)

                        if task_rf == '回帰器':
                            r2 = r2_score(y_test, y_pred)
                            rmse = sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            st.write('評価指標（回帰）:')
                            st.write('R2:', r2)
                            st.write('RMSE:', rmse)
                            st.write('MAE:', mae)
                        
                        # 特徴量の重要度可視化
                        importances = rf.feature_importances_
                        feature_names = X_train.columns
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots()
                        ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Importance')
                        ax.set_ylabel('Feature')
                        ax.set_title('Feature Importances')
                        
                        st.pyplot(fig)
        ''')

st.markdown('''
        ハイパーパラメータを決定したら、`学習`というボタンを押下することで学習を開始させます。\n
        ボタン押下時の処理も条件分岐として実装することができます。\n
        モデルを学習させ、予測値を算出させてみましょう。\n
        算出された予測値を用いて、モデルの評価を行います。\n
        今回は各学習器に沿った評価指標を用います。\n
        また、使用した特徴量がターゲットに対してどの程度起因しているかの指標として**特徴量の重要度**を算出することがよくあります。\n
        最後にこの重要度を可視化するコードを書いておきましょう。\n
        同様の機能を他のモデルでも実装していきます。以下のコードを追記してください。
        ''')

st.code('''
        if pages == '機械学習デモ' :
                
                (略)

                if model_list == 'Random Forest':
        
                    (略)
        
                if model_list == 'XGBoost':

                    task_xgb = st.selectbox('学習器を選択してください', ['回帰器', '分類器'])

                    if task_xgb == '分類器':
                        st.error('未実装です。')

                    st.subheader('XGBoostのハイパーパラメータを決定してください。')

                    # ハイパーパラメータの設定
                    n_estimators = st.slider('n_estimators', 10, 1000, 100, 10)
                    max_depth = st.slider('max_depth', 1, 10, 3, 1)
                    learning_rate = st.slider('learning_rate', 0.01, 1.0, 0.1, 0.01)
                    subsample = st.slider('subsample', 0.1, 1.0, 1.0, 0.1)
                    colsample_bytree = st.slider('colsample_bytree', 0.1, 1.0, 1.0, 0.1)

                    if st.button('学習'):
                    # モデル学習
                        if task_xgb == '分類器':
                            xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                                    subsample=subsample, colsample_bytree=colsample_bytree, random_state=80)
                        else:
                            xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                                    subsample=subsample, colsample_bytree=colsample_bytree, random_state=80)
                        
                        xgb_model.fit(X_train, y_train)
                        
                        # 予測値の取得
                        y_pred = xgb_model.predict(X_test)
                        
                        # 学習が終了したかの確認
                        st.success('学習が完了しました。')
                        
                        # 評価指標の計算と表示
                        if task_xgb == '分類器':
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            st.write('評価指標（分類）:')
                            st.write('Accuracy:', accuracy)
                            st.write('Precision:', precision)
                            st.write('Recall:', recall)
                            st.write('F1-score:', f1)
                            
                        if task_xgb == '回帰器':
                            r2 = r2_score(y_test, y_pred)
                            rmse = sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            st.write('評価指標（回帰）:')
                            st.write('R2:', r2)
                            st.write('RMSE:', rmse)
                            st.write('MAE:', mae)
                        
                        # 特徴量の重要度の可視化
                        importances = xgb_model.feature_importances_
                        feature_names = X_train.columns
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots()
                        ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Importance')
                        ax.set_ylabel('Feature')
                        ax.set_title('Feature Importances')
                        
                        st.pyplot(fig)

                if model_list == 'LightGBM':


                    task_lgb = st.selectbox('学習器を選択してください。', ['回帰器', '分類器'])

                    if task_lgb == '分類器':
                        st.error('未実装です。')

                    st.subheader('LightGBMのハイパーパラメータを決定してください。')

                    # ハイパーパラメータの設定
                    num_leaves = st.slider('num_leaves', 2, 100, 31, 1)
                    max_depth = st.slider('max_depth', -1, 100, -1, 1)
                    learning_rate = st.slider('learning_rate', 0.01, 1.0, 0.1, 0.01)
                    n_estimators = st.slider('n_estimators', 10, 1000, 100, 10)
                    subsample = st.slider('subsample', 0.1, 1.0, 1.0, 0.1)
                    colsample_bytree = st.slider('colsample_bytree', 0.1, 1.0, 1.0, 0.1)
                
                    if st.button('学習'):
                        # モデル学習
                        if task_lgb == '回帰器':
                            lgb_model = lgb.LGBMRegressor(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate,
                                                    n_estimators=n_estimators, subsample=subsample, colsample_bytree=colsample_bytree,
                                                    random_state=80)
                        else:
                            lgb_model = lgb.LGBMClassifier(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate,
                                                    n_estimators=n_estimators, subsample=subsample, colsample_bytree=colsample_bytree,
                                                    random_state=80)
                        
                        lgb_model.fit(X_train, y_train)
                        
                        # 予測値の取得
                        y_pred = lgb_model.predict(X_test)
                        
                        # 学習が終了したかの確認
                        st.success('学習が完了しました。')
                        
                        # 評価指標の計算と表示
                        if task_lgb == '回帰器':
                            r2 = r2_score(y_test, y_pred)
                            rmse = sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            st.write('評価指標（回帰）:')
                            st.write('R2:', r2)
                            st.write('RMSE:', rmse)
                            st.write('MAE:', mae)
                        else:
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            st.write('評価指標（分類）:')
                            st.write('Accuracy:', accuracy)
                            st.write('Precision:', precision)
                            st.write('Recall:', recall)
                            st.write('F1-score:', f1)
                        
                        # 特徴量の重要度の可視化
                        importances = lgb_model.feature_importances_
                        feature_names = X_train.columns
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots()
                        ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Importance')
                        ax.set_ylabel('Feature')
                        ax.set_title('Feature Importances')
                        
                        st.pyplot(fig)
        ''')

st.markdown('''
        お疲れ様でした。\n
        本ページでは機械学習アプリを作成しました。\n
        コードはリファクタリングできる部分が多く残っているかと思いますので、良ければ関数化などしてみてください。\n
        また、分類器の機能実装にも挑戦してみてください。\n
        以下に全体のコードを記載しておきます。
        ''')

with st.expander('全体コード'):
    st.code('''
            ### ---- 機械学習デモアプリ ---- ###

            ### ---- ライブラリ ---- ###
            # データ分析用
            import streamlit as st 
            import pandas as pd 
            import matplotlib.pyplot as plt 
            import japanize_matplotlib

            # モデル学習用
            from sklearn.model_selection import train_test_split

            # ランダムフォレスト
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # XGBosot
            import xgboost as xgb

            # LightGBM
            import lightgbm as lgb

            # 評価指標
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
            from math import sqrt


            ### ---- マルチページ作成 ---- ###
            # サイドバーページを作成
            pages = st.sidebar.radio('Pages', ['データ読み込み', '機械学習デモ'])

            ### ---- データ読み込み ---- ###
            if pages == 'データ読み込み':

                # ページタイトルと説明
                st.subheader('データ読み込み')
                st.write('このページではデータの読み込み、およびデータの概要を確認することを目的とします。')

                # セッションステートの初期化
                if 'data' not in st.session_state:
                    st.session_state.data = None

                uploaded_file = st.file_uploader('csvファイルをアップロードしてください。', type = ['csv'])

                if uploaded_file :
                    df = pd.read_csv(uploaded_file)
                    st.write('データテーブル')
                    st.dataframe(df)
                    st.write('データの統計量')
                    st.write(df.describe())

                    col1, col2 = st.columns(2)

                    with col1 :
                        st.write('欠損値の有無')
                        st.write(df.isnull().sum())

                    with col2 :
                        st.write('データの型')
                        st.write(df.dtypes)
                    
                    st.session_state.data = df

            if pages == '機械学習デモ':

                # ページタイトルと説明
                st.subheader('機械学習デモ')
                st.write('このページでは、非データサイエンティスト向けの機械学習デモを行います。')
                st.write('機械学習モデルは、機械学習のためのライブラリを用意しています。')
                st.write('予測させたいターゲット、モデル、学習器を選択してください。')

                st.error('分類器モデルについては各モデル未実装です。')

                # セッションステートからデータを取得
                if st.session_state.data is not None:
                    df = st.session_state.data

                else:
                    st.warning('データが読み込まれていません。データ読み込みページで、CSVファイルをアップロードしてください。')

                # ターゲットの選択
                target = st.selectbox('ターゲットを選択してください。', df.columns)

                # ターゲットと特徴量を指定
                X = df.drop(columns=[target])
                y = df[target]

                # データの分割
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

                # モデル選択
                model_list = st.selectbox('モデルを選択してください。', ['Random Forest', 'XGBoost', 'LightGBM'])

                # 各モデルで条件分岐
                # ランダムフォレスト
                if model_list == 'Random Forest':

                    # 分類器か回帰器の選択
                    task_rf = st.selectbox('学習器を選択してください。', ['回帰器', '分類器'])

                    if task_rf == '分類器':
                        rf = RandomForestClassifier()
                        st.error('未実装です。')
                    if task_rf == '回帰器':
                        rf = RandomForestRegressor()

                    # パラメータをGUIで指定
                    st.subheader('Random Forestのハイパーパラメータを決定してください。')
                    n_estimators = st.slider('n_estimators', 100, 500, 100, 100)
                    max_depth = st.slider('max_depth', 3, 9, 3, 2)
                    min_samples_split = st.slider('min_samples_split', 2, 10, 2, 2)
                    min_samples_leaf = st.slider('min_samples_leaf', 1, 5, 1)

                    if st.button('学習'):

                        # モデル学習
                        rf.set_params(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf)
                        rf.fit(X_train, y_train)

                        # 予測値の取得
                        y_pred = rf.predict(X_test)

                        # 学習が終了したかの確認
                        st.success('学習が完了しました。')

                        # 評価指標の計算と表示
                        if task_rf == '分類器':
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            st.write('評価指標（分類）:')
                            st.write('Accuracy:', accuracy)
                            st.write('Precision:', precision)
                            st.write('Recall:', recall)
                            st.write('F1-score:', f1)

                        if task_rf == '回帰器':
                            r2 = r2_score(y_test, y_pred)
                            rmse = sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            st.write('評価指標（回帰）:')
                            st.write('R2:', r2)
                            st.write('RMSE:', rmse)
                            st.write('MAE:', mae)
                        
                        # 特徴量の重要度可視化
                        importances = rf.feature_importances_
                        feature_names = X_train.columns
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots()
                        ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Importance')
                        ax.set_ylabel('Feature')
                        ax.set_title('Feature Importances')
                        
                        st.pyplot(fig)
                
                if model_list == 'XGBoost':

                    task_xgb = st.selectbox('学習器を選択してください', ['回帰器', '分類器'])

                    if task_xgb == '分類器':
                        st.error('未実装です。')

                    st.subheader('XGBoostのハイパーパラメータを決定してください。')

                    # ハイパーパラメータの設定
                    n_estimators = st.slider('n_estimators', 10, 1000, 100, 10)
                    max_depth = st.slider('max_depth', 1, 10, 3, 1)
                    learning_rate = st.slider('learning_rate', 0.01, 1.0, 0.1, 0.01)
                    subsample = st.slider('subsample', 0.1, 1.0, 1.0, 0.1)
                    colsample_bytree = st.slider('colsample_bytree', 0.1, 1.0, 1.0, 0.1)

                    if st.button('学習'):
                    # モデル学習
                        if task_xgb == '分類器':
                            xgb_model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                                    subsample=subsample, colsample_bytree=colsample_bytree, random_state=80)
                        else:
                            xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                                    subsample=subsample, colsample_bytree=colsample_bytree, random_state=80)
                        
                        xgb_model.fit(X_train, y_train)
                        
                        # 予測値の取得
                        y_pred = xgb_model.predict(X_test)
                        
                        # 学習が終了したかの確認
                        st.success('学習が完了しました。')
                        
                        # 評価指標の計算と表示
                        if task_xgb == '分類器':
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            st.write('評価指標（分類）:')
                            st.write('Accuracy:', accuracy)
                            st.write('Precision:', precision)
                            st.write('Recall:', recall)
                            st.write('F1-score:', f1)
                            
                        if task_xgb == '回帰器':
                            r2 = r2_score(y_test, y_pred)
                            rmse = sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            st.write('評価指標（回帰）:')
                            st.write('R2:', r2)
                            st.write('RMSE:', rmse)
                            st.write('MAE:', mae)
                        
                        # 特徴量の重要度の可視化
                        importances = xgb_model.feature_importances_
                        feature_names = X_train.columns
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots()
                        ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Importance')
                        ax.set_ylabel('Feature')
                        ax.set_title('Feature Importances')
                        
                        st.pyplot(fig)

                if model_list == 'LightGBM':


                    task_lgb = st.selectbox('学習器を選択してください。', ['回帰器', '分類器'])

                    if task_lgb == '分類器':
                        st.error('未実装です。')

                    st.subheader('LightGBMのハイパーパラメータを決定してください。')

                    # ハイパーパラメータの設定
                    num_leaves = st.slider('num_leaves', 2, 100, 31, 1)
                    max_depth = st.slider('max_depth', -1, 100, -1, 1)
                    learning_rate = st.slider('learning_rate', 0.01, 1.0, 0.1, 0.01)
                    n_estimators = st.slider('n_estimators', 10, 1000, 100, 10)
                    subsample = st.slider('subsample', 0.1, 1.0, 1.0, 0.1)
                    colsample_bytree = st.slider('colsample_bytree', 0.1, 1.0, 1.0, 0.1)
                
                    if st.button('学習'):
                        # モデル学習
                        if task_lgb == '回帰器':
                            lgb_model = lgb.LGBMRegressor(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate,
                                                    n_estimators=n_estimators, subsample=subsample, colsample_bytree=colsample_bytree,
                                                    random_state=80)
                        else:
                            lgb_model = lgb.LGBMClassifier(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate,
                                                    n_estimators=n_estimators, subsample=subsample, colsample_bytree=colsample_bytree,
                                                    random_state=80)
                        
                        lgb_model.fit(X_train, y_train)
                        
                        # 予測値の取得
                        y_pred = lgb_model.predict(X_test)
                        
                        # 学習が終了したかの確認
                        st.success('学習が完了しました。')
                        
                        # 評価指標の計算と表示
                        if task_lgb == '回帰器':
                            r2 = r2_score(y_test, y_pred)
                            rmse = sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            st.write('評価指標（回帰）:')
                            st.write('R2:', r2)
                            st.write('RMSE:', rmse)
                            st.write('MAE:', mae)
                        else:
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted')
                            recall = recall_score(y_test, y_pred, average='weighted')
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            st.write('評価指標（分類）:')
                            st.write('Accuracy:', accuracy)
                            st.write('Precision:', precision)
                            st.write('Recall:', recall)
                            st.write('F1-score:', f1)
                        
                        # 特徴量の重要度の可視化
                        importances = lgb_model.feature_importances_
                        feature_names = X_train.columns
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots()
                        ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Importance')
                        ax.set_ylabel('Feature')
                        ax.set_title('Feature Importances')
                        
                        st.pyplot(fig)
            ''')
