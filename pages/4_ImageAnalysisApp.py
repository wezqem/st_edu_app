import streamlit as st

st.set_page_config(layout='centered')

st.title('画像解析アプリ')

st.markdown('''
        このページでは、画像解析の一種である物体検出を`Pytorch`で実装していきます。\n
        新しく`ImageAnalysisApp.py`というファイルを作成し、アプリを起動してください。\n
        まずは、ページタイトルやレイアウト設定を行います。\n
        また、今回はPytorchの`torchvision`ライブラリから事前学習済みの`Faster-R-CNN`というモデルをロードしましょう。\n
        以下のコードを追加してください。
            ''')

st.code('''
        import streamlit as st
        import torch
        from torchvision import models, transforms
        import numpy as np
        from PIL import Image, ImageDraw
        import pandas as pd

        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        ''')

st.markdown('''
        モデルをロードした際の`pretrained=True`というのは事前学習済みの重みを使用することを意味します。\n
        また、Pytorchでは、データの型を`Tensor型`に変換する必要があるので、型を変換するためのコードも併せて記載しておきましょう。\n
        次に画像ファイルをアップロードするためのウィジェットを作成し、読み込んだ画像を表示してみましょう。\n
        以下のコードを追記してください。
            ''')

st.code('''
        uploaded_file = st.file_uploader("画像ファイルを読み込んでください。", type=["jpg", "jpeg", "png"])

        if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Uploaded Image.', use_column_width=True)
        ''')

st.markdown('''
        画像ファイルは複数種類あるので、今回は`jpg`、`jpeg`、`png`を読み込めるようにtype引数にリストで渡してあげます。\n
        画像ファイルが読み込まれたら条件分岐で処理内容を記載していきます。\n
        今回は`OpenCV`ではなく、標準モジュールである`Pillow`を使用していきます。\n
        `Pillow`を使用した理由は、今回は画像自体の前処理は簡易的なためです。\n
        複雑な前処理が必要な場合は、`OpenCV`を使用してください。\n
        上記コードでは、読み込んで、それを表示させています。\n
        それでは、続きの処理を行うためのボタンを作成します。\n
        ここでボタンを用意する理由としては、**画像読み込み → 前処理・予測**という処理を分けることで、処理時間を分割できるためです。\n
        それでは、ボタンを作成して、検出結果の処理を行いましょう。以下のコードを追記してください。
            ''')

st.code('''
        if uploaded_file:
                
                (略)

                if st.button('Detect Objects'):
                        # 画像の前処理
                        image_np = np.array(image)
                        image_tensor = transform(image_np)
                        image_tensor = image_tensor.unsqueeze(0)  

                        with torch.no_grad():
                                predictions = model(image_tensor)

                        prediction = predictions[0]
                        boxes = prediction['boxes'].numpy()
                        scores = prediction['scores'].numpy()
                        labels = prediction['labels'].numpy()
        ''')

st.markdown('''
        まずは必要な画像の前処理を行います。\n
        今回は、画像を配列に変換し、Tensor型に変換します。その後、Pytorchのモデルはバッチ次元を必要とするため、バッチ次元を追加しています。\n
        **バッチ次元の追加について**：Tensorの形状`(H, W, C)`の場合、1番目に次元を追加しているので、`(1, H, W, C)`という形状に変換されます。\n
        このあたりのPytorchに関する内容は興味があれば調べてみてください。\n
        その後、オブジェクト検出を行い、検出結果を処理しています。\n
        それでは、以下のコードを追記してください。
            ''')

st.code('''
        if uploaded_file:
                
                (略)

                if st.button('Detect Objects'):
        
                      (略)
  
                        results = []
                        for box, score, label in zip(boxes, scores, labels):
                                if score > 0.5:  
                                        results.append({
                                        "Label": label,
                                        "Score": score,
                                        "Box": box
                                        })
        ''')

st.markdown('''
        上記コードでは、信頼度が50%以上の検出結果のみを結果として取得しています。
        結果を取得したので、表示させていきましょう。\n
        今回は、結果をデータフレームに格納させて表示させてみましょう。\n
        併せて、読み込んだ画像ファイルに対して、バウンディングボックス、スコア、ラベルを描画して、画像のどの部分を物体検出したかを確認できるようにします。\n
        また、信頼度が50%以上の検出結果がなかった場合の表示もしておきましょう。\n
        以下のコードを追記してください。
            ''')

st.code('''
        if uploaded_file:
                
                (略)

                if st.button('Detect Objects'):
        
                        (略)
        
                        if results:
                        df = pd.DataFrame(results)
                        st.write("Detection Results:")
                        st.dataframe(df)
                        
                        draw = ImageDraw.Draw(image)
                        for result in results:
                                box = result["Box"]
                                label = result["Label"]
                                score = result["Score"]
                                draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=3)
                                draw.text((box[0], box[1]), f"Label: {label} Score: {score:.2f}", fill="red")
                        
                        st.image(image, caption='Detected Image.', use_column_width=True)
                        else:
                        st.write("No objects detected with confidence above 50%")
        ''')

st.markdown('''
        お疲れ様でした。\n
        本ページではPytorchのモデルを用いた物体検出アプリを作成しました。\n
        今回は単純なコンポーネントの追加のみでしたが、画像の前処理が複雑になった場合でもアプリケーション内で処理を行うことが可能です。\n
        以下に全体のコードを記載しておきます。
            ''')

with st.expander('全体コード'):
        st.code('''
                import streamlit as st
                import torch
                from torchvision import models, transforms
                import numpy as np
                import cv2
                from PIL import Image, ImageDraw
                import pandas as pd

                # モデルのロード
                model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
                model.eval()

                # 画像変換用のトランスフォーム
                transform = transforms.Compose([
                    transforms.ToTensor()
                ])

                # ファイルアップロード
                uploaded_file = st.file_uploader("画像ファイルを読み込んでください。", type=["jpg", "jpeg", "png"])

                if uploaded_file is not None:
                # 画像の読み込み
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Uploaded Image.', use_column_width=True)
                
                # ボタンの配置
                if st.button('Detect Objects'):
                        # 画像の前処理
                        image_np = np.array(image)
                        image_tensor = transform(image_np)
                        image_tensor = image_tensor.unsqueeze(0)  # バッチサイズを追加

                        # オブジェクト検出
                        with torch.no_grad():
                        predictions = model(image_tensor)

                        # 検出結果の処理
                        prediction = predictions[0]
                        boxes = prediction['boxes'].numpy()
                        scores = prediction['scores'].numpy()
                        labels = prediction['labels'].numpy()

                        # 結果をデータフレームに格納
                        results = []
                        for box, score, label in zip(boxes, scores, labels):
                        if score > 0.5:  # 信頼度が50%以上の検出結果のみ表示
                                results.append({
                                "Label": label,
                                "Score": score,
                                "Box": box
                                })
                        
                        # 結果の表示
                        if results:
                        df = pd.DataFrame(results)
                        st.write("Detection Results:")
                        st.dataframe(df)
                        
                        # 検出結果の描画
                        draw = ImageDraw.Draw(image)
                        for result in results:
                                box = result["Box"]
                                label = result["Label"]
                                score = result["Score"]
                                draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline="red", width=3)
                                draw.text((box[0], box[1]), f"Label: {label} Score: {score:.2f}", fill="red")
                        
                        st.image(image, caption='Detected Image.', use_column_width=True)
                        else:
                        st.write("No objects detected with confidence above 50%")

        ''')

