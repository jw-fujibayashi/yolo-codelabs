id: yolo
satus: making
summary: yolov3

# tensorflow2.0 YOLOv3

## 概要
Duration: 10

物体検出のアルゴリズムの1つ、YOLOv3をtensorflow2.0で試します。

自分の好きなものをAIに学習させて検出できるようになりましょう！

以下の画像は、「グー」「チョキ」「パー」の手の形を学習させて検出させている様子です。画像などを用意するのは単純作業で辛いですが、上手く検出できるとその反面嬉しさも増えるので頑張りましょう。

<!-- ![](https://i.imgur.com/m7mGVq4.jpg) -->
![img1](img/img1.jpg)

### 今回使用するものについて

#### Google Colaboratory

<!-- ![](https://i.imgur.com/d3qlWm1.png) -->
![img2](img/img2.png)   
Googleが提供する「機械学習/Deep Learningの教育や研究の促進を目的としたプロジェクト」です。ブラウザから操作できるので、環境構築不要です。また、GPUというハイスペックな環境を使用できます。

Google Colaboratoryで使用するファイルを「ノートブック」と呼び、今回はこちらが用意したノートブックを使用します。

<!-- ![](https://i.imgur.com/xCTfskl.png) -->
![img3](img/img3.png)   

このように、ノートブックにはプログラミング言語だけではなく、ドキュメントも合わせて記すことができます。

<!-- ![](https://i.imgur.com/4IYIf27.png) -->
![img5](img/img5.png)   

こちらにプログラミングコードが記述されます。セル毎に分かれていますが、環境や変数などは同一ノートブック内では共通されます。
<!-- ![](https://i.imgur.com/OGij9Ld.png) -->

コードを実行するには、実行したいセルにカーソルを合わせた状態で、`ctrl`+`Enter`(Macの場合は、`cmd`+`Enter`)です。もしくは、再生ボタンをクリックします。

Google Colaboratoryについて気になる方は公式の[説明](https://colab.research.google.com/notebooks/intro.ipynb?hl=ja)もご参照ください。

#### ラズベリーパイ

<!-- ![](https://i.imgur.com/EWQhbEh.jpg) -->
![img6](img/img6.jpg)

「ラズベリーパイ」は、必要最低限な機能を備えた小型PCです。その分スペックは低いですが数千円で購入できる手軽さから、子ども用のプログラミング入門キットとしてだけでなく、大学の工学部や情報学部などでの授業に取り入れられてます。また大人も楽しめる趣味として世界中で幅広い層に利用されています。

今回はこのラズパイにカメラをつけて、みなさんの手元でリアルタイム物体検出を行います。

#### VoTT

<!-- ![](https://i.imgur.com/dOZEdjv.jpg) -->
![img7](img/img7.jpg)

マイクロソフトが開発している[VoTT](https://github.com/microsoft/VoTT)を使って、アノテーションと呼ばれる作業を行います。詳しい説明は、後述しますが作業を補助する機能がついていて便利です。

### 大まかな物体検出の流れ

1. 学習データの作成
    - 検出したいものの画像の用意
    - 画像に対してアノテーション作業
2. 学習
3. 推論（検出）

今回はオリジナルデータで物体検出をするために、まずは学習データを皆さんに作成してもらいます。

その後、上述したGoogle Colaboratoryで学習を行い、最後にラズパイでリアルタイム画像検出を行います。

## 事前準備
Duration: 10

まずは、YOLOv3の講義から入りたいところですが、いくつか時間のかかる作業があるのでそちらを先に済ませます。

### 1.ノートブックのコピー

まずは、今回用意したノートブックをみなさんのGoogle Driveにコピーします。

[こちら](https://colab.research.google.com/drive/1FzwCZIIsqvSDktRDWVclO_jokT55qoA0?usp=sharing)をクリックして開いてください。

<!-- ![](https://i.imgur.com/b1L4A3k.png) -->
![img8](img/img8.png)

開いたら「ファイル」 > 「ドライブにコピーを保存」をクリックします。これでみなさんのドライブにコピーが作成されます。

<!-- ![](https://i.imgur.com/bygQXEG.png) -->
![img9](img/img9.png)

ただし、このコピーししたノートブックは実行できる状態ではないので「ファイル」>「Playgroundモードで開く」をクリックして実行できるように変更する。

コピーできたら、1つ目のセル「1.Google Colaboratory テスト用コード」を試しに実行してみましょう！実行するには、`ctrl`+`Enter`(Macの場合は、`cmd`+`Enter`)です。もしくは、再生ボタンをクリックします。

<!-- ![](https://i.imgur.com/oGL2h4c.png) -->
![img10](img/img10.png)  

このようになれば問題ありません。

### 2.Google Drive との連携

無事にノートブックをコピーできて、実行できるようになったら作業を進めていきます。まずは、Google DriveとGoogle Colaboratoryを連携させます。

「2.Google Driveとの連携」を実行しましょう。

<!-- ![](https://i.imgur.com/51i8BWm.png) -->
![img11](img/img11.png)

実行すると、このようにリンクが現れるのでクリックして開きます。

<!-- ![](https://i.imgur.com/Flp7a2r.png) -->
![img12](img/img12.png)

クリックすると、連携先のGoogleアカウントを選びます。複数のアカウントを持っている方は、ノートブックをコピーしたアカウントを選んでください。

<!-- ![](https://i.imgur.com/CBWAWm0.png) -->
![img13](img/img13.png)

「許可」を押します。

<!-- ![](https://i.imgur.com/MK8qC3d.png) -->
![img14](img/img14.png)

赤色で隠されている部分をコピーします。赤枠箇所をクリックすると自動でコピーされます。

<!-- ![](https://i.imgur.com/uJkHczM.png) -->
![img15](img/img15.png)

コピーできたら、ノートブックに戻ってこの箇所に貼り付けて、`Enter`を押します。

<!-- ![](https://i.imgur.com/zqQBbKR.png) -->
![img16](img/img16.png)

少し時間が経って、このように表示されると連携の終了です。

<!-- ![](https://i.imgur.com/LDzl5ik.png) -->
![img17](img/img17.png)

左側のフォルダマークをクリックすると、`drive`フォルダーが表示され、ここでdrive内のファイルを確認できます。

### 3. Gitプロジェクトのダウンロード

今回使用するソースコードをダウンロードします。「3.使用するgitプロジェクトをダウンロード」を実行しましょう。

実行すると、Google Driveの`Colabo NoteBooks`というフォルダーの中に`yolo-tf2-test`というフォルダーが自動でダウンロードされます。

<!-- ![](https://i.imgur.com/g4MUDdm.png) -->
![img18](img/img18.png)

このようにDrive内の`Colab Notebooks`フォルダー内にダウンロードされます。

### 4. 重みのダウンロードと変換

後ほど使う、「重み」というものをダウンロードし変換します。ダウンロードには時間がかなりかかります。
「4.重みのダウンロードと変換」を実行しましょう。

ダウンロードしている間に、次の「簡単なYOLOv3の仕組みについて講義」を確認しましょう。

## 簡単なYOLOv3の仕組みについて講義

それでは、今回試す「YOLO」について簡単に解説したいと思います。

難しい内容ですが、理解できなくてもプログラムを実行して物体検出を体験できますので安心してください。そういうものなのか、という程度で読み流してもらえればと思います。

YOLOとは、物体検出の手法の1つです。「v3」というのは、そのままバージョン3のことを表します（※2020年5月にv4が考案されたようです。）。物体領域検出と画像認識を同時に行うことでより速く物体検出を行えるのが特徴です（その反面精度は少し落ちています。）。

### YOLOの手法について

まずは、YOLOがどのように物体検出を行っているのか簡単なイメージで解説します。v3での改善点は最後に解説します。

<!-- ![](https://i.imgur.com/9qsnhf0.png) -->
![img19](img/img19.png)  

https://pjreddie.com/ から引用

YOLOでは、まず対象の画像をいくつかのセルに分けます。この各セルに対して、「物体領域検出（どこに物体があるのか）」と「画像認識（そのセルが何なのか）」を行います。

<!-- ![](https://i.imgur.com/KtT7V0n.png) -->
![img20](img/img20.png)  

https://pjreddie.com/ から引用

上の画像が「物体領域検出」の様子です。このように、サイズの異なる複数のバウンディングボックスを用意し、それぞれに対して「物体がどれぐらい存在しているのか」を計算します。これを「信頼度スコア」と言います。
物体が存在していなければ、信頼度スコアは0になります。

<!-- ![](https://i.imgur.com/37GEo38.png) -->
![img21](img/img21.png)  

https://pjreddie.com/ から引用

上の画像が「画像認識」の様子です。各セルに対して画像認識を行い、そのセルがどういった画像なのかを計算します。

<!-- ![](https://i.imgur.com/69exYVz.png) -->
![img22](img/img22.png)

https://pjreddie.com/ から引用

上記の「物体領域検出」と「画像認識」の結果を合わせることで、最終的に「物体検出」を行います。

<!-- ![](https://i.imgur.com/Bq5716P.png) -->
![img23](img/img23.png)  

https://qiita.com/mdo4nt6n/items/68dcda71e90321574a2b から引用

一連の流れを画像で表すとこのようになります。

### YOLOv3 での改善点

上記に加えて、v3ではいくつかの工夫を追加して精度や速度の改善を行いました。

1. 1/32、1/16、1/8のサイズに分けて物体検出の結果を出力します。複数のサイズに分けることで、さまざまな画像の大きさに対応可能
2. 物体領域検出の際に使用するバウンディングボックスのサイズを調整

以上で、YOLOについての簡単な講義は終わりです。詳細が気になる方は、Qiitaや個人のブログなどで取り上げられたりまとめられているのでそちらをご参照ください。

## 今回のポイント
Duration: 10

ラズパイで物体検出を簡単に試せるよう、以下の3つの工夫をしました。

- 転移学習
- YOLOv3-tinyの採用
- 画像の水増し

### 転移学習

物体検出は、AIに学習させる時間や必要なデータ数が大量になります。そのため、今回は「転移学習」と呼ばれるものを行い「時間」「データ数」の必要量を減らします。

簡単に言うと、ある領域で学習させたモデルを別の領域に適応させる技術です。事前に学習されているため、学習に時間がかかりません。たとえば、「犬の種類の画像認識を行うモデル」を「猫の種類の画像認識」に応用できます。

「4.重みのダウンロードと変換」で今回使用する「学習済みの重み」をダウンロードしました。

### YOLOv3-tinyの採用

また、YOLOv3-tinyと呼ばれる「正確さよりもリアルタイム性や軽量さを重視したモデル」を使用します。そのため、重みもtiny版をダウンロードしています。

もし通常のYOLOの重みも試してみたい方は、

```python=
!wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolo_weight/yolov3-tiny.weights
!python convert.py --weights ./data/yolo_weight/yolov3-tiny.weights --output ./data/weight/yolov3-tiny.tf --tiny

# tinyを使わない場合は以下
# !wget https://pjreddie.com/media/files/yolov3.weights -O data/yolo_weight/yolov3.weights
# !python convert.py --weights ./data/yolo_weight/yolov3.weights --output ./data/weight/yolov3.tf
```
1,2行目をコメントにして、5,6行目をコメントアウトしてください。

```python=
# !wget https://pjreddie.com/media/files/yolov3-tiny.weights -O data/yolo_weight/yolov3-tiny.weights
# !python convert.py --weights ./data/yolo_weight/yolov3-tiny.weights --output ./data/weight/yolov3-tiny.tf --tiny

# tinyを使わない場合は以下
!wget https://pjreddie.com/media/files/yolov3.weights -O data/yolo_weight/yolov3.weights
!python convert.py --weights ./data/yolo_weight/yolov3.weights --output ./data/weight/yolov3.tf
```

つまり、これを実行してもらえれば大丈夫です。

### 学習データの水増し

画像認識でもよく採用されている「学習データの水増し」を使って、必要な画像枚数を少なくします。また、この後に作成するアノテーションデータ自体の水増しも行うため、時間短縮できます。

「水増し」という言葉の印象はあまりよくありませんが、単純に学習データとなる画像をプログラムを使って複製するだけです。つまり、元の画像枚数が20枚しか

自分が試したところ、1つのクラスに対して20種類ぐらいの画像がを用いると水増しが上手く効果してくれました。

## テスト
Duration: 10

次に、必要なデータが無事にダウンロードできているかをテストします。

ノートブックに戻って、ダウンロードが完了していたら「5.テスト」を実行しましょう。

<!-- ![](https://i.imgur.com/1y96u5X.jpg) -->
![img24](img/img24.jpg)

このように画像が出てきたらテストには問題ありません。検出が上手くいっていなくても大丈夫です。おそらくtinyのため精度が低いのでしょう。

<!-- ![](https://i.imgur.com/FYsQcjM.jpg) -->
![img25](img/img25.jpg)

ちなみに、tinyではなく通常の重みを使用すると高精度に検出してくれます。

```python
# tiny
# !python detect.py --weights ./data/weight/yolov3-tiny.tf --tiny --image ./data/meme.jpg

# not tiny
!python detect.py --weights ./data/weight/yolov3.tf --image ./data/meme.jpg

from IPython.display import Image,display_jpeg
display_jpeg(Image('output.jpg'))
```
先ほどtinyではない重みもダウンロードした方はこちらを実行すればテストできます。

また、他の画像を使用することもできます。

<!-- ![](https://i.imgur.com/Xbgknwa.png) -->
![img26](img/img26.png)

左のフォルダマークをクリックして、`drive` > `My Drive` > `Colab Notebooks` > `yolo-tf2-test` > `data`の順にクリックしていきます。

そして、検出させたい画像を最後の`data`フォルダーの中にアップロードします。そのままドラッグアンドドロップするか赤枠のアップロードから画像を選ぶだけです。

```python=
# tiny
!python detect.py --weights ./data/weight/yolov3-tiny.tf --tiny --image ./data/meme.jpg # ← `meme.jpg` の箇所を検出したい画像ファイル名にする

# not tiny
# !python detect.py --weights ./data/weight/yolov3.tf --image ./data/meme.jpg

from IPython.display import Image,display_jpeg
display_jpeg(Image('output.jpg'))
```
画像をアップロードしたら、2行目の`meme.jpg`のところをアップロードしたファイル名に変更して、再度実行すればOKです。

```python=
# tiny
!python detect.py --weights ./data/weight/yolov3-tiny.tf --tiny --image ./data/test.jpg # ← `meme.jpg` の箇所を検出したい画像ファイル名にする

# not tiny
# !python detect.py --weights ./data/weight/yolov3.tf --image ./data/meme.jpg

from IPython.display import Image,display_jpeg
display_jpeg(Image('output.jpg'))
```

たとえば、`test.jpg`を対象に実行する場合、上記のように変更します。

## 学習データの前準備
Duration: 10

テストが終わったら次は、任意の物を検出する重みを作成します。そのためには、まずは学習データを用意する必要があります。

物体検出の場合、学習データは「画像」と「バウンディングボックス（正確にはバウンディングボックスの座標位置）」となります。

画像は、みなさんが用意したものを使用します。その画像に「VoTT」というツールを使って検出したい箇所にバウディングボックスをつけていきます（この作業をアノテーションと呼びます）。

### 6.オリジナルデータの解凍

まずは、用意した画像を`original_images`という名前のフォルダーにまとめてください。まとめたら`original_images.zip`とzip形式に圧縮してください。

圧縮したらGoogle Driveの`data`フォルダー内にアップロードします。アップロードしたら「6.オリジナルデータの解凍」を実行します。アップロードしたzipフォルダーを解凍します。

### 7.データの整形

解凍が終わったら「7.データの整形」を行い、416*416サイズに変更します。サイズ変更した画像は`resized_images`に保存されます。

### 8.画像の水増し

次に画像の水増しを行います。
```python
############
volume = 60 # 1枚あたりどれだけ増やすか
img_name = 'img' # 生成された画像のファイル名
############
```
この箇所の数値を変更することで、画像１枚に対して水増しする量を調整できます。上記の例の場合、1枚に対しておよそ60枚複製されることになります。

枚数を決定したら「8.画像の水増し」を実行しましょう。最初は1クラスに対して合計100枚ぐらいになるように設定することをお勧めします。一度簡単に試してみて上手く検出できていたら画像を増やしていく方が効率的です。

また、新たに画像を追加する場合は、`img_name = 'img'`の`img`部分を変更して、水増しされた画像のファイル名が被らないようにしましょう（言ってることがわかりにくい方は、実際に水増しされたあとの画像ファイル名を確認してみるとわかりやすいです。）。

```python
datagen = ImageDataGenerator(
    channel_shift_range=30,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.1,
    )
```
画像の水増しは、ただ複製するだけではなく、少し加工した上で複製を行います。パラメーターについては、すべてが解説されているわけではありませんが[こちら](https://qiita.com/takurooo/items/c06365dd43914c253240)で詳しく解説されています。

ここも調整することで精度が変化するので各自色々と試してみましょう。

### 9.変形した画像を圧縮してダウンロード

データを水増しができたら最後に「9.変形した画像を圧縮してダウンロード」を実行します。実行すると自動でダウンロードされますが、たまに、ポップアップが出現しダウンロードの許可ボタンを押す必要がありますので注意してください。

## 学習データの作成(アノテーション)
Duration: 10

これで画像を用意できたので、次はアノテーションを行っています。

まずは、VoTTを[こちら](https://github.com/Microsoft/VoTT/releases)からダウンロードしてください。

<!-- ![](https://i.imgur.com/pWx2odP.png) -->
![img27](img/img27.png)

自分のOSに合わせて`.dmg`,`.snap`,`.exe`ファイルのどれかをダウンロードしてください。ダウンロードしたらファイルを実行して、VoTTを立ち上げましょう。

<!-- ![](https://i.imgur.com/fN6RNbs.png) -->
![img28](img/img28.png)

立ち上がったら`New Project`を選びます。

<!-- ![](https://i.imgur.com/nBW0G9v.png) -->
![img29](img/img29.png)

プロジェクトの名前などを設定します。

- Display Name: プロジェクトの名前を記入
- Security Token: とくに設定不要
- Source Connection: アノテーションする画像フォルダーを選ぶ
- Target Connection: データの出力先を選ぶ
- Description: とくに設定不要
- Video Settings: 動画のフレーム数を設定、今回は不要
- Tags: タグの作成

Tagsに検出したい画像の名前を追加してください。

Source / Target Connectionは、まずは`Add Connection`から新しく作成する必要があります。

<!-- ![](https://i.imgur.com/szcHprr.png) -->
![img30](img/img30.png)

- Display Name: Connectionimの名前
- Provider: Local File Systemを選ぶ
- Local File System: ローカル内の任意のディレクトリを選ぶ

出力先のフォルダーは、別途`out`のような名前のフォルダーを追加するとわかりやすいです。

<!-- ![](https://i.imgur.com/qnm3aiF.png) -->
![img31](img/img31.png)

このようにダウンロードしてきた`changed_images`をSource　Connectionに設定し、その中に`out`を作ってTarget Connectionにするのがお勧めです。

それぞれ設定できたら`Save Connection`をクリックします。これでアノテーションに入れますが、もう少し設定を変更します。

<!-- ![](https://i.imgur.com/TZgGh3S.png) -->
![img32](img/img32.png)

まずは、左の上から4つ目のアイコンをクリックして、出力形式を上記画像のように変更します。

- Provider: Pascal VOC
- Asset State: Only tagged Assets
- Test / Train Split: 100

変更したら、`Save Export Settings`をクリックします。

<!-- ![](https://i.imgur.com/PzS3Ley.png) -->
![img33](img/img33.png)

次に、左の上から5番目のアイコンをクリックして、上記のように設定します。VoTTの機能でアノテーションを補助してくれます。ここでは、自動でバウンディングボックスをつけるようにしています。不要なバウンディングボックスが多ければ、`Auto Detect`のチェックマークを外してください。

変更したら、`Save Project`をクリックします。

これで設定はお終いです。左の上から2番目をクリックしてアノテーション作業に入ります。

<!-- ![](https://i.imgur.com/JxNBQfN.jpg) -->
![img34](img/img34.jpg)

直感的に操作できるUIですが、簡単に操作説明をします。

1. アノテーション用のボックスを作成
2. 前/次の画像に移動
3. 保存
4. データの出力
5. タグの選択

基本的に1 → 5 → 2の流れで行います。全部アノテーションしたら、4をクリックしてください。

その他以下の機能があります。
- タグは鍵マークをクリックすると固定になり自動でそのタグがつく
- タグの右側にある数字キーを押すとそのタグがつく
- バウンディングボックスは重なっていても問題ない
- タグの左側をクリックした後に、鉛筆ボタンをクリックすると色を選べる（どの色を使っても問題ない）
- Wキーを押すと前の画像に移動
- Sキーを押すと次の画像に移動

以上が、VoTTの使い方です。細かな使い方や機能は各自で調べてもらえればと思います。慣れると単調な作業になり辛いですが、根気よく行うしかありません。

## 学習
Duration: 10

次に学習を実行します。画像の量に応じて、「画像の解凍」「tfrecordへの変換」「学習」それぞれに時間がかかります。

### 10.画像の解凍

アノテーション作業が終わったら、VoTTから必要なデータを出力します。出力ボタンを押すだけで、Target Connectionに設定したフォルダーにデータが出力されます。

<!-- ![](https://i.imgur.com/A8NxHu7.png) -->
![img35](img/img35.png)

右上にこのようなポップが表示されたら無事出力成功です。

<!-- ![](https://i.imgur.com/rW8Cq0P.png) -->
![img36](img/img36.png)

このようなポップが出てきた場合は、失敗です。原因不明でたまに失敗します。
プロジェクトを保存して、一度VoTTを閉じて再度起動して出力ボタンを再度押してください。また、失敗したらこれを繰り返してください。

<!-- ![](https://i.imgur.com/73l2dON.png) -->
![img37](img/img37.png)

Target Connectionには、このようにjsonデータが大量に生成されています。今回必要なのは`~~~~~ -PascalVOC-export`というフォルダーです（変更日など時間でソートをすると見つけやすいです。）。

<!-- ![](https://i.imgur.com/LdrErYw.png) -->
![img38](img/img38.png)

このフォルダーを`train_images`に名前を変えて、zip形式に圧縮します。圧縮したら、Google Driveにアップロードします。アップロード先は、同様に`data`フォルダー内です。

アップロードできたら「10.画像の解凍」を実行してください。

### 11.tfrecordの作成

次に用意したデータを「tfrecord」という形式に変換します。tfrecordとは、tensorflowが推奨する軽量なデータフォーマットです。

大量の学習データを1つにまとめていきます。また、合わせてデータの水増しも行っています。

「11.tfrecordの作成」を実行する前に、9~11行目を変更してください。

- `classes_list`: VoTTで使用したタグ名を記載（順番は関係なし）
- `tf_file_name`: tfrecordのファイル名（分かりやすければなんでもよし）
- `augment_size`: 画像の水増しの量

```python
classes_list = ['rock', 'scisor', 'paper'] # タグ(順番はVoTTと一致していなくても大丈夫)
tf_file_name = 'hand' # ファイル名
augment_size = 1 # 画像の水増し量
```

このような感じで変更してください。
水増し量は、各画像に対して何枚増やすかになります。そのため、数枚程度に留めておいた方が良いと思います。

設定したら「11.tfrecordの作成」を実行してください。

<!-- ![](https://i.imgur.com/6mSVsvk.png) -->
![img39](img/img39.png)

実行すると`data`フォルダに`train`,`val`,`test`の3つのtfrecordが生成されます。ファイル名は、`tf_file_name`で指定した文字列 + `_train`,`_val`,`_test`となります。

それぞれ訓練用、検証用、テスト用です。

### 12.tfrecordの表示

tfrecordを作成できたら、テストとして表示させてみます。「12.tfrecordの表示」を実行する前に、14,15行目を設定します。

```python
#######################
tfrecord_name = 'hand' # tfrecordの名前
tfrecord_kind = 'train' # tfrecotdの種類, train, val, test のどれか
#######################
```

`tfrecord_name`は表示させたいtfrecodのファイル名を、`tfrecord_lind`にはtrain,val,testのどれかを設定します。

設定したら「12.tfrecordの表示」を実行しましょう。

バウンディングボックスのついた画像が表示されると思います。

### 13.学習

これで学習データの準備ができました。次に学習を行っていきます。

「13.学習」を実行する前に、25,26行目を変更してください。

```python
###################################
your_weight_name = 'cola_tea_monster_test'
tfrecord_name = 'cola_tea_monster'
###################################
```
保存する重みのファイル名、学習の際に使用するtfrecordの名前を設定します。設定したら「13.学習」を実行しましょう。画像の量によって学習にかかる時間は変わりますが、100~200枚程度であれば30~60分で終わると思います。

また、ディープラーニングのハイパーパラメーターは、30~34行目で設定できます。

```python
###################################
epochs = 100
batch_size = 16
learning_rate = 1e-3 # 0.001
tiny = True
###################################
```
tinyを`False`にすると通常のYOLOv3の重みで転移学習を行うことができます（もちろん、事前に重みのダウンロードは必要です）。

EaryStoppingを適用しているので、過学習が発生している場合には上記で設定した`epochs`分学習しない場合もあります。

## ラズパイで物体検出（ジャンケンの手を検出）
Duration: 10

学習している間にラズパイでの物体検出を体感してもらいます。こちらで用意したジャンケンの手を検出する重みを使用します。

まずは、ラズパイを起動してターミナルを起動します。
<!-- ![](https://i.imgur.com/TObmSv7.png) -->
![img40](img/img40.png)

ラズパイのホーム画面の左上の赤枠をクリックするか`Ctrl`+`Alt`+`t`のショートカットキーでも起動できます。

起動したら、`cd Desktop`でデスクトップに移動します。

```shell
git clone https://jw-fujibayashi:moyashi314%40tmge@github.com/jw-fujibayashi/yolo-tf2-test.git
```
同じようにGitのプロジェクトをクローンします。クローンしたら`cd yolo-tf2-test`を実行します。

クローンしたら、以下のコマンドを順に入力します。モジュールなどをダウンロードします。

`virtualenv .env`, `source .env/bin/activate`,`pip install -r requirements.txt`

無事ダウンロードできたら、[こちら](https://drive.google.com/open?id=1FvNGTYcpmRwvAkKUUwK4YzLDGmlfUPPs)の重みをダウンロードしてください。3ファイルありますが、3ファイルともダウンロードしてください。

ダウンロードできたら、ラズパイの`yolov2-tf-test`の`data`の`weight`フォルダーに入れます。

```shell
python detect_video.py --video 0 --tiny --weights data/weight/hands.tf
```

最後に上記コマンドを実行すると、リアルタイム物体検出のプログラムが実行します。ラズパイにカメラをつけた状態で実行する必要があるので注意してください。

プログラムを終了する際には、`q`キーを押してください。

## テスト＆ラズパイで物体検出（オリジナルデータ）
Duration: 10

### 14.物体検出

無事に学習が終わったら、でき上がった重みを使って物体検出を行っていきます。まずは、Goolge Colaboratory上でテストをしてみます。

「14.物体検出」を実行する前に、16,17行目を変更してください。

```python
###################################
your_weight_name = ''
tfrecord_name = ''
###################################
```

`your_weight_name`にさっきの学習でできた重みファイルの名前を、`tfrecord_name`に使用したtfrecordの名前を設定します（test用のtfrecordを使用します。`_test`は不要です。）。

設定したら実行してください。上手く検出できたでしょうか？納得のいかない精度の場合は、学習データを増やしたりしてみましょう。（学習のコツは次のスライドでまとめています。）

それでは、最後にラズパイでリアルタイム物体検出をしてみましょう。まずは、作成した「重みファイル3つ（`data`>`weight`フォルダーにあります。）」と「`classes.txt`（`data`フォルダーにあります。）」をダウンロードします。

重みファイルは、`data`の中の`weight`フォルダーに移動させましょう。
`classes.txt`は、`data`フォルダーに移動させます。既にある`classes.txt`は上書きしても大丈夫です。

移動させたらターミナルを開いて、`cd Desktop/yolo-tf2-master`を実行して

```shell
python detect_video.py --video 0 --tiny --weights data/weight/{各自の重みファイル名}.tf
```

を実行します。重みファイルは、`~~~.tf.index`,`~~~.data-~~~`などのようになっていると思いますが、`.tf`までで指定すれば大丈夫です。

## 学習のコツ

自分が試してみて上手くいったと感じたコツを以下にまとめます。論理的な部分もない場合もありますので、軽く参考にしてもらえればと思います。

また、一般的に高精度の物体検出を行うには、1クラスあたり1000枚の画像が必要されると言われています（単純に水増しして1000枚に達すれば良いわけではありません）。

それと比べると今回は圧倒的にデータ量が足りていなく、かつtinyで学習しているので精度に限界があることは念頭に置いておいてください。

### 学習データは量よりも種類

似たような画像をたくさん用意するよりも、バリエーションを増やした方が精度は向上します。たとえば、背景を変えるだけでも大丈夫です。また、遠くから撮った画像なども用意するとより精度は向上します。

その他、角度/光加減などを変えた画像を用意するとなお良くなります。

### 転移学習を活かす

今回、転移学習に使っている重みは「coco」と呼ばれるデータセットを使用して学習された重みです。
- person
- bicycle
- car
- motorbike
- aeroplane
- bus
- train
- truck
- boat
- traffic light
- fire hydrant
- stop sign
- parking meter
- bench
- bird
- cat
- dog
- horse
- sheep
- cow
- elephant
- bear
- zebra
- giraffe
- backpack
- umbrella
- handbag
- tie
- suitcase
- frisbee
- skis
- snowboard
- sports ball
- kite
- baseball bat
- baseball glove
- skateboard
- surfboard
- tennis racket
- bottle
- wine glass
- cup
- fork
- knife
- spoon
- bowl
- banana
- apple
- sandwich
- orange
- broccoli
- carrot
- hot dog
- pizza
- donut
- cake
- chair
- sofa
- pottedplant
- bed
- diningtable
- toilet
- tvmonitor
- laptop
- mouse
- remote
- keyboard
- cell phone
- microwave
- oven
- toaster
- sink
- refrigerator
- book
- clock
- vase
- scissors
- teddy bear
- hair drier
- toothbrush

たとえば、ペットボトルのお茶とコカコーラの物体検出は、比較的少ない画像データで精度が高くなりました。おそらく、bottoleが事前に学習されていたからだと思います。

### 推論時と同じ環境のデータを用意する

どんな時でも物体検出を高精度で行う重みを作成するのは大変です。そのため、AIが推論を上手く行えるように寄り添ってあげる必要があります。

その1つとして、学習データを推論時に映る風景を合わせてあげると精度が上ります。

たとえば、防犯カメラのような定点カメラとして使用するのであれば、その風景でのみ物体検出できれば良いので学習データを合わせてあげると良いと思います。